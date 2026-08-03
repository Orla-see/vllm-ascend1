# mypy: ignore-errors
"""DSD FULL graph + DP: align verify K_prev across DP ranks.

Independent of balance scheduling. Monkey-patches:
  - the always-on ``BalanceScheduler`` (the scheduler class installed by
    patch_balance_schedule): cap ``spec_token_ids`` before the running loop.
  - ``EngineCoreProc._process_engine_step`` (engine core): gloo the DP-wide
    min verify K_prev *before* schedule() runs.

Why before schedule(): the cap must affect the CURRENT round's verify batch
(the running loop sizes verify from spec_token_ids). Putting the gloo in
_update_after_schedule (after schedule) would only affect next round (1-round
lag, same flaw as the old runner-side Part A).

Why _process_engine_step for the gloo: every DP rank calls it every busy-loop
iteration, including idle ranks (which then do execute_dummy_batch). So the
collective never starves on an idle rank -> no deadlock. (A gloo inside
schedule() would deadlock whenever a rank is idle, since idle ranks skip
schedule().)

Mechanism: verify K_prev = len(spec_token_ids) (= num_spec_tokens_to_schedule
of the previous round, set as [-1]*K placeholders in _update_after_schedule).
Cap it to the DP-wide min so all ranks share one K_prev -> same ql -> same
graph cell -> the captured cross-DP MoE all-to-all can replay consistently.

Only the verify (main model) needs alignment; the drafter is eager (fixed
maxK step), so it is not aligned here.
"""

import torch
import torch.distributed as dist
from vllm.v1.engine.core import EngineCoreProc

from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler


def _dsd_gather_kprev(self, dp_group):
    """gloo the DP-wide min verify K_prev (= len(spec_token_ids)) and stash it.

    For uniform-decode batches (graph-eligible) the local value is the shared
    spec_token_ids length; for mixed/prefill/idle batches it is the neutral
    maxK so the rank still participates in the collective without dragging
    the min.
    """
    spec_lens = [len(r.spec_token_ids) for r in self.running]
    # K_prev=0 (plain decode, K=0 tier) counts as uniform too -- it must
    # participate in the min so a K=0 rank and a K>0 rank align (else both go
    # FULL on different ql cells -> cross-DP all-to-all mismatch).
    if spec_lens and len(set(spec_lens)) == 1:
        local_k = spec_lens[0]
        self._dsd_uniform = True
    else:
        local_k = self.num_spec_tokens  # neutral: mixed / prefill / idle
        self._dsd_uniform = False
    k_tensor = torch.tensor([local_k], dtype=torch.int, device="cpu")
    dist.all_reduce(k_tensor, op=dist.ReduceOp.MIN, group=dp_group)
    self._dsd_min_kprev = int(k_tensor.item())


# Attach the gather + default state to the always-on scheduler class.
BalanceScheduler._dsd_min_kprev = None
BalanceScheduler._dsd_uniform = False
BalanceScheduler.dsd_gather_kprev = _dsd_gather_kprev


# Wrap schedule(): cap spec_token_ids to the DP min BEFORE the running loop.
_orig_schedule = BalanceScheduler.schedule


def _dsd_schedule(self, throttle_prefills: bool = False):
    if (not self._balance_enabled
            and getattr(self, "_dsd_uniform", False)
            and getattr(self, "_dsd_min_kprev", None) is not None):
        # spec_token_ids is [-1]*K placeholders; trim to [-1]*min so the
        # running loop sees a uniform, DP-aligned verify K_prev.
        for r in self.running:
            if len(r.spec_token_ids) > self._dsd_min_kprev:
                r.spec_token_ids = r.spec_token_ids[:self._dsd_min_kprev]
    return _orig_schedule(self, throttle_prefills)


BalanceScheduler.schedule = _dsd_schedule


# Wrap _process_engine_step(): gloo min K_prev before schedule() (current
# round). Gated to DP (dp_group only exists on DPEngineCoreProc) + spec decode
# (skip the per-step collective when speculative decoding is off).
_orig_pes = EngineCoreProc._process_engine_step


def _dsd_process_engine_step(self):
    if (getattr(self, "dp_group", None) is not None
            and self.scheduler is not None
            and getattr(self.scheduler, "num_spec_tokens", 0) > 0):
        self.scheduler.dsd_gather_kprev(self.dp_group)
    return _orig_pes(self)


EngineCoreProc._process_engine_step = _dsd_process_engine_step
