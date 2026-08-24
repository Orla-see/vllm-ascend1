# mypy: ignore-errors
"""DSD + DP: align verify K_prev across DP ranks.

Independent of balance scheduling. Monkey-patches:
  - BalanceScheduler: cap spec_token_ids before the running loop.
  - DPEngineCoreProc._has_global_unfinished_reqs: gloo the DP-wide min verify
    K_prev after the cross-rank sync point in the busy loop.

Why _has_global_unfinished_reqs (not _process_engine_step): the DP busy loop
calls _has_global_unfinished_reqs AFTER the idle `continue` gate, so all
non-idle ranks reach it every step -> the collective never starves -> no
deadlock. _process_engine_step is called BEFORE the idle gate; at wave
boundaries one rank can be blocked in _process_input_queue while another
reaches it, deadlocking the all-reduce. See patch_balance_schedule.py for the
full rationale.

1-round lag: values gathered in step N cap K_prev in step N+1's schedule
(_has_global_unfinished_reqs runs after schedule+execute). Harmless in FULL
mode -- the 2-D catalog has padding/PIECEWISE fallback for the transient
mismatch.

Mechanism: verify K_prev = len(spec_token_ids). Cap it to the DP-wide min so
all ranks share one K_prev -> same ql -> same graph cell -> cross-DP MoE
all-to-all replays consistently.

The drafter is eager (enforce_eager) and its per-step forward count
(_step_k) is threaded from scheduler_output.num_spec_tokens_to_schedule,
which is decided per-rank from the LOCAL running count via
dynamic_sd_lookup.  At a K-switch transition DP ranks pick different
_step_k -> different cross-DP EP All2All counts -> HCCL deadlock.  So the
draft _step_k is ALSO aligned here: a raw-K gloo gather + OVERRIDE
(1-round lag, mirroring verify-K_prev).  The raw (pre-override) K is
gathered, not the post-override value, so the MIN can still see a rank
whose local K dropped -- otherwise the system never transitions down.
"""

import torch
import torch.distributed as dist
from vllm.v1.engine.core import DPEngineCoreProc

from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler


def _dsd_gather_kprev(self, dp_group):
    """gloo the DP-wide min verify K_prev and stash it."""
    spec_lens = [len(r.spec_token_ids) for r in self.running]
    if spec_lens and len(set(spec_lens)) == 1:
        local_k = spec_lens[0]
        self._dsd_uniform = True
    else:
        local_k = self.num_spec_tokens
        self._dsd_uniform = False
    k_tensor = torch.tensor([local_k], dtype=torch.int, device="cpu")
    dist.all_reduce(k_tensor, op=dist.ReduceOp.MIN, group=dp_group)
    self._dsd_min_kprev = int(k_tensor.item())


def _dsd_gather_stepk(self, dp_group):
    """gloo the DP-wide min raw draft _step_k and stash it.

    Gathers the RAW (pre-override) num_spec_tokens_to_schedule stashed by
    _dsd_schedule, NOT the post-override value: if the post-override K
    were gathered, a rank whose local K dropped would be overridden UP
    before generating spec_token_ids, so the MIN would never see the
    smaller K and the system would stick at the old K forever.
    """
    local_k = getattr(self, "_dsd_raw_stepk", None)
    if local_k is None:
        # First step / dummy step: contribute maxK (neutral) so this rank
        # does not pull the MIN down.
        local_k = self.num_spec_tokens
    k_tensor = torch.tensor([local_k], dtype=torch.int, device="cpu")
    dist.all_reduce(k_tensor, op=dist.ReduceOp.MIN, group=dp_group)
    self._dsd_min_stepk = int(k_tensor.item())


BalanceScheduler._dsd_min_kprev = None
BalanceScheduler._dsd_uniform = False
BalanceScheduler.dsd_gather_kprev = _dsd_gather_kprev

BalanceScheduler._dsd_min_stepk = None
BalanceScheduler._dsd_raw_stepk = None
BalanceScheduler.dsd_gather_stepk = _dsd_gather_stepk

_orig_schedule = BalanceScheduler.schedule


def _dsd_schedule(self, throttle_prefills: bool = False):
    # Verify K_prev cap (existing): runs BEFORE schedule so the capped
    # spec_token_ids flows into scheduled_spec_decode_tokens upstream.
    if (not self._balance_enabled
            and getattr(self, "_dsd_uniform", False)
            and getattr(self, "_dsd_min_kprev", None) is not None):
        for r in self.running:
            if len(r.spec_token_ids) > self._dsd_min_kprev:
                r.spec_token_ids = r.spec_token_ids[:self._dsd_min_kprev]
    scheduler_output = _orig_schedule(self, throttle_prefills)
    # Draft _step_k alignment: stash the raw per-rank K for the gather,
    # then OVERRIDE (not cap) with the lagged DP-wide MIN so every rank
    # runs the same number of draft forwards -> same All2All count.
    if scheduler_output is not None:
        raw_k = scheduler_output.num_spec_tokens_to_schedule
        # raw_k==0 means keepalive: skip the override and stash maxK
        # (neutral) so a keepalive rank does not force the whole DP
        # group to keepalive via the MIN.
        self._dsd_raw_stepk = raw_k if raw_k > 0 else self.num_spec_tokens
        min_k = getattr(self, "_dsd_min_stepk", None)
        if min_k is not None and raw_k > 0 and raw_k != min_k:
            scheduler_output.num_spec_tokens_to_schedule = min_k
    return scheduler_output


BalanceScheduler.schedule = _dsd_schedule

# Hook gloo into DPEngineCoreProc._has_global_unfinished_reqs -- the safe
# cross-rank sync point in the DP busy loop (after the idle continue gate).
# All non-idle ranks reach it every step, so the all_reduce never starves.
# When BalanceDPEngineCoreProc is active, super()._has_global_unfinished_reqs
# calls this patched method, so DSD gloo runs before balance_gather -- both
# in the same safe region.
_orig_hgur = DPEngineCoreProc._has_global_unfinished_reqs


def _dsd_has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
    result = _orig_hgur(self, local_unfinished)
    if (getattr(self, "dp_group", None) is not None
            and self.scheduler is not None
            and getattr(self.scheduler, "num_spec_tokens", 0) > 0):
        self.scheduler.dsd_gather_kprev(self.dp_group)
        self.scheduler.dsd_gather_stepk(self.dp_group)
    return result


DPEngineCoreProc._has_global_unfinished_reqs = _dsd_has_global_unfinished_reqs
