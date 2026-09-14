# mypy: ignore-errors
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Dynamic-K mamba block accounting for DSD.

MambaManager sizes the per-request speculative state tail with the boot-time
top-level num_speculative_tokens (= max K of the DSD table), so a tier with a
shallower K never admits more requests than the max-K static deployment.
Charge the tail with the current tier K instead: a scheduler wrapper publishes
dynamic_sd_lookup[len(running)] before each schedule() call. The tier K is
non-increasing in batch size, so a stale value can only over-charge, never
under-charge. Active only when VLLM_ASCEND_DSD_MAMBA_K=1.
"""

import os
from collections.abc import Sequence

import vllm.v1.core.sched.scheduler as _sched_mod
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

from vllm_ascend import dsd_mamba_k
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager

_ENABLED = int(os.environ.get("VLLM_ASCEND_DSD_MAMBA_K", "0"))

_orig_get_num_blocks_to_allocate = AscendMambaManager.get_num_blocks_to_allocate
_orig_allocate_new_blocks = AscendMambaManager.allocate_new_blocks


def _dyn_k(self) -> int:
    k = dsd_mamba_k.current_k() if _ENABLED else None
    if k is None or not (0 <= k < self.num_speculative_blocks):
        return self.num_speculative_blocks
    return k


def _asc_get_num_blocks_to_allocate(
    self,
    request_id: str,
    num_tokens: int,
    new_computed_blocks: Sequence,
    total_computed_tokens: int,
    num_local_computed_tokens: int | None = None,
    num_tokens_main_model: int | None = None,
    apply_admission_cap: bool = False,
) -> int:
    if num_tokens_main_model is None:
        assert num_local_computed_tokens is not None
        num_tokens_main_model = num_local_computed_tokens
    k = _dyn_k(self)
    if self.mamba_cache_mode != "align" and k != self.num_speculative_blocks:
        if k > 0:
            num_tokens += self.block_size * k
        num_new_blocks = SingleTypeKVCacheManager.get_num_blocks_to_allocate(
            self,
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )
    else:
        num_new_blocks = _orig_get_num_blocks_to_allocate(
            self,
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )
    local_hit_tokens = len(new_computed_blocks) * self.block_size
    has_external_tokens = total_computed_tokens > local_hit_tokens
    has_new_scheduled_tokens = num_tokens_main_model > total_computed_tokens
    if has_external_tokens and has_new_scheduled_tokens:
        num_new_blocks += 1
    return num_new_blocks


def _asc_allocate_new_blocks(
    self,
    request_id: str,
    num_tokens: int,
    num_tokens_main_model: int,
):
    k = _dyn_k(self)
    if self.mamba_cache_mode != "align" and k != self.num_speculative_blocks:
        if k > 0:
            num_tokens += self.block_size * k
        req_blocks = self.req_to_blocks.get(request_id)
        if req_blocks:
            num_required_blocks = -(-num_tokens // self.block_size)
            excess = len(req_blocks) - num_required_blocks
            if excess > 0:
                freed = req_blocks[num_required_blocks:]
                del req_blocks[num_required_blocks:]
                self.block_pool.free_blocks(freed)
        return SingleTypeKVCacheManager.allocate_new_blocks(
            self, request_id, num_tokens, num_tokens_main_model
        )
    return _orig_allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model)


def _make_schedule_wrapper(orig_schedule):
    def _schedule(self, *args, **kwargs):
        if _ENABLED and self.dynamic_sd_lookup is not None:
            bs = min(max(len(self.running), 1), len(self.dynamic_sd_lookup) - 1)
            dsd_mamba_k.set_k(self.dynamic_sd_lookup[bs])
        else:
            dsd_mamba_k.set_k(None)
        return orig_schedule(self, *args, **kwargs)

    _schedule._dsd_mamba_k = True
    return _schedule


def _wrap_schedule(cls):
    if getattr(cls.schedule, "_dsd_mamba_k", False):
        return
    cls.schedule = _make_schedule_wrapper(cls.schedule)


AscendMambaManager.get_num_blocks_to_allocate = _asc_get_num_blocks_to_allocate
AscendMambaManager.allocate_new_blocks = _asc_allocate_new_blocks

_wrap_schedule(Scheduler)
if _sched_mod.Scheduler is not Scheduler:
    _wrap_schedule(_sched_mod.Scheduler)
