# mypy: ignore-errors
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Tier-aware lookahead reservation for DSD schedules.

num_lookahead_tokens is static (maxK + spec-mode extra), so K=0 tiers still
reserve maxK+1 draft slots per request, evicting prefix-cache blocks. Set the
per-step lookahead to the current tier's K (0 for K=0) around each schedule()
call. Active only when VLLM_ASCEND_DSD_TIER_LOOKAHEAD=1.
"""

import os

import vllm.v1.core.sched.scheduler as _sched_mod
from vllm.v1.core.sched.scheduler import Scheduler

_ENABLED = int(os.environ.get("VLLM_ASCEND_DSD_TIER_LOOKAHEAD", "0"))


def _tier_lookahead(self) -> int:
    if self.dynamic_sd_lookup is None:
        return self.num_lookahead_tokens
    bs = min(max(len(self.running), 1), len(self.dynamic_sd_lookup) - 1)
    k = self.dynamic_sd_lookup[bs]
    if k == 0:
        return 0
    return k + (self.num_lookahead_tokens - self.num_spec_tokens)


def _make_schedule_wrapper(orig_schedule):
    def _schedule(self, *args, **kwargs):
        if not _ENABLED:
            return orig_schedule(self, *args, **kwargs)
        orig = self.num_lookahead_tokens
        self.num_lookahead_tokens = _tier_lookahead(self)
        try:
            return orig_schedule(self, *args, **kwargs)
        finally:
            self.num_lookahead_tokens = orig

    _schedule._dsd_tier_lookahead = True
    return _schedule


def _wrap_schedule(cls):
    if getattr(cls.schedule, "_dsd_tier_lookahead", False):
        return
    cls.schedule = _make_schedule_wrapper(cls.schedule)


_wrap_schedule(Scheduler)
# BalanceScheduler (installed at the module name) overrides schedule; wrap it too.
if _sched_mod.Scheduler is not Scheduler:
    _wrap_schedule(_sched_mod.Scheduler)
