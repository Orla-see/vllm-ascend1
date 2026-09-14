# SPDX-License-Identifier: Apache-2.0
"""UT for DSD dynamic-K mamba block accounting (patch_dsd_mamba_k)."""

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import MambaSpec

import vllm_ascend.dsd_mamba_k as dk
import vllm_ascend.patch.platform.patch_dsd_mamba_k as pm


class FakeBlockPool:

    def __init__(self):
        self.freed = 0

    def get_new_blocks(self, count):
        return [object() for _ in range(count)]

    def free_blocks(self, blocks):
        self.freed += len(blocks)


def make_mgr(mode="none", k_static=7, held=0):
    m = object.__new__(pm.AscendMambaManager)
    m.mamba_cache_mode = mode
    m.num_speculative_blocks = k_static
    m.block_size = 16384
    m.kv_cache_spec = MambaSpec(
        shapes=((2048, 10),), dtypes=(torch.bfloat16,), block_size=16384)
    m.req_to_blocks = {"r1": [object() for _ in range(held)]}
    m.num_cached_block = {"r1"}
    m.block_pool = FakeBlockPool()
    m._record_new_block_ids = False
    m._max_admission_blocks_per_request = None
    m.new_block_ids = []
    m._partial_hit_reqs = {}
    return m


def admission_count(m):
    return m.get_num_blocks_to_allocate(
        "r1", 3072, [], 0,
        num_local_computed_tokens=0,
        num_tokens_main_model=3072,
        apply_admission_cap=True,
    )


def test_dynamic_k_admission_charges_tier_k(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 1)
    dk.set_k(3)
    m = make_mgr()
    # 3072 tokens -> 1 base block, + 3 spec-state tail blocks
    assert admission_count(m) == 4


def test_k0_admission_charges_base_block_only(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 1)
    dk.set_k(0)
    m = make_mgr()
    assert admission_count(m) == 1


def test_static_fallback_when_k_unpublished(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 1)
    dk.set_k(None)
    m = make_mgr()
    assert admission_count(m) == 8


def test_k_above_static_clamps_to_static(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 1)
    dk.set_k(9)
    m = make_mgr()
    assert admission_count(m) == 8


def test_k_equal_static_takes_orig_path(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 1)
    dk.set_k(7)
    m = make_mgr()
    monkeypatch.setattr(pm, "_orig_get_num_blocks_to_allocate",
                        lambda self, *a, **kw: 99)
    assert admission_count(m) == 99


def test_align_mode_delegates_orig(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 1)
    dk.set_k(3)
    m = make_mgr(mode="align")
    monkeypatch.setattr(pm, "_orig_get_num_blocks_to_allocate",
                        lambda self, *a, **kw: 99)
    assert admission_count(m) == 99


def test_disabled_env_uses_static(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 0)
    dk.set_k(3)
    m = make_mgr()
    assert admission_count(m) == 8


def test_upflip_growth_appends_blocks(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 1)
    dk.set_k(7)
    m = make_mgr(held=4)
    out = m.allocate_new_blocks("r1", 8, 8)
    assert len(out) == 4
    assert len(m.req_to_blocks["r1"]) == 8


def test_downflip_shrink_frees_tail(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 1)
    dk.set_k(3)
    m = make_mgr(held=8)
    out = m.allocate_new_blocks("r1", 4, 4)
    assert len(out) == 0
    assert m.block_pool.freed == 4
    assert len(m.req_to_blocks["r1"]) == 4


class FakeSched:

    dynamic_sd_lookup = [7] * 131 + [3] * 200

    def __init__(self, nrun=0):
        self.running = [object()] * nrun

    def schedule(self):
        return "out"


def test_scheduler_wrapper_publishes_tier_k(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 1)
    pm._wrap_schedule(FakeSched)
    dk.set_k(None)
    FakeSched(130).schedule()
    assert dk.current_k() == 7
    FakeSched(131).schedule()
    assert dk.current_k() == 3


def test_scheduler_wrapper_without_lookup_resets(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 1)

    class NoLookupSched:
        dynamic_sd_lookup = None
        running = []

        def schedule(self):
            return "out"

    pm._wrap_schedule(NoLookupSched)
    dk.set_k(7)
    NoLookupSched().schedule()
    assert dk.current_k() is None


def test_wrapper_idempotent(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 1)
    pm._wrap_schedule(FakeSched)
    pm._wrap_schedule(FakeSched)
    dk.set_k(None)
    FakeSched(1).schedule()
    assert dk.current_k() == 7


def test_disabled_wrapper_keeps_k_none(monkeypatch):
    monkeypatch.setattr(pm, "_ENABLED", 0)
    dk.set_k(None)

    class Sched(SimpleNamespace):
        pass

    s = Sched(dynamic_sd_lookup=[5] * 10, running=[object()] * 2)

    def orig_schedule(self):
        return "out"

    wrapped = pm._make_schedule_wrapper(orig_schedule)
    wrapped(s)
    assert dk.current_k() is None
