# SPDX-License-Identifier: Apache-2.0
"""UT for the DSD 2-D FULL cudagraph catalog and dispatch (patch_cudagraph)."""

from types import SimpleNamespace

import pytest
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

import vllm_ascend.patch.worker.patch_cudagraph as pc


def make_dispatcher(
    monkeypatch,
    table,
    num_spec=None,
    max_num_seqs=256,
    max_cg=4096,
    tp=1,
    mode=CUDAGraphMode.FULL,
):
    monkeypatch.setattr(pc, "enable_sp", lambda *a, **kw: False)
    if num_spec is None:
        num_spec = max(k for _, _, k in table) if table else 0
    spec = SimpleNamespace(
        num_speculative_tokens_per_batch_size=table,
        num_speculative_tokens=num_spec,
        uses_dynamic_speculative_decoding=lambda: table is not None,
    )
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=spec,
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
            compilation_config=SimpleNamespace(max_cudagraph_capture_size=max_cg),
            parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        ),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=max_cg),
        cudagraph_mode=mode,
        uniform_decode_query_len=8,
        _bs_to_padded_graph_size={nt: nt for nt in range(1, 4096)},
        _dsd_num_reqs=None,
        _dsd_bs_by_ql={},
        cudagraph_keys={
            CUDAGraphMode.FULL: set(),
            CUDAGraphMode.PIECEWISE: set(),
        },
    )


def dispatch(d, num_tokens, uniform_decode=True):
    return pc._create_padded_batch_descriptor(d, num_tokens, uniform_decode, False)


# ---- catalog build path: exact (num_tokens, num_reqs) cell ----


def test_catalog_build_emits_exact_cell(monkeypatch):
    d = make_dispatcher(monkeypatch, [[1, 16, 3]])
    desc = pc._create_padded_batch_descriptor(d, 8, True, False, num_reqs=2)
    assert desc == BatchDescriptor(num_tokens=8, num_reqs=2, uniform=True)


def test_catalog_build_clamps_num_reqs_to_max_num_seqs(monkeypatch):
    d = make_dispatcher(monkeypatch, [[1, 16, 3]], max_num_seqs=1)
    desc = pc._create_padded_batch_descriptor(d, 8, True, False, num_reqs=2)
    assert desc.num_reqs == 1


# ---- dispatch path: 2-D keys ----


def _seed_two_ql_cells(d):
    d.cudagraph_keys[CUDAGraphMode.FULL] = {
        BatchDescriptor(num_tokens=8, num_reqs=2, uniform=True),
        BatchDescriptor(num_tokens=8, num_reqs=1, uniform=True),
    }
    d._dsd_bs_by_ql = {4: [1, 2], 8: [1]}


def test_same_num_tokens_distinct_num_reqs_get_distinct_keys(monkeypatch):
    d = make_dispatcher(monkeypatch, [[1, 16, 3]])
    _seed_two_ql_cells(d)
    d._dsd_num_reqs = 2
    desc2 = dispatch(d, 8)
    d._dsd_num_reqs = 1
    desc1 = dispatch(d, 8)
    assert desc2 == BatchDescriptor(num_tokens=8, num_reqs=2, uniform=True)
    assert desc1 == BatchDescriptor(num_tokens=8, num_reqs=1, uniform=True)
    assert desc2 != desc1


def test_dispatch_is_deterministic(monkeypatch):
    d = make_dispatcher(monkeypatch, [[1, 16, 3]])
    _seed_two_ql_cells(d)
    d._dsd_num_reqs = 2
    assert dispatch(d, 8) == dispatch(d, 8)


def test_offgrid_bs_pads_up_to_next_cell(monkeypatch):
    d = make_dispatcher(monkeypatch, [[1, 16, 3]])
    d.cudagraph_keys[CUDAGraphMode.FULL] = {
        BatchDescriptor(num_tokens=32, num_reqs=8, uniform=True),
        BatchDescriptor(num_tokens=64, num_reqs=16, uniform=True),
    }
    d._dsd_bs_by_ql = {4: [8, 16]}
    d._dsd_num_reqs = 9
    desc = dispatch(d, 36)
    assert desc == BatchDescriptor(num_tokens=64, num_reqs=16, uniform=True)


def test_mixed_batch_falls_back_to_piecewise_path(monkeypatch):
    d = make_dispatcher(monkeypatch, [[1, 16, 3]], max_num_seqs=256)
    _seed_two_ql_cells(d)
    d._dsd_num_reqs = 2
    desc = dispatch(d, 36, uniform_decode=False)
    assert desc.uniform is False
    assert desc.num_reqs == min(36, 256)


def test_nondsd_uses_upstream_1d_path(monkeypatch):
    d = make_dispatcher(
        monkeypatch, None, mode=CUDAGraphMode.FULL_AND_PIECEWISE)
    d._dsd_num_reqs = 2
    desc = dispatch(d, 32)
    assert desc == BatchDescriptor(num_tokens=32, num_reqs=4, uniform=True)


# ---- 2-D catalog generation from the DSD table ----


def test_2d_cells_three_tier_table(monkeypatch):
    d = make_dispatcher(
        monkeypatch, [[1, 50, 5], [51, 100, 3], [101, 200, 1]],
        max_num_seqs=200, max_cg=1536)
    cells = set(pc._dsd_2d_cells(d, 1))
    # tier1: ks {5,3,1} x bs {8..48 step 8, 50}; tier2: {3,1} x {56..96, 100};
    # tier3: {1} x {104..192 step 8, 200}
    assert len(cells) == 7 * 3 + 7 * 2 + 13
    assert (300, 50) in cells  # tier1 own K=5 at tier end
    assert (400, 100) in cells  # tier2 own K=3 at tier end
    assert (400, 200) in cells  # tier3 own K=1 at tier end
    assert (16, 8) in cells  # tier1 lowest reachable K=1
    for nt, nr in cells:
        assert nt % nr == 0
        assert nt // nr in (2, 4, 6)  # qlen of K in {1, 3, 5}
        assert nt <= 1536
        assert 1 <= nr <= 200


def test_2d_cells_k0_tier_captures_ql1_and_ql2(monkeypatch):
    d = make_dispatcher(
        monkeypatch, [[1, 64, 7], [65, 192, 3], [193, 256, 0]],
        max_num_seqs=256, max_cg=2048)
    cells = set(pc._dsd_2d_cells(d, 1))
    # K0 tier bs grid {200..256 step 8}: ql=1 (target decode) + ql=2 (keep-alive)
    for bs in (200, 256):
        assert (bs, bs) in cells
        assert (2 * bs, bs) in cells
    assert sum(1 for _, nr in cells if nr >= 200) == 16
    assert not any(193 <= nr <= 199 for _, nr in cells)
    # K=0 reachable from the K7 tier as K_prev: plain-decode cell at bs=8
    assert (8, 8) in cells


def test_2d_cells_drops_cells_above_max_capture_size(monkeypatch):
    d = make_dispatcher(monkeypatch, [[1, 64, 7]], max_cg=100)
    assert set(pc._dsd_2d_cells(d, 1)) == {(64, 8)}


def test_2d_cells_clamps_bs_to_max_num_seqs(monkeypatch):
    d = make_dispatcher(monkeypatch, [[1, 64, 7]], max_num_seqs=32, max_cg=2048)
    assert set(pc._dsd_2d_cells(d, 1)) == {
        (64, 8), (128, 16), (192, 24), (256, 32)}


def _make_replace_target(monkeypatch, table, max_num_seqs, max_cg):
    d = make_dispatcher(monkeypatch, table, max_num_seqs=max_num_seqs, max_cg=max_cg)
    recorded = []
    d._get_lora_cases = lambda: [0]

    def _add_key(mode, desc):
        recorded.append(desc)
        d.cudagraph_keys[mode].add(desc)

    d.add_cudagraph_key = _add_key
    d._create_padded_batch_descriptor = (
        lambda nt, uni, hl, nal, num_reqs=None:
        BatchDescriptor(num_tokens=nt, num_reqs=num_reqs, uniform=True))
    return d, recorded


def test_replace_full_keys_rejects_tier_above_capture_range(monkeypatch):
    d, _ = _make_replace_target(
        monkeypatch, [[1, 256, 7]], max_num_seqs=256, max_cg=1536)
    with pytest.raises(AssertionError, match="max_cudagraph_capture_size"):
        pc._replace_full_keys_with_dsd_2d_catalog(d, 1)


def test_replace_full_keys_swaps_in_2d_catalog(monkeypatch):
    d, recorded = _make_replace_target(
        monkeypatch, [[1, 16, 3]], max_num_seqs=16, max_cg=64)
    d.cudagraph_keys[CUDAGraphMode.PIECEWISE].add("pw")
    pc._replace_full_keys_with_dsd_2d_catalog(d, 1)
    assert d.cudagraph_keys[CUDAGraphMode.FULL] == set(recorded)
    assert d.cudagraph_keys[CUDAGraphMode.PIECEWISE] == {"pw"}
    assert d._dsd_bs_by_ql  # dispatch padding index built


# ---- FULL mode is not downgraded for DSD ----


def test_dsd_full_mode_override_disabled():
    from vllm.config.vllm import VllmConfig
    assert VllmConfig._maybe_override_dynamic_sd_cudagraph_mode(
        SimpleNamespace()) is None


# ---- tier lookup contract used by tier-switch E2E ----


def test_lookup_inclusive_boundaries():
    table = [[1, 50, 5], [51, 100, 3], [101, 200, 1]]
    lookup = build_dynamic_sd_schedule_lookup(table, 200, 5)
    assert lookup[1] == 5 and lookup[50] == 5
    assert lookup[51] == 3 and lookup[100] == 3
    assert lookup[101] == 1 and lookup[200] == 1


def test_lookup_gap_carries_previous_k():
    lookup = build_dynamic_sd_schedule_lookup([[1, 16, 3], [32, 128, 2]], 128, 8)
    assert lookup[16] == 3
    assert lookup[17] == 3 and lookup[31] == 3
    assert lookup[32] == 2 and lookup[128] == 2


def test_lookup_clamps_to_num_speculative_tokens():
    lookup = build_dynamic_sd_schedule_lookup([[1, 10, 9]], 10, 5)
    assert all(k == 5 for k in lookup[1:])
