# SPDX-License-Identifier: Apache-2.0
"""UT for DSD step_k resolution and the K=0 short-circuit (_propose head)."""

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.dsd_probe as probe
import vllm_ascend.spec_decode.llm_base_proposer as lb
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def make_proposer(method="dflash"):
    p = object.__new__(AscendSpecDecodeBaseProposer)
    p.device = torch.device("cpu")
    p.method = method
    p.get_model = lambda: object()
    return p


def call_propose(p, sk):
    cam = SimpleNamespace(
        batch_size=lambda: 4,
        query_start_loc=torch.tensor([0, 4]),
    )
    return p._propose(
        target_token_ids=torch.zeros(8, dtype=torch.long),
        target_positions=torch.zeros(8, dtype=torch.long),
        target_hidden_states=torch.zeros(8, 4),
        next_token_ids=torch.zeros(4, dtype=torch.long),
        token_indices_to_sample=None,
        common_attn_metadata=cam,
        target_model_batch_desc=None,
        sampling_metadata=None,
        scheduler_output=SimpleNamespace(num_spec_tokens_to_schedule=sk),
    )


def recorded_k():
    return probe._pending.get("sk"), probe._pending.get("pk")


def test_k0_skip_returns_zero_draft_tokens(monkeypatch):
    monkeypatch.setattr(lb, "_dsd_k0_skip", 1)
    monkeypatch.setattr(probe, "ENABLED", True)
    probe._pending.clear()
    p = make_proposer()
    out = call_propose(p, sk=0)
    assert out.shape == (4, 0)
    assert recorded_k() == (0, 1)


def test_k0_keepalive_floors_step_k_to_one(monkeypatch):
    monkeypatch.setattr(lb, "_dsd_k0_skip", 0)
    monkeypatch.setattr(probe, "ENABLED", True)
    probe._pending.clear()
    p = make_proposer()
    with pytest.raises(AssertionError):
        call_propose(p, sk=0)
    assert recorded_k() == (0, 1)


def test_positive_k_passes_through_step_k(monkeypatch):
    monkeypatch.setattr(lb, "_dsd_k0_skip", 0)
    monkeypatch.setattr(probe, "ENABLED", True)
    probe._pending.clear()
    p = make_proposer()
    with pytest.raises(AssertionError):
        call_propose(p, sk=3)
    assert recorded_k() == (3, 3)


def test_no_scheduler_output_falls_back_to_static(monkeypatch):
    monkeypatch.setattr(lb, "_dsd_k0_skip", 0)
    monkeypatch.setattr(probe, "ENABLED", True)
    probe._pending.clear()
    p = make_proposer()
    p.num_speculative_tokens = 7
    cam = SimpleNamespace(
        batch_size=lambda: 4,
        query_start_loc=torch.tensor([0, 4]),
    )
    with pytest.raises(AssertionError):
        p._propose(
            target_token_ids=torch.zeros(8, dtype=torch.long),
            target_positions=torch.zeros(8, dtype=torch.long),
            target_hidden_states=torch.zeros(8, 4),
            next_token_ids=torch.zeros(4, dtype=torch.long),
            token_indices_to_sample=None,
            common_attn_metadata=cam,
            target_model_batch_desc=None,
            sampling_metadata=None,
        )
    assert recorded_k() == (7, 7)
