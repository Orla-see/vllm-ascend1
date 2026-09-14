# SPDX-License-Identifier: Apache-2.0
"""DSD tier-switch and greedy-equivalence E2E on Qwen3.6-35B-A3B (TP4).

Run `pytest tests/e2e/pull_request/four_card/spec_decode/test_dsd_dynamic_qwen36.py`.
Asserts, via the DSD probe log: scheduler K stays within the table's tier set,
every tier K is exercised, and K returns to the low-bs tier after load drops.
Greedy outputs must match the no-speculation baseline; at the first
divergence, the DSD token must be one of the baseline's top-5 logprob
alternatives at that position, since speculative and non-speculative decode
run different numeric paths (verify batches multiple tokens per request) and
can flip the argmax between plausible alternatives on this quantized model.
"""

import os
import re

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ASCEND_GDN_CONV1D_BACKEND", "triton")
os.environ.setdefault("HCCL_BUFFSIZE", "1024")

from vllm import SamplingParams
from vllm.config import CompilationConfig

from tests.e2e.conftest import VllmRunner

MAIN_MODEL = "/mnt/share/weights/Qwen3.6-35B-A3B-w4a8"
DRAFT_MODEL = "/mnt/share/t00886357/sw-eagle3/dflash_weights/Qwen3.6-35B-A3B-DFlash"

# Table tiers: bs 1-4 -> K=5, 5-8 -> K=3, 9-16 -> K=1.
DSD_TABLE = [[1, 4, 5], [5, 8, 3], [9, 16, 1]]
TIER_KS = {5, 3, 1}
MAX_K = 5
MAX_NUM_SEQS = 16
# (k+1)*max_num_seqs must stay within the capture range for every tier.
CAPTURE_SIZES = [8, 16, 24, 32, 48, 64, 96]
PROBE_LOG = "/tmp/dsd_probe_emit.log"
GREEDY_ALT_TOPK = 5

PROMPTS = [
    "Hello, your name is",
    "The capital of France is",
    "Write a short poem about autumn.",
    "Explain why the sky is blue.",
    "List three primary colors.",
    "Translate 'good morning' to French.",
    "What is 2 + 2?",
    "Name a famous scientist.",
] * 4

SPECULATIVE_CONFIG = {
    "method": "dflash",
    "model": DRAFT_MODEL,
    "num_speculative_tokens": MAX_K,
    "num_speculative_tokens_per_batch_size": DSD_TABLE,
}


def _run_dsd():
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=CAPTURE_SIZES,
    )
    with VllmRunner(
        MAIN_MODEL,
        max_model_len=4096,
        disable_log_stats=True,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        max_num_seqs=MAX_NUM_SEQS,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.88,
        speculative_config=SPECULATIVE_CONFIG,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
        seed=1024,
    ) as llm:
        sampling_params = SamplingParams(
            temperature=0, ignore_eos=False, max_tokens=32)
        outputs = []
        for num_prompts in (2, 8, 16, 2):
            outputs.append(llm.model.generate(
                PROMPTS[:num_prompts], sampling_params))
    return outputs


def _run_nospec():
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=CAPTURE_SIZES,
        fast_moe_cold_start=False,
    )
    with VllmRunner(
        MAIN_MODEL,
        max_model_len=4096,
        disable_log_stats=True,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        max_num_seqs=MAX_NUM_SEQS,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.88,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
        seed=1024,
    ) as llm:
        sampling_params = SamplingParams(
            temperature=0, ignore_eos=False, max_tokens=32, logprobs=10)
        return llm.model.generate(PROMPTS[:8], sampling_params)


def _read_probe_sk():
    if not os.path.exists(PROBE_LOG):
        return []
    sks = []
    with open(PROBE_LOG) as fh:
        for line in fh:
            m = re.search(r"\bsk=(\d+)", line)
            if m:
                sks.append(int(m.group(1)))
    return sks


def _first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def _assert_greedy_close(dsd_out, ref_out):
    dsd_ids = dsd_out.outputs[0].token_ids
    ref_ids = ref_out.outputs[0].token_ids
    idx = _first_divergence(dsd_ids, ref_ids)
    if idx is None:
        return
    lps = ref_out.outputs[0].logprobs[idx]
    dsd_tok = dsd_ids[idx]
    top = sorted(lps.items(), key=lambda kv: kv[1].logprob, reverse=True)
    assert any(tok == dsd_tok for tok, _ in top[:GREEDY_ALT_TOPK]), (
        f"greedy divergence for prompt {dsd_out.prompt!r}: token {dsd_tok} "
        f"not in baseline top-{GREEDY_ALT_TOPK} logprobs at pos {idx}")


def test_dsd_tier_switch_and_greedy_equivalence():
    if os.path.exists(PROBE_LOG):
        os.remove(PROBE_LOG)
    outputs = _run_dsd()
    for phase in outputs:
        assert len(phase) > 0
        for out in phase:
            assert out.outputs[0].token_ids

    sks = _read_probe_sk()
    assert sks, "no DSD probe records emitted"
    assert set(sks) <= TIER_KS
    # bs 1-4 -> K=5, bs 5-8 -> K=3, bs 9-16 -> K=1, back to bs<=4 -> K=5
    assert 5 in sks and 3 in sks and 1 in sks
    assert 5 in sks[-10:]

    ref = _run_nospec()
    for dsd_out, ref_out in zip(outputs[1], ref):
        _assert_greedy_close(dsd_out, ref_out)
