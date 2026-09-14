from __future__ import annotations

import os
import re

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import DFLASH

# Table tiers: bs 1-4 -> K=5, 5-8 -> K=3, 9-16 -> K=1.
DSD_TABLE = [[1, 4, 5], [5, 8, 3], [9, 16, 1]]
TIER_KS = {5, 3, 1}
MAX_K = 5
MAX_NUM_SEQS = 16
# (k+1)*max_num_seqs must stay within the capture range for every tier.
CAPTURE_SIZES = [8, 16, 24, 32, 48, 64, 96]
PROBE_LOG = "/tmp/dsd_probe_emit.log"


def _make_prompts(tokenizer, n):
    base = [
        "Hello, your name is",
        "The capital of France is",
        "Write a short poem about autumn.",
        "Explain why the sky is blue.",
        "List three primary colors.",
        "Translate 'good morning' to French.",
        "What is 2 + 2?",
        "Name a famous scientist.",
    ] * 4
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in base[:n]
    ]


def _run_dsd(main_model_name, spec_model_name, run_phases):
    tokenizer = AutoTokenizer.from_pretrained(main_model_name, trust_remote_code=True)
    speculative_config = {
        "method": "dflash",
        "model": spec_model_name,
        "num_speculative_tokens": MAX_K,
        "num_speculative_tokens_per_batch_size": DSD_TABLE,
    }
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=CAPTURE_SIZES,
    )
    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
    ) as llm:
        sampling_params = SamplingParams(temperature=0, ignore_eos=False, max_tokens=64)
        outputs = []
        for num_prompts in run_phases:
            prompts = _make_prompts(tokenizer, num_prompts)
            outputs.append(llm.model.generate(prompts, sampling_params))
    return outputs


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


def test_dsd_tier_switch():
    if os.path.exists(PROBE_LOG):
        os.remove(PROBE_LOG)
    main_model_name = DFLASH["dflash"]["main"]
    spec_model_name = DFLASH["dflash"]["spec"]
    outputs = _run_dsd(main_model_name, spec_model_name, [2, 8, 16, 2])
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


def test_dsd_greedy_matches_no_spec():
    main_model_name = DFLASH["dflash"]["main"]
    spec_model_name = DFLASH["dflash"]["spec"]
    dsd_outputs = _run_dsd(main_model_name, spec_model_name, [8])[0]

    tokenizer = AutoTokenizer.from_pretrained(main_model_name, trust_remote_code=True)
    prompts = _make_prompts(tokenizer, 8)
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=CAPTURE_SIZES,
        fast_moe_cold_start=False,
    )
    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
    ) as llm:
        outputs = llm.model.generate(
            prompts, SamplingParams(temperature=0, ignore_eos=False, max_tokens=64))

    for dsd_out, ref_out in zip(dsd_outputs, outputs):
        assert dsd_out.outputs[0].token_ids == ref_out.outputs[0].token_ids, (
            f"greedy divergence for prompt {dsd_out.prompt!r}")
