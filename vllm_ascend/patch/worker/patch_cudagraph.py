import bisect
import math
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm_ascend.utils import enable_sp

# bs step for the capture grid within each tier.
_DSD_BS_STEP = 2


def _is_dsd(self) -> bool:
    spec = self.vllm_config.speculative_config
    return spec is not None and spec.uses_dynamic_speculative_decoding()


def _create_padded_batch_descriptor(
    self,
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    num_active_loras: int = 0,
    num_reqs: int | None = None,
) -> BatchDescriptor:
    """DSD 2-D dispatch. When num_reqs is given (catalog build), emit an exact
    (num_tokens, num_reqs) cell. On the dispatch path, pick the smallest FULL
    cell that covers the real (num_tokens, actual_bs) -- exact, then pad up to
    the next captured bs for this query_len, then PIECEWISE fallback."""
    max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs

    if _is_dsd(self):
        # Catalog build path: explicit (num_tokens, num_reqs) cell.
        if num_reqs is not None:
            return BatchDescriptor(
                num_tokens=num_tokens,
                num_reqs=min(num_reqs, max_num_seqs),
                uniform=True,
                has_lora=has_lora,
                num_active_loras=num_active_loras,
            )

        # Dispatch path. Only uniform batches can reuse a FULL graph; mixed
        # batches (e.g. a request mid-join with K_prev=0) fall through to the
        # original logic below and dispatch as PIECEWISE.
        actual_bs = self._dsd_num_reqs
        if (
            uniform_decode
            and actual_bs is not None
            and actual_bs > 0
        ):
            query_len = num_tokens // actual_bs  # = 1 + K_prev
            full_keys = self.cudagraph_keys.get(CUDAGraphMode.FULL, set())

            def _mk(nt, nr, uni):
                return BatchDescriptor(
                    num_tokens=nt,
                    num_reqs=nr,
                    uniform=uni,
                    has_lora=has_lora,
                    num_active_loras=num_active_loras,
                )

            # 1. Exact cell: bs on the step-2 grid -> (nt, bs) is a catalog cell.
            exact_desc = _mk(num_tokens, actual_bs, True)
            if exact_desc in full_keys:
                desc = exact_desc
            else:
                # 2. Pad up to the smallest captured bs' >= actual_bs whose
                #    query_len matches (dummy reqs fill bs' - actual_bs).
                desc = None
                bs_list = self._dsd_bs_by_ql.get(query_len)
                if bs_list:
                    idx = bisect.bisect_left(bs_list, actual_bs)
                    if idx < len(bs_list):
                        bs_pad = bs_list[idx]
                        cand = _mk(bs_pad * query_len, bs_pad, True)
                        if cand in full_keys:
                            desc = cand

            if desc is not None:
                return desc

            # 3. No FULL graph for this shape: PIECEWISE for this one step.
            num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]
            return _mk(num_tokens_padded, None, False)

    # ---- original (non-DSD / mixed-batch) logic ----
    uniform_decode_query_len = self.uniform_decode_query_len
    num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]
    if (
        uniform_decode
        and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
        and self.cudagraph_mode != CUDAGraphMode.FULL
    ):
        num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
        assert num_tokens_padded % uniform_decode_query_len == 0
    else:
        uniform_decode = False
        num_reqs = min(num_tokens_padded, max_num_seqs)
    return BatchDescriptor(
        num_tokens=num_tokens_padded,
        num_reqs=num_reqs,
        uniform=uniform_decode,
        has_lora=has_lora,
        num_active_loras=num_active_loras,
    )


def _dsd_2d_cells(self, uniform_decode_query_len: int):
    """2-D FULL catalog for DSD: per tier, own k + all strictly-lower k, at bs
    on a step-2 grid within the tier's [start, end] (clamped to max_num_seqs,
    inclusive of the end). Returns deduped (num_tokens, num_reqs) cells."""
    spec = self.vllm_config.speculative_config
    max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
    max_cg = self.compilation_config.max_cudagraph_capture_size or 0
    # Each DSD table entry is [bs_start, bs_end, K]. K=0 is plain decode
    # (ql=1) -- still capture a graph cell for it (no speculation, but
    # graphified decode), so no tier is filtered out.
    table = [
        (start, end, k)
        for start, end, k in (spec.num_speculative_tokens_per_batch_size or [])
    ]
    # Distinct K values, descending. Lower-bs tiers are assumed to have higher K.
    all_ks = sorted({k for _, _, k in table}, reverse=True)

    cells = []
    seen = set()

    def add(num_tokens, bs):
        if max_cg and num_tokens > max_cg:
            return
        if not (1 <= bs <= max_num_seqs):
            return
        cell = (num_tokens, bs)
        if cell not in seen:
            seen.add(cell)
            cells.append(cell)

    for start, end, k_own in table:
        # Capture own k + every strictly-lower k (reachable as K_prev when bs
        # decreases out of a higher-bs tier that uses that lower k).
        ks = [k for k in all_ks if k <= k_own]
        if k_own == 0:
            # K=0 keep-alive drafts run at ql=2; capture those cells too.
            ks = ks + [1]
        bs_lo = max(start, 1)
        bs_hi = min(end, max_num_seqs)
        if bs_hi < bs_lo:
            continue
        for k in ks:
            qlen = k + 1
            # SP rounds num_tokens up to a TP multiple, which breaks the DSD
            # dispatch gate. Align the bs grid so bs*qlen is always a TP
            # multiple (step = TP // gcd(qlen, TP)), making SP padding a no-op.
            sp_on = enable_sp(self.vllm_config)
            # per-tier step: fine grid for the highest-K (win-window) tier, coarse elsewhere
            step = (self.vllm_config.parallel_config.tensor_parallel_size
                    // math.gcd(qlen, self.vllm_config.parallel_config.tensor_parallel_size)
                    ) if sp_on else (2 if k_own == max(all_ks) else 8)
            bs_start = ((bs_lo + step - 1) // step) * step
            bs_vals = list(range(bs_start, bs_hi + 1, step))
            if not bs_vals or bs_vals[-1] != bs_hi:
                # SP rejects num_tokens that is not a TP multiple. bs_hi for a
                # lower-K reachability grid can violate it (e.g. bs=63, ql=6);
                # drop the cell, runtime dispatch never lands there.
                if sp_on and (bs_hi * qlen) % self.vllm_config.parallel_config.tensor_parallel_size != 0:
                    pass
                else:
                    bs_vals.append(bs_hi)
            for bs in bs_vals:
                add(bs * qlen, bs)
    return cells


# Capture the upstream initializer before we monkey-patch it (the assignment is
# at the bottom of this module). Wrapping (instead of copying the body) keeps
# the upstream 1-D / PIECEWISE flow fully tracked: any upstream change is picked
# up automatically; we only override the FULL key set when DSD is on.
_orig_initialize_cudagraph_keys = CudagraphDispatcher.initialize_cudagraph_keys


def initialize_cudagraph_keys(
    self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
):
    """DSD wrapper: run upstream init unchanged, then (if DSD + FULL decode)
    replace the 1-D FULL keys with the 2-D {(num_tokens, num_reqs)} catalog.
    Non-DSD and PIECEWISE / prefill paths are 100% upstream behavior."""
    # DSD 2-D side-channel fields, read by _create_padded_batch_descriptor.
    self._dsd_num_reqs = None
    self._dsd_bs_by_ql = {}
    # Upstream init, verbatim: sets cudagraph_mode, builds PIECEWISE + 1-D FULL
    # keys, sets keys_initialized. (For DSD the 1-D FULL keys are thrown away
    # below, but building them is cheap — only BatchDescriptor objects, no graph
    # capture happens here.)
    _orig_initialize_cudagraph_keys(self, cudagraph_mode, uniform_decode_query_len)
    # DSD replaces the 1-D FULL catalog with the 2-D one.
    if (
        _is_dsd(self)
        and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        and cudagraph_mode.separate_routine()
    ):
        _replace_full_keys_with_dsd_2d_catalog(self, uniform_decode_query_len)


def _replace_full_keys_with_dsd_2d_catalog(self, uniform_decode_query_len: int):
    """Build the DSD 2-D FULL catalog and replace the upstream 1-D FULL keys
    with it. PIECEWISE keys (prefill fallback) are left untouched."""
    spec = self.vllm_config.speculative_config
    # Safety: each K-tier's max num_tokens must fit within capture range.
    max_cg_size = self.compilation_config.max_cudagraph_capture_size
    for _, _, k in (spec.num_speculative_tokens_per_batch_size or []):
        if k > 0:
            tier_max = (k + 1) * self.vllm_config.scheduler_config.max_num_seqs
            assert tier_max <= max_cg_size, (
                f"DSD 2-D: K={k} tier max num_tokens={tier_max} exceeds "
                f"max_cudagraph_capture_size={max_cg_size}. Increase "
                f"max_cudagraph_capture_size or reduce max_num_seqs.")
    lora_cases = self._get_lora_cases()
    dsd_cells = _dsd_2d_cells(self, uniform_decode_query_len)
    # Precompute bs grid per query_len for dispatch padding (round actual_bs up
    # to the next captured bs with the same query_len).
    self._dsd_bs_by_ql = {}
    for nt, nr in dsd_cells:
        self._dsd_bs_by_ql.setdefault(nt // nr, []).append(nr)
    for ql in self._dsd_bs_by_ql:
        self._dsd_bs_by_ql[ql].sort()
    # Replace the 1-D FULL keys (built by upstream) with the 2-D cells.
    self.cudagraph_keys[CUDAGraphMode.FULL] = set()
    for num_tokens, num_reqs in dsd_cells:
        for num_active_loras in lora_cases:
            self.add_cudagraph_key(
                CUDAGraphMode.FULL,
                self._create_padded_batch_descriptor(
                    num_tokens,
                    True,
                    num_active_loras > 0,
                    num_active_loras,
                    num_reqs=num_reqs,
                ),
            )


CudagraphDispatcher._create_padded_batch_descriptor = _create_padded_batch_descriptor
CudagraphDispatcher.initialize_cudagraph_keys = initialize_cudagraph_keys

# DSD 2-D FULL catalog requires FULL mode. vllm base forcibly downgrades
# FULL_AND_PIECEWISE → PIECEWISE when DSD is active, overriding the user
# configuration. Disable this override so FULL mode can be used with DSD.
from vllm.config.vllm import VllmConfig
VllmConfig._maybe_override_dynamic_sd_cudagraph_mode = lambda self: None
