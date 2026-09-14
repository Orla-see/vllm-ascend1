# Per-step DSD dispatch probe.
#
# Records the scheduler/proposer/runner K chain and the graph selection
# (hit type, padding, capture cell) for each step, then emits one compact
# line per target dispatch from TP rank 0. Active only when
# VLLM_ASCEND_DSD_PROBE=1; all record/emit calls are no-ops otherwise.
import os
import time

from vllm.logger import init_logger

logger = init_logger("vllm_ascend.dsd_probe")

ENABLED = os.environ.get("VLLM_ASCEND_DSD_PROBE", "1") != "0"

_pending: dict[str, object] = {}
_step = 0

try:
    with open("/tmp/dsd_probe_imports.log", "a") as fh:
        fh.write(f"imported pid={os.getpid()} enabled={ENABLED}\n")
except OSError:
    pass


def emit_if_target(rank: int) -> None:
    """Emit from the dispatcher when the pending dispatch is target-side."""
    if _pending.get("side") == "target":
        emit(rank)


def set_enabled(flag: bool) -> None:
    """Enable via the vllm_config channel (worker env vars are filtered)."""
    global ENABLED
    ENABLED = ENABLED or flag


def set_side(side: str) -> None:
    if not ENABLED:
        return
    _pending["side"] = side


def record(**fields: object) -> None:
    if not ENABLED:
        return
    _pending.update(fields)


def record_side(**fields: object) -> None:
    """Record dispatch fields prefixed with the side set via set_side()."""
    if not ENABLED:
        return
    side = _pending.get("side", "x")
    _pending.update({f"{side}_{k}": v for k, v in fields.items()})


def emit(rank: int) -> None:
    global _pending, _step
    if not ENABLED:
        return
    if rank != 0:
        _pending.clear()
        return
    fields = _pending
    _pending = {}
    if not fields:
        return
    _step += 1
    parts = [f"{k}={fields[k]}" for k in sorted(fields)]
    line = f"DSDPROBE step={_step} ts={time.time():.2f} " + " ".join(parts)
    logger.info("%s", line)
    with open("/tmp/dsd_probe_emit.log", "a") as fh:
        fh.write(line + "\n")
