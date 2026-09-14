"""Cross-module holder for the current DSD tier K.

The scheduler patch sets it before each schedule() call; the mamba manager
patch reads it to size the per-request speculative state blocks.
"""

_current_k: int | None = None


def set_k(k: int | None) -> None:
    global _current_k
    _current_k = k


def current_k() -> int | None:
    return _current_k
