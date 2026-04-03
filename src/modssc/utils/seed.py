from __future__ import annotations


def mix_seed32(seed: int, salt: int) -> int:
    """Deterministic 32-bit seed mixer."""
    x = (int(seed) + 0x9E3779B97F4A7C15 + int(salt)) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 30) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 31) & 0xFFFFFFFFFFFFFFFF
    return int(x & 0xFFFFFFFF)
