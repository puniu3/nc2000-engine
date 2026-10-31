"""Bit-exact 32-bit implementation of Pokemon Showdown's Gen5RNG (64-bit LCG).

On a Cloud TPU (`jax_enable_x64=False`), 64-bit integers (`uint64`) are not native
and are truncated to `uint32` by default. This module represents the 64-bit seed as
a pair of `uint32` words `[hi32, lo32]` and performs the exact 64-bit LCG update:
    seed_{n+1} = seed_n * 0x5D588B656C078965 + 0x269EC3 (mod 2^64)
and upper-32-bit multiply-shift `floor(next_u32 * n / 2^32)` using 16-bit limb
arithmetic inside `uint32`, with zero 32-bit overflow and bit-for-bit parity with
`nc2000_engine::prng::Prng`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Multiplier 0x5D58_8B65_6C07_8965 and addend 0x0000_0000_0026_9EC3 as 16-bit limbs.
_M0 = jnp.uint32(0x8965)
_M1 = jnp.uint32(0x6C07)
_M2 = jnp.uint32(0x8B65)
_M3 = jnp.uint32(0x5D58)

_A0 = jnp.uint32(0x9EC3)
_A1 = jnp.uint32(0x0026)
_MASK16 = jnp.uint32(0xFFFF)


def seed_from_u64_int(seed_int: int) -> jax.Array:
    """Convert a Python 64-bit integer seed into a `uint32[2]` `[hi32, lo32]` array."""
    seed_int = int(seed_int) & 0xFFFFFFFFFFFFFFFF
    hi = (seed_int >> 32) & 0xFFFFFFFF
    lo = seed_int & 0xFFFFFFFF
    return jnp.array([hi, lo], dtype=jnp.uint32)


def seed_from_str(s: str) -> jax.Array:
    """Parse Showdown seed strings (`"1,2,3,4"` or `"gen5,0123456789abcdef"`) into `uint32[2]`."""
    s = s.strip()
    if s.startswith("gen5,"):
        val = int(s[5:], 16)
        return seed_from_u64_int(val)
    limbs = [int(p.strip()) & 0xFFFF for p in s.split(",")]
    if len(limbs) != 4:
        raise ValueError(f"Invalid Gen5RNG seed string: {s!r}")
    hi = (limbs[0] << 16) | limbs[1]
    lo = (limbs[2] << 16) | limbs[3]
    return jnp.array([hi, lo], dtype=jnp.uint32)


def seed_to_u64_int(seed: jax.Array) -> int:
    """Convert a `uint32[2]` `[hi32, lo32]` seed back to a Python 64-bit int."""
    hi = int(seed[0]) & 0xFFFFFFFF
    lo = int(seed[1]) & 0xFFFFFFFF
    return (hi << 32) | lo


def seed_to_str(seed: jax.Array) -> str:
    """Format a `uint32[2]` seed as Showdown's `"l0,l1,l2,l3"` decimal limb string."""
    hi = int(seed[0]) & 0xFFFFFFFF
    lo = int(seed[1]) & 0xFFFFFFFF
    return f"{(hi >> 16) & 0xFFFF},{hi & 0xFFFF},{(lo >> 16) & 0xFFFF},{lo & 0xFFFF}"


def prng_next_u32(seed: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Advance the 64-bit Gen5 LCG by 1 step using pure `uint32` 16-bit limb math.

    Args:
        seed: `uint32[..., 2]` where `[..., 0]` is `hi32` and `[..., 1]` is `lo32`.

    Returns:
        `(new_seed, draw_u32)` where `draw_u32` is `new_seed[..., 0]` (`uint32[...]`).
    """
    hi32 = seed[..., 0].astype(jnp.uint32)
    lo32 = seed[..., 1].astype(jnp.uint32)

    s0 = lo32 & _MASK16
    s1 = lo32 >> jnp.uint32(16)
    s2 = hi32 & _MASK16
    s3 = hi32 >> jnp.uint32(16)

    # Limb 0: max 0xFFFF * 0x8965 + 0x9EC3 = 0x8965155E < 2^32
    c0 = s0 * _M0 + _A0
    r0 = c0 & _MASK16
    carry0 = c0 >> jnp.uint32(16)

    # Limb 1: max 0x8965 + 0xFFFF * (0x6C07 + 0x8965) + 0x26 = 0xF56B969E < 2^32
    c1 = carry0 + s0 * _M1 + s1 * _M0 + _A1
    r1 = c1 & _MASK16
    carry1 = c1 >> jnp.uint32(16)

    # Limb 2: split into two uint32 additions to avoid > 2^32 overflow
    c2a = carry1 + s0 * _M2 + s1 * _M1
    c2b = (c2a & _MASK16) + s2 * _M0
    r2 = c2b & _MASK16
    carry2 = (c2a >> jnp.uint32(16)) + (c2b >> jnp.uint32(16))

    # Limb 3: split into two uint32 additions
    c3a = carry2 + s0 * _M3 + s1 * _M2
    c3b = (c3a & _MASK16) + s2 * _M1 + s3 * _M0
    r3 = c3b & _MASK16

    new_lo32 = (r1 << jnp.uint32(16)) | r0
    new_hi32 = (r3 << jnp.uint32(16)) | r2
    new_seed = jnp.stack([new_hi32, new_lo32], axis=-1)
    return new_seed, new_hi32


def prng_random(seed: jax.Array, n: int | jax.Array) -> tuple[jax.Array, jax.Array]:
    """Advance PRNG and return integer in `[0, n)` matching `Prng::random(n)` (`n < 65536`)."""
    new_seed, draw_u32 = prng_next_u32(seed)
    n_u32 = jnp.asarray(n, dtype=jnp.uint32)
    r2 = draw_u32 & _MASK16
    r3 = draw_u32 >> jnp.uint32(16)
    p0 = r2 * n_u32
    p1 = r3 * n_u32 + (p0 >> jnp.uint32(16))
    val = (p1 >> jnp.uint32(16)).astype(jnp.int32)
    return new_seed, val


def prng_random_chance(
    seed: jax.Array, num: int | jax.Array, den: int | jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Advance PRNG and return boolean `random(den) < num` matching `Prng::random_chance`."""
    new_seed, draw = prng_random(seed, den)
    num_i32 = jnp.asarray(num, dtype=jnp.int32)
    return new_seed, draw < num_i32
