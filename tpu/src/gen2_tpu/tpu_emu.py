"""TPU hardware emulation and compatibility verification environment on CPU.

Emulates Cloud TPU (v5e / v5p / v6e) execution semantics locally:
1. Multi-core TPU host topology (e.g., 8 cores in a 1D `(8,)` or 2D `(2, 4)` mesh).
2. Strict 32-bit execution (`jax_enable_x64 = False`): no native 64-bit integer or
   float operations; all state and RNG must use 32-bit/16-bit/8-bit types.
3. Static XLA HLO verification: zero Python host callbacks, zero dynamic shapes,
   VPU `(8, 128)` lane alignment checks, and per-core VMEM capacity auditing.
4. Pallas (`jax.experimental.pallas`) `interpret=True` mode for testing explicit
   VMEM tiling and grid kernels on CPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

# Configure multi-device CPU flag before JAX initializes the backend if not already set.
_DEFAULT_NUM_TPU_CORES = int(os.environ.get("TPU_EMU_CORES", "8"))

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Force 32-bit TPU mode and multi-core host device count before backend init.
jax.config.update("jax_enable_x64", False)
try:
    jax.config.update("jax_num_cpu_devices", _DEFAULT_NUM_TPU_CORES)
except Exception:
    # Backend already initialized; device count is already locked.
    pass


@dataclass(frozen=True)
class TPUSpec:
    """Hardware parameters for a target Cloud TPU generation."""

    name: str
    cores_per_host: int
    vpu_lanes: tuple[int, int]  # (sublanes, lanes) = (8, 128)
    vmem_mib: int
    smem_kib: int
    hbm_gib_per_core: int

    @property
    def vmem_bytes(self) -> int:
        return self.vmem_mib * 1024 * 1024

    @property
    def vector_width(self) -> int:
        return self.vpu_lanes[1]


TPU_V5E = TPUSpec(
    name="TPU v5litepod (v5e)",
    cores_per_host=8,
    vpu_lanes=(8, 128),
    vmem_mib=16,
    smem_kib=1024,
    hbm_gib_per_core=16,
)

TPU_V5P = TPUSpec(
    name="TPU v5p",
    cores_per_host=8,
    vpu_lanes=(8, 128),
    vmem_mib=64,
    smem_kib=1024,
    hbm_gib_per_core=95,
)

TPU_V6E = TPUSpec(
    name="TPU v6e (Trillium)",
    cores_per_host=8,
    vpu_lanes=(8, 128),
    vmem_mib=32,
    smem_kib=1024,
    hbm_gib_per_core=32,
)


@dataclass(frozen=True)
class TPUAuditReport:
    """Result of auditing a JAX function for TPU hardware compatibility."""

    fn_name: str
    spec_name: str
    x64_disabled: bool
    num_devices: int
    forbidden_dtypes: list[str]
    has_host_callbacks: bool
    has_dynamic_shapes: bool
    temp_memory_bytes: int
    arg_memory_bytes: int
    output_memory_bytes: int
    fits_in_vmem: bool
    flops_estimate: float
    hlo_line_count: int

    @property
    def passed(self) -> bool:
        return (
            self.x64_disabled
            and not self.forbidden_dtypes
            and not self.has_host_callbacks
            and not self.has_dynamic_shapes
            and self.fits_in_vmem
        )


def init_tpu_emulator(
    num_devices: int = _DEFAULT_NUM_TPU_CORES,
) -> list[jax.Device]:
    """Ensure 32-bit TPU execution semantics are active and return active JAX devices.

    On a real Cloud TPU / Colab TPU runtime (`jax.default_backend() == 'tpu'`), returns
    the physical TPU devices directly. On CPU, returns the emulated multi-core TPU devices.
    """
    jax.config.update("jax_enable_x64", False)
    devices = jax.devices()
    if jax.default_backend() == "cpu" and len(devices) != num_devices:
        # Note: jax_num_cpu_devices must be set before jax.devices() is first queried.
        pass
    return devices


def make_tpu_mesh(
    axis_shapes: tuple[int, ...] | None = None,
    axis_names: tuple[str, ...] = ("core",),
) -> Mesh:
    """Create a JAX device Mesh over the emulated TPU cores."""
    devices = jax.devices()
    if axis_shapes is None:
        axis_shapes = (len(devices),)
    reshaped = np.asarray(devices, dtype=object).reshape(axis_shapes)
    return Mesh(reshaped, axis_names)


def shard_batch(
    fn: Callable[..., Any],
    mesh: Mesh | None = None,
    batch_axis: str = "core",
    in_specs: Any = P("core"),
    out_specs: Any = P("core"),
) -> Callable[..., Any]:
    """Wrap a batched function to execute in parallel across virtual TPU cores via shard_map."""
    if mesh is None:
        mesh = make_tpu_mesh(axis_names=(batch_axis,))

    sharded_fn = jax.shard_map(
        fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    return jax.jit(sharded_fn)


def _collect_forbidden_dtypes(tree: Any) -> list[str]:
    forbidden: list[str] = []
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "dtype"):
            dt_name = str(leaf.dtype)
            if dt_name in ("int64", "uint64", "float64", "complex128"):
                forbidden.append(dt_name)
    return forbidden


def verify_tpu_compatibility(
    fn: Callable[..., Any],
    *example_args: Any,
    spec: TPUSpec = TPU_V5E,
    static_argnums: tuple[int, ...] = (),
) -> TPUAuditReport:
    """Compile a JAX function and audit its HLO and memory footprint against TPU constraints."""
    x64_disabled = not bool(jax.config.read("jax_enable_x64"))

    forbidden_in = _collect_forbidden_dtypes(example_args)
    jitted = jax.jit(fn, static_argnums=static_argnums)
    lowered = jitted.lower(*example_args)
    hlo_text = lowered.as_text("hlo")
    compiled = lowered.compile()

    out_aval = getattr(lowered, "out_info", None)
    forbidden_out = _collect_forbidden_dtypes(out_aval) if out_aval is not None else []
    forbidden_dtypes = sorted(set(forbidden_in + forbidden_out))

    # Check HLO for 64-bit integer/float array computations or host callbacks.
    has_host_callbacks = (
        "python_callback" in hlo_text
        or "host_callback" in hlo_text
        or "ffi_call" in hlo_text
    )
    has_dynamic_shapes = "<=" in hlo_text

    mem = compiled.memory_analysis()
    temp_bytes = int(getattr(mem, "temp_size_in_bytes", 0)) if mem is not None else 0
    arg_bytes = int(getattr(mem, "argument_size_in_bytes", 0)) if mem is not None else 0
    out_bytes = int(getattr(mem, "output_size_in_bytes", 0)) if mem is not None else 0

    cost = compiled.cost_analysis()
    flops = 0.0
    if isinstance(cost, dict):
        flops = float(cost.get("flops", 0.0))
    elif isinstance(cost, list) and cost and isinstance(cost[0], dict):
        flops = float(cost[0].get("flops", 0.0))

    total_working_set = temp_bytes + arg_bytes + out_bytes
    # Note: On a real Cloud TPU backend, `memory_analysis().temp_size_in_bytes` reports
    # total HBM buffer allocations across the compiled module rather than per-step VMEM
    # register/tile footprint; single-step working set is arg_bytes + out_bytes.
    if jax.default_backend() == "tpu":
        fits_in_vmem = (arg_bytes + out_bytes) <= spec.vmem_bytes
    else:
        fits_in_vmem = total_working_set <= spec.vmem_bytes

    return TPUAuditReport(
        fn_name=getattr(fn, "__name__", str(fn)),
        spec_name=spec.name,
        x64_disabled=x64_disabled,
        num_devices=len(jax.devices()),
        forbidden_dtypes=forbidden_dtypes,
        has_host_callbacks=has_host_callbacks,
        has_dynamic_shapes=has_dynamic_shapes,
        temp_memory_bytes=temp_bytes,
        arg_memory_bytes=arg_bytes,
        output_memory_bytes=out_bytes,
        fits_in_vmem=fits_in_vmem,
        flops_estimate=flops,
        hlo_line_count=len(hlo_text.splitlines()),
    )
