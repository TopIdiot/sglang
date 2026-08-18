# Vendored from the k-dash kernel source package welm/v45_80a3_attention.
# The CUDA kernel.so is resolved at runtime by k_dash.get(); that repo is a
# k-dash Source Package, not a Python distribution, so the host planner is
# mirrored here. Keep edits in sync with the upstream repo.
"""Resolve and load the WeLM v4.5 80A3 attention kernels through k-dash."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

import torch
import tvm_ffi

from .decode_executors import DecodeExecutor, resolve_decode_executor
from .verify_executors import resolve_verify_executor

KERNEL_NAME = "welm/v45_80a3_attention"
DEFAULT_KERNEL_VERSION = "dev-local"

_EMPTY: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}
_EMPTY_LOCK = threading.Lock()
_MODULES: dict[tuple[str, int, str], Any] = {}
_MODULES_LOCK = threading.Lock()


class ConfigError(ValueError):
    """Invalid attention configuration or inputs."""


class LaunchError(RuntimeError):
    """Kernel launch or host prepare failed."""


def _empty_tensor(device: torch.device, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    key = (str(device), dtype)
    with _EMPTY_LOCK:
        tensor = _EMPTY.get(key)
        if tensor is None:
            tensor = torch.empty((0,), device=device, dtype=dtype)
            _EMPTY[key] = tensor
        return tensor


def _as_tensor(value: object, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if value is None:
        return _empty_tensor(device, dtype)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor or None, got {type(value)!r}")
    return value


def kernel_so_path(*, kind: str, worker_count: int, executor: str):
    """Ask k-dash to solve the download/cache path for one ``kernel.so``."""

    import k_dash

    version = os.environ.get("K_DASH_KERNEL_VERSION", DEFAULT_KERNEL_VERSION)
    return k_dash.get(
        KERNEL_NAME,
        version=version,
        jit_args={
            "kernel": kind,
            "worker_count": int(worker_count),
            "executor": str(executor),
        },
    )


def load_module(*, kind: str, worker_count: int, executor: str):
    key = (kind, int(worker_count), str(executor))
    with _MODULES_LOCK:
        cached = _MODULES.get(key)
        if cached is not None:
            return cached

    try:
        so_path = kernel_so_path(
            kind=kind, worker_count=worker_count, executor=executor
        )
    except ImportError as exc:
        raise ImportError(
            "the WeLM v4.5 80A3 attention kernels are distributed as the k-dash "
            "kernel 'welm/v45_80a3_attention'; install k-dash and provide "
            "~/.config/k-dash.yaml"
        ) from exc
    module = tvm_ffi.load_module(str(so_path))

    with _MODULES_LOCK:
        _MODULES[key] = module
    return module


@dataclass
class KernelLibrary:
    """Verify-attention TVM-FFI module wrapper (22-arg ABI)."""

    module: Any
    worker_count: int
    executor: str
    name: str

    def workspace_size(
        self,
        args: tuple[object, ...],
        *,
        device_id: int,
        stream: int,
        context: object | None = None,
    ) -> int:
        del stream, context
        with tvm_ffi.use_torch_stream():
            return int(self.module.workspace_size(*self._pack(args, device_id)))

    def init_cpu_workspace(
        self,
        args: tuple[object, ...],
        *,
        device_id: int,
        stream: int,
        context: object | None = None,
    ) -> None:
        del stream, context
        with tvm_ffi.use_torch_stream():
            self.module.init_cpu_workspace(*self._pack(args, device_id))

    def launch_prepared(
        self,
        args: tuple[object, ...],
        *,
        device_id: int,
        stream: int,
        context: object | None = None,
    ) -> None:
        del stream, context
        with tvm_ffi.use_torch_stream():
            self.module.launch(*self._pack(args, device_id))

    def create_full_plan_stager(self, *args: object, **kwargs: object) -> object:
        raise ConfigError(
            "dynamic runtime replan/stager is not exported in the k-dash package yet"
        )

    def replan_and_stage(self, *args: object, **kwargs: object) -> int:
        raise ConfigError(
            "dynamic runtime replan/stager is not exported in the k-dash package yet"
        )

    def _pack(self, args: tuple[object, ...], device_id: int) -> tuple[object, ...]:
        if len(args) != 22:
            raise ValueError(f"{self.name} expects 22 ABI args, got {len(args)}")
        device = torch.device("cuda", device_id)
        cpu = torch.device("cpu")

        def tensor(index: int, *, prefer_cpu: bool = False, dtype: torch.dtype = torch.uint8):
            value = args[index]
            target = cpu if prefer_cpu else device
            if isinstance(value, torch.Tensor):
                return value
            return _as_tensor(value, device=target, dtype=dtype)

        return (
            tensor(0, dtype=torch.bfloat16),
            tensor(1, dtype=torch.bfloat16),
            tensor(2, dtype=torch.bfloat16),
            int(args[3] or 0),
            int(args[4]),
            tensor(5, dtype=torch.int32),
            tensor(6, dtype=torch.bfloat16),
            tensor(7, dtype=torch.bfloat16),
            tensor(8, dtype=torch.int32),
            int(args[9]),
            int(args[10]),
            int(args[11]),
            float(args[12]),
            tensor(13),
            tensor(14),
            tensor(15, prefer_cpu=True),
            tensor(16, dtype=torch.int32),
            int(args[17] or 0),
            int(args[18] if args[18] is not None else -1),
            int(args[19]),
            int(args[20]),
            int(args[21]),
        )


@dataclass
class DecodeKernelLibrary:
    """Decode-attention TVM-FFI module wrapper."""

    module: Any
    worker_count: int
    executor: str
    name: str

    def scratch_workspace_size(self) -> int:
        return int(self.module.scratch_workspace_size())

    def plan_create(self, *args: object) -> int:
        with tvm_ffi.use_torch_stream():
            return int(self.module.plan_create(*args))

    def plan_destroy(self, plan_handle: int) -> None:
        self.module.plan_destroy(int(plan_handle))

    def plan_workspace_size(self, plan_handle: int) -> int:
        return int(self.module.plan_workspace_size(int(plan_handle)))

    def plan_scratch_workspace_size(self, plan_handle: int) -> int:
        return int(self.module.plan_scratch_workspace_size(int(plan_handle)))

    def plan_init_workspace(self, *args: object) -> None:
        packed = list(args)
        if packed[1] is None:
            packed[1] = _empty_tensor(torch.device("cuda"), torch.uint8)
        with tvm_ffi.use_torch_stream():
            self.module.plan_init_workspace(*packed)

    def init_workspace_same_plan(self, *args: object) -> None:
        packed = list(args)
        if packed[8] is None:
            packed[8] = _empty_tensor(torch.device("cuda"), torch.uint8)
        with tvm_ffi.use_torch_stream():
            self.module.init_workspace_same_plan(*packed)

    def launch_prepared(
        self,
        args: tuple[object, ...],
        *,
        device_id: int,
        stream: int,
        context: object | None = None,
    ) -> None:
        del stream, context
        with tvm_ffi.use_torch_stream():
            self.module.launch(*self._pack(args, device_id))

    def _pack(self, args: tuple[object, ...], device_id: int) -> tuple[object, ...]:
        # query..gate (17) + workspace + cpu_workspace = 19
        if len(args) != 19:
            raise ValueError(f"{self.name} expects 19 ABI args, got {len(args)}")
        device = torch.device("cuda", device_id)
        cpu = torch.device("cpu")

        def tensor(index: int, *, prefer_cpu: bool = False, dtype: torch.dtype = torch.uint8):
            value = args[index]
            target = cpu if prefer_cpu else device
            if isinstance(value, torch.Tensor):
                return value
            return _as_tensor(value, device=target, dtype=dtype)

        return (
            tensor(0, dtype=torch.bfloat16),
            tensor(1, dtype=torch.bfloat16),
            tensor(2, dtype=torch.bfloat16),
            tensor(3, dtype=torch.int32),
            tensor(4, dtype=torch.bfloat16),
            tensor(5, dtype=torch.float32),
            tensor(6, dtype=torch.bfloat16),
            tensor(7, prefer_cpu=True, dtype=torch.int32),
            int(args[8]),
            int(args[9]),
            int(args[10]),
            int(args[11]),
            float(args[12]),
            int(args[13]),
            int(args[14]),
            tensor(15),
            tensor(16, dtype=torch.bfloat16),
            tensor(17),
            tensor(18, prefer_cpu=True),
        )


@dataclass
class PreparedKernel:
    _library: KernelLibrary
    _context: None = None
    _device_id: int = 0


@dataclass
class PreparedDecodeKernel:
    _library: DecodeKernelLibrary
    _context: None = None
    _device_id: int = 0


def prepare_verify_kernel(
    *,
    schedule_policy: str,
    partial_merge_mode: str,
    use_wgmma_static_persistent: bool,
    worker_count: int,
    name: str,
) -> PreparedKernel:
    executor = resolve_verify_executor(
        schedule_policy,
        partial_merge_mode=partial_merge_mode,
        use_wgmma_static_persistent=use_wgmma_static_persistent,
    )
    module = load_module(
        kind="verify",
        worker_count=worker_count,
        executor=executor.value,
    )
    library = KernelLibrary(
        module=module,
        worker_count=worker_count,
        executor=executor.value,
        name=name,
    )
    return PreparedKernel(_library=library)


def prepare_decode_kernel(
    *,
    worker_count: int,
    device_id: int,
    has_window: bool = False,
    executor: str | None = None,
) -> PreparedDecodeKernel:
    resolved = (
        executor
        if executor is not None
        else resolve_decode_executor(has_window=has_window).value
    )
    if resolved not in (
        DecodeExecutor.DECODE_DEFAULT.value,
        DecodeExecutor.DECODE_WINDOW.value,
    ):
        raise ConfigError(f"unsupported decode executor: {resolved!r}")
    module = load_module(
        kind="decode",
        worker_count=worker_count,
        executor=resolved,
    )
    library = DecodeKernelLibrary(
        module=module,
        worker_count=worker_count,
        executor=resolved,
        name=f"decode_attn_w{worker_count}_{resolved}",
    )
    return PreparedDecodeKernel(_library=library, _device_id=int(device_id))


# Back-compat name used by the ported planner.
prepare_kernel = prepare_verify_kernel
