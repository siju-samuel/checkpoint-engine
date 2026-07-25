"""Cross-process device-buffer IPC for Intel XPU via SYCL ``ipc_memory``.

``sycl_ipc.cpp`` wraps ``ipc_memory`` (``get``/``open``/``close``), exported by
torch's own libsycl (oneAPI >= 2026.0); this module JIT-compiles it with
``-fsycl``. The handle is a self-contained portable byte blob (no dma-buf fd, no
offset to carry), so it rides the existing ZMQ channel like CUDA's
``reduce_tensor`` tuple -- see ``XpuIpcWeightTransport``.
"""

from __future__ import annotations

import functools
import glob
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger


if TYPE_CHECKING:
    from types import ModuleType

    import torch


def _find_sycl_include_dir() -> str | None:
    """Locate a directory containing <sycl/ext/oneapi/experimental/ipc_memory.hpp>."""
    candidates: list[str] = []
    root = os.getenv("CMPLR_ROOT")
    if root:
        candidates.append(os.path.join(root, "include"))
    # Common oneAPI install layouts (versioned + `latest` symlink).
    candidates += sorted(glob.glob("/opt/intel/oneapi/compiler/*/include"), reverse=True)
    # Derive from the discovered compiler (<root>/bin/icpx -> <root>/include), which
    # covers a PATH-only icpx whose oneAPI root is outside /opt.
    icpx = _find_icpx()
    if icpx:
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(icpx)), "include"))
    for inc in candidates:
        if os.path.exists(
            os.path.join(inc, "sycl", "ext", "oneapi", "experimental", "ipc_memory.hpp")
        ):
            return inc
    return None


def _find_icpx() -> str | None:
    """Locate the icpx (SYCL) compiler needed for the -fsycl build."""
    candidates: list[str] = []
    root = os.getenv("CMPLR_ROOT")
    if root:
        candidates.append(os.path.join(root, "bin", "icpx"))
    candidates += sorted(glob.glob("/opt/intel/oneapi/compiler/*/bin/icpx"), reverse=True)
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    # Fallback to PATH: covers oneAPI layouts outside /opt and a sourced setvars.sh
    # that puts icpx on PATH without exporting CMPLR_ROOT.
    return shutil.which("icpx")


@functools.lru_cache(maxsize=1)
def load_ext() -> ModuleType:
    """JIT-compile (``-fsycl``, linking torch's libsycl) and cache the SYCL IPC extension.

    Raises on any failure; callers treat an exception as "XPU IPC unavailable".
    """
    icpx = _find_icpx()
    if icpx is None:
        raise RuntimeError("icpx (oneAPI SYCL compiler) not found; cannot build XPU IPC extension")
    icx = os.path.join(os.path.dirname(icpx), "icx")

    from torch.utils.cpp_extension import load

    src = Path(__file__).with_name("sycl_ipc.cpp")

    sycl_include_flags: list[str] = []
    inc = _find_sycl_include_dir()
    if inc:
        sycl_include_flags = [f"-I{inc}", f"-I{os.path.join(inc, 'sycl')}"]

    # torch.utils.cpp_extension picks the compiler from CC/CXX. Force icx/icpx for the
    # -fsycl build: a conda/CI env often exports CXX=g++ (gxx_linux-64), which cannot
    # compile -fsycl, and setdefault would keep it. Save/restore the process env.
    prev_cc, prev_cxx = os.environ.get("CC"), os.environ.get("CXX")
    if prev_cxx and os.path.realpath(prev_cxx) != os.path.realpath(icpx):
        logger.debug(f"overriding CXX={prev_cxx!r} with icpx for the SYCL IPC build ({icpx})")
    os.environ["CC"], os.environ["CXX"] = icx, icpx
    try:
        # Do NOT pin -std: torch.utils.cpp_extension injects the standard its ATen
        # headers require, and a pin here would override it.
        module = load(
            name="checkpoint_engine_sycl_ipc",
            sources=[str(src)],
            extra_cflags=["-fsycl", "-O2", *sycl_include_flags],
            extra_ldflags=["-fsycl"],
            verbose=False,
        )
    finally:
        for var, prev in (("CC", prev_cc), ("CXX", prev_cxx)):
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev
    return module


# Cache only a *successful* probe so a transient first failure can be retried
# (the build itself is memoised by load_ext()'s lru_cache).
_AVAILABLE: bool = False


def is_available() -> bool:
    """Whether native XPU SYCL IPC can be built and used here (successes cached, failures retried)."""
    global _AVAILABLE
    if _AVAILABLE:
        return True
    try:
        import torch

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return False
        load_ext()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"xpu sycl ipc unavailable: {e}")
        return False
    _AVAILABLE = True
    return True


def prewarm() -> bool:
    """Build the extension ahead of time (outside any weight-update timeout); safe on non-XPU hosts."""
    return is_available()


def get_handle(ptr: int) -> bytes:
    """Portable IPC handle bytes for a device pointer (interior pointers ok; offset is in the blob)."""
    return bytes(load_ext().ipc_get_handle(ptr))


def open_handle(handle_bytes: bytes, device: int) -> int:
    """Open another process's handle -> device pointer (offset included); free via :func:`close_handle`."""
    return load_ext().ipc_open_handle(list(handle_bytes), device)


def release_handle(ptr: int) -> None:
    """Release the exporter handle from :func:`get_handle`; no-op if ``ptr`` was never exported.

    Deferred until the consumer has opened: releasing earlier can free the fd under
    the level-zero-v2 UR adapter.
    """
    load_ext().ipc_release_handle(ptr)


def close_handle(ptr: int) -> None:
    load_ext().ipc_close_handle(ptr)


def wrap_tensor(ptr: int, nbytes: int, device: int) -> torch.Tensor:
    """Wrap an IPC-mapped device pointer as a non-owning torch XPU uint8 tensor."""
    return load_ext().ipc_wrap_tensor(ptr, nbytes, device)
