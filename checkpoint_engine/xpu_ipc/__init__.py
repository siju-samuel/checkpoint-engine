"""Cross-process device-buffer IPC for Intel XPU via SYCL IPC memory.

``sycl_ipc.cpp`` wraps the SYCL IPC memory API (``get``/``open``/``close``),
exported by torch's own libsycl (oneAPI >= 2026.0); this module JIT-compiles it
with ``with_sycl``. The handle is a self-contained portable byte blob (no dma-buf
fd, no offset to carry), so it rides the existing ZMQ channel like CUDA's
``reduce_tensor`` tuple -- see ``XpuIPCHandler``.
"""

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


def _has_ipc_memory(icpx: str) -> bool:
    """Whether this icpx ships the SYCL IPC memory header (oneAPI >= 2026.0)."""
    root = os.path.dirname(os.path.dirname(icpx))
    header = os.path.join(
        root, "include", "sycl", "ext", "oneapi", "experimental", "ipc_memory.hpp"
    )
    return os.path.exists(header)


def _icpx_version_key(path: str) -> tuple[int, list[int]]:
    """Sort key for ``.../compiler/<ver>/bin/icpx``: numeric parts, so 2026.10 > 2026.9.

    The non-numeric ``latest`` symlink sorts first (it points at the newest install).
    """
    version = path.split("/")[-3]
    parts = version.split(".")
    if not all(p.isdigit() for p in parts):
        return (1, [])
    return (0, [int(p) for p in parts])


def _find_icpx() -> str | None:
    """Locate an icpx (SYCL) compiler new enough for the SYCL IPC memory build.

    Compilers without the header are skipped: they build a device image that
    torch's newer libsycl cannot load, aborting the process on dlopen rather
    than raising something we could catch.
    """
    candidates: list[str] = []
    root = os.getenv("CMPLR_ROOT")
    if root:
        candidates.append(os.path.join(root, "bin", "icpx"))
    candidates += sorted(
        glob.glob("/opt/intel/oneapi/compiler/*/bin/icpx"),
        key=_icpx_version_key,
        reverse=True,
    )
    # Fallback to PATH: covers oneAPI layouts outside /opt and a sourced setvars.sh
    # that puts icpx on PATH without exporting CMPLR_ROOT.
    which = shutil.which("icpx")
    if which:
        candidates.append(which)
    return next(
        (c for c in candidates if os.path.exists(c) and _has_ipc_memory(c)),
        None,
    )


@functools.lru_cache(maxsize=1)
def load_ext() -> "ModuleType":
    """JIT-compile (``with_sycl``, linking torch's libsycl) and cache the SYCL IPC extension.

    Raises on any failure; callers treat an exception as "XPU IPC unavailable".
    """
    icpx = _find_icpx()
    if icpx is None:
        raise RuntimeError(
            "no icpx with SYCL ipc_memory support found (needs oneAPI >= 2026.0); "
            "cannot build XPU IPC extension"
        )

    from torch.utils.cpp_extension import load

    src = Path(__file__).with_name("sycl_ipc.cpp")

    # with_sycl=True supplies the SYCL include paths and device link, but torch invokes
    # a bare "icpx", so it must be on PATH; keep -O2 or the host object is built -O0.
    prev_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.path.dirname(icpx) + os.pathsep + prev_path
    try:
        module = load(
            name="checkpoint_engine_sycl_ipc",
            sources=[str(src)],
            extra_cflags=["-O2"],
            with_sycl=True,
            verbose=False,
        )
    finally:
        os.environ["PATH"] = prev_path
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


def wrap_tensor(ptr: int, nbytes: int, device: int) -> "torch.Tensor":
    """Wrap an IPC-mapped device pointer as a non-owning torch XPU uint8 tensor."""
    return load_ext().ipc_wrap_tensor(ptr, nbytes, device)
