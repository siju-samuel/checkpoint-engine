"""Pluggable device-buffer handoff between the ParameterServer and the worker.

The broadcast path shares a device buffer with the colocated worker. CUDA/NPU use
:class:`IpcWeightTransport` (``torch.multiprocessing`` CUDA IPC, wire-format
unchanged); XPU uses :class:`XpuIpcWeightTransport` (native SYCL ``ipc_memory``).
The handle is always a picklable, self-contained value, so the producer's
``export`` -> ZMQ ``send_pyobj`` -> consumer ``attach`` flow is identical for both;
each side calls ``detach`` on cleanup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from torch.multiprocessing.reductions import reduce_tensor


if TYPE_CHECKING:
    from collections.abc import Callable

    from checkpoint_engine.device_utils import DeviceManager


def _rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    return func(*list_args)


class WeightTransport(ABC):
    """Hands a device buffer from the producer (ps) to the consumer (worker)."""

    @abstractmethod
    def export(self, buffer: torch.Tensor) -> Any:
        """Producer: return the picklable handle to send over ZMQ."""

    @abstractmethod
    def attach(self, handle: Any, device_id: int) -> torch.Tensor:
        """Consumer: reconstruct the shared device buffer from ``handle``."""

    def detach(self) -> None:
        """Release IPC resources on either side. No-op by default."""


class IpcWeightTransport(WeightTransport):
    """CUDA/NPU zero-copy handoff via torch.multiprocessing CUDA IPC (unchanged)."""

    def export(self, buffer: torch.Tensor) -> Any:
        return reduce_tensor(buffer)

    def attach(self, handle: Any, device_id: int) -> torch.Tensor:
        assert isinstance(handle, tuple), f"expected reduce_tensor tuple, got {type(handle)}"
        buffer = _rebuild_ipc(handle, device_id)
        assert buffer.dtype == torch.uint8
        return buffer


class XpuIpcWeightTransport(WeightTransport):
    """Intel XPU zero-copy handoff via native SYCL ``ipc_memory`` (portable byte blob)."""

    kind = "xpu_sycl"  # wire tag: identifies this handle format to the consumer

    def __init__(self) -> None:
        self._opened_ptr: int | None = None  # consumer: the mapping to unmap
        self._exported_ptr: int | None = None  # producer: the retained exporter handle

    def export(self, buffer: torch.Tensor) -> Any:
        from checkpoint_engine import xpu_ipc

        ptr = buffer.data_ptr()
        handle_bytes = xpu_ipc.get_handle(ptr)
        # Release only in detach(): freeing before the consumer opens can drop the
        # fd under the level-zero-v2 UR adapter.
        self._exported_ptr = ptr
        return {
            "kind": self.kind,
            "handle_bytes": handle_bytes,
            "nbytes": buffer.nbytes,
        }

    def attach(self, handle: Any, device_id: int) -> torch.Tensor:
        from checkpoint_engine import xpu_ipc

        assert isinstance(handle, dict) and handle.get("kind") == self.kind, (
            f"expected {self.kind} handle dict, got {type(handle)}"
        )
        ptr = xpu_ipc.open_handle(handle["handle_bytes"], device_id)
        self._opened_ptr = ptr
        buffer = xpu_ipc.wrap_tensor(ptr, handle["nbytes"], device_id)
        assert buffer.dtype == torch.uint8
        return buffer

    def detach(self) -> None:
        # Consumer unmaps its opened pointer; producer releases its exported handle.
        # At most one of the two is set on any given instance.
        from checkpoint_engine import xpu_ipc

        if self._opened_ptr is not None:
            try:
                torch.xpu.synchronize()  # no in-flight reads before unmapping
                xpu_ipc.close_handle(self._opened_ptr)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"xpu ipc close_handle failed during detach: {e}")
            self._opened_ptr = None
        if self._exported_ptr is not None:
            try:
                xpu_ipc.release_handle(self._exported_ptr)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"xpu ipc release_handle failed during detach: {e}")
            self._exported_ptr = None


def build_transport(device_manager: DeviceManager) -> WeightTransport:
    """Select the weight transport for the current device backend."""
    if device_manager.device_type == "xpu":
        return XpuIpcWeightTransport()
    return IpcWeightTransport()
