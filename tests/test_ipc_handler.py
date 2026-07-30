"""Unit tests for the IPC-handler seam (CPU-only, no accelerator required).

The zero-copy device handoff itself is exercised by a hardware-gated end-to-end
test on real XPU/CUDA. Here we cover the dispatch logic and wire formats.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from checkpoint_engine.ipc_handler import (
    TorchIPCHandler,
    XpuIPCHandler,
    build_ipc_handler,
)
from checkpoint_engine.worker import _ipc_handler_for_handle


def _dm(device_type: str) -> object:
    return SimpleNamespace(device_type=device_type)


@pytest.mark.parametrize(
    "device_type,expected",
    [("cuda", TorchIPCHandler), ("npu", TorchIPCHandler), ("xpu", XpuIPCHandler)],
)
def test_build_ipc_handler_dispatch(device_type: str, expected: type):
    assert isinstance(build_ipc_handler(_dm(device_type)), expected)


def test_consumer_dispatch_by_handle_shape():
    # CUDA/NPU send a reduce_tensor tuple; XPU sends a tagged dict.
    tuple_handle = (lambda *a: None, (1, 2, 3))
    assert isinstance(_ipc_handler_for_handle(tuple_handle), TorchIPCHandler)

    xpu_handle = {"kind": XpuIPCHandler.kind, "handle_bytes": b"", "nbytes": 0}
    assert isinstance(_ipc_handler_for_handle(xpu_handle), XpuIPCHandler)

    # An unrelated dict must not be mistaken for the XPU handle.
    assert isinstance(_ipc_handler_for_handle({"foo": "bar"}), TorchIPCHandler)


def test_torch_handler_export_uses_reduce_tensor():
    sentinel = ("REDUCED",)
    with patch("checkpoint_engine.ipc_handler.reduce_tensor", return_value=sentinel) as m:
        t = TorchIPCHandler()
        out = t.export(SimpleNamespace())
    assert out is sentinel
    m.assert_called_once()


def test_xpu_export_returns_self_contained_handle():
    # The SYCL handle bytes travel over ZMQ as a picklable dict; no fd, no offset,
    # no companion socket. Mock the native extension so this runs on CPU CI.
    buffer = SimpleNamespace(data_ptr=lambda: 0xDEAD, nbytes=256)
    with patch("checkpoint_engine.xpu_ipc.get_handle", return_value=b"HANDLE") as m:
        handle = XpuIPCHandler().export(buffer)
    m.assert_called_once_with(0xDEAD)
    assert handle == {"kind": "xpu_sycl", "handle_bytes": b"HANDLE", "nbytes": 256}


def test_xpu_export_defers_release_until_detach():
    # The exporter handle must be released only in detach() (not export()), against
    # the exact pointer exported -- releasing early can free the fd under UR v2.
    buffer = SimpleNamespace(data_ptr=lambda: 0xBEEF, nbytes=128)
    with (
        patch("checkpoint_engine.xpu_ipc.get_handle", return_value=b"H"),
        patch("checkpoint_engine.xpu_ipc.release_handle") as release,
    ):
        t = XpuIPCHandler()
        t.export(buffer)
        release.assert_not_called()  # not released during export
        t.detach()
        release.assert_called_once_with(0xBEEF)
        # Idempotent: a second detach must not double-release.
        t.detach()
        release.assert_called_once_with(0xBEEF)


def test_xpu_handler_detach_is_safe_when_unused():
    # detach() before any export/attach must not raise (and must not touch the ext).
    with (
        patch("checkpoint_engine.xpu_ipc.release_handle") as release,
        patch("checkpoint_engine.xpu_ipc.close_handle") as close,
    ):
        XpuIPCHandler().detach()
    release.assert_not_called()
    close.assert_not_called()


def test_xpu_consumer_detach_closes_opened_mapping():
    # The consumer (attach) side unmaps its opened pointer on detach, and must not
    # try to release an exporter handle it never took.
    handle = {"kind": "xpu_sycl", "handle_bytes": b"H", "nbytes": 64}
    with (
        patch("checkpoint_engine.xpu_ipc.open_handle", return_value=0x7000),
        patch("checkpoint_engine.xpu_ipc.wrap_tensor") as wrap,
        patch("checkpoint_engine.ipc_handler.torch.xpu.synchronize"),
        patch("checkpoint_engine.xpu_ipc.close_handle") as close,
        patch("checkpoint_engine.xpu_ipc.release_handle") as release,
    ):
        import torch as _torch

        wrap.return_value = SimpleNamespace(dtype=_torch.uint8)
        t = XpuIPCHandler()
        t.attach(handle, device_id=0)
        t.detach()
    close.assert_called_once_with(0x7000)
    release.assert_not_called()
