"""Unit tests for the weight-transport seam (CPU-only, no accelerator required).

The zero-copy device handoff itself is exercised by a hardware-gated end-to-end
test on real XPU/CUDA. Here we cover the dispatch logic and wire formats.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from checkpoint_engine.transport import (
    IpcWeightTransport,
    XpuIpcWeightTransport,
    build_transport,
)
from checkpoint_engine.worker import _transport_for_handle


def _dm(device_type: str) -> object:
    return SimpleNamespace(device_type=device_type)


@pytest.mark.parametrize(
    "device_type,expected",
    [("cuda", IpcWeightTransport), ("npu", IpcWeightTransport), ("xpu", XpuIpcWeightTransport)],
)
def test_build_transport_dispatch(device_type: str, expected: type):
    assert isinstance(build_transport(_dm(device_type)), expected)


def test_consumer_dispatch_by_handle_shape():
    # CUDA/NPU send a reduce_tensor tuple; XPU sends a tagged dict.
    tuple_handle = (lambda *a: None, (1, 2, 3))
    assert isinstance(_transport_for_handle(tuple_handle), IpcWeightTransport)

    xpu_handle = {"kind": XpuIpcWeightTransport.kind, "handle_bytes": b"", "nbytes": 0}
    assert isinstance(_transport_for_handle(xpu_handle), XpuIpcWeightTransport)

    # An unrelated dict must not be mistaken for the XPU handle.
    assert isinstance(_transport_for_handle({"foo": "bar"}), IpcWeightTransport)


def test_ipc_transport_export_uses_reduce_tensor():
    sentinel = ("REDUCED",)
    with patch("checkpoint_engine.transport.reduce_tensor", return_value=sentinel) as m:
        t = IpcWeightTransport()
        out = t.export(SimpleNamespace())
    assert out is sentinel
    m.assert_called_once()


def test_xpu_export_returns_self_contained_handle():
    # The SYCL handle bytes travel over ZMQ as a picklable dict; no fd, no offset,
    # no companion socket. Mock the native extension so this runs on CPU CI.
    buffer = SimpleNamespace(data_ptr=lambda: 0xDEAD, nbytes=256)
    with patch("checkpoint_engine.xpu_ipc.get_handle", return_value=b"HANDLE") as m:
        handle = XpuIpcWeightTransport().export(buffer)
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
        t = XpuIpcWeightTransport()
        t.export(buffer)
        release.assert_not_called()  # not released during export
        t.detach()
        release.assert_called_once_with(0xBEEF)
        # Idempotent: a second detach must not double-release.
        t.detach()
        release.assert_called_once_with(0xBEEF)


def test_xpu_transport_detach_is_safe_when_unused():
    # detach() before any export/attach must not raise (and must not touch the ext).
    with (
        patch("checkpoint_engine.xpu_ipc.release_handle") as release,
        patch("checkpoint_engine.xpu_ipc.close_handle") as close,
    ):
        XpuIpcWeightTransport().detach()
    release.assert_not_called()
    close.assert_not_called()


def test_xpu_consumer_detach_closes_opened_mapping():
    # The consumer (attach) side unmaps its opened pointer on detach, and must not
    # try to release an exporter handle it never took.
    handle = {"kind": "xpu_sycl", "handle_bytes": b"H", "nbytes": 64}
    with (
        patch("checkpoint_engine.xpu_ipc.open_handle", return_value=0x7000),
        patch("checkpoint_engine.xpu_ipc.wrap_tensor") as wrap,
        patch("checkpoint_engine.transport.torch.xpu.synchronize"),
        patch("checkpoint_engine.xpu_ipc.close_handle") as close,
        patch("checkpoint_engine.xpu_ipc.release_handle") as release,
    ):
        import torch as _torch

        wrap.return_value = SimpleNamespace(dtype=_torch.uint8)
        t = XpuIpcWeightTransport()
        t.attach(handle, device_id=0)
        t.detach()
    close.assert_called_once_with(0x7000)
    release.assert_not_called()
