"""Unit tests for DeviceManager multi-accelerator dispatch (cuda / npu / xpu).

These tests mock the device backends so they run on CPU-only CI (``-m "not gpu"``). A separate
hardware-gated test exercises the real ``torch.xpu`` path when an Intel GPU is present.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from checkpoint_engine.device_utils import DeviceManager


def _make_manager(device_type: str, device_module: object) -> DeviceManager:
    """Build a DeviceManager without touching real hardware by stubbing detection/setup."""
    dm = DeviceManager.__new__(DeviceManager)
    dm.device_type = device_type
    dm.device_module = device_module
    return dm


@pytest.mark.parametrize(
    "device_type,expected_backend",
    [("cuda", "nccl"), ("npu", "hccl"), ("xpu", "xccl")],
)
def test_backend_mapping(device_type: str, expected_backend: str):
    dm = _make_manager(device_type, SimpleNamespace())
    assert dm.backend == expected_backend


def test_backend_unsupported():
    dm = _make_manager("tpu", SimpleNamespace())
    with pytest.raises(TypeError):
        _ = dm.backend


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_transfer_engine_protocol_rdma(device_type: str):
    dm = _make_manager(device_type, SimpleNamespace())
    with patch("checkpoint_engine.device_utils.has_efa_pci", return_value=False):
        assert dm.transfer_engine_protocol == "rdma"
    with patch("checkpoint_engine.device_utils.has_efa_pci", return_value=True):
        assert dm.transfer_engine_protocol == "efa"


def test_transfer_engine_protocol_npu():
    dm = _make_manager("npu", SimpleNamespace())
    assert dm.transfer_engine_protocol == "ascend_direct"


def test_ipc_collect_present_is_called():
    called = []
    module = SimpleNamespace(ipc_collect=lambda: called.append(True))
    dm = _make_manager("cuda", module)
    dm.ipc_collect()
    assert called == [True]


def test_ipc_collect_xpu_is_noop():
    # SYCL frees on close_handle, so XPU has no handle cache to collect (and
    # torch.xpu has no ipc_collect); it must be a no-op, not an error.
    dm = _make_manager("xpu", SimpleNamespace())
    dm.ipc_collect()


def test_ipc_collect_rejects_unsupported_device():
    # An unsupported backend must fail loudly rather than silently skipping the
    # collect, matching backend/transfer_engine_protocol/_setup_device_module.
    dm = _make_manager("tpu", SimpleNamespace())
    with pytest.raises(TypeError, match="not supported"):
        dm.ipc_collect()


@pytest.mark.parametrize(
    "device_type,expected",
    [("cuda", True), ("npu", False), ("xpu", False)],
)
def test_supports_inplace_pin(device_type: str, expected: bool):
    dm = _make_manager(device_type, SimpleNamespace())
    assert dm.supports_inplace_pin() is expected


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_supports_device_ipc_true_for_cuda_npu(device_type: str):
    dm = _make_manager(device_type, SimpleNamespace())
    assert dm.supports_device_ipc() is True


def test_supports_device_ipc_xpu_uses_sycl_extension():
    # On XPU we do not rely on torch reductions (PyTorch has no XPU tensor IPC);
    # instead we detect our native SYCL ipc_memory extension via xpu_ipc.is_available().
    dm = _make_manager("xpu", SimpleNamespace())
    with patch("checkpoint_engine.xpu_ipc.is_available", return_value=True):
        assert dm.supports_device_ipc() is True
    with patch("checkpoint_engine.xpu_ipc.is_available", return_value=False):
        assert dm.supports_device_ipc() is False


def test_supports_device_ipc_unknown_device():
    dm = _make_manager("tpu", SimpleNamespace())
    assert dm.supports_device_ipc() is False


@pytest.mark.parametrize(
    "device_type,expected",
    [("cuda", True), ("npu", True), ("xpu", False), ("tpu", False)],
)
def test_supports_device_p2p(device_type: str, expected: bool):
    # Mooncake has no Level Zero backend, so XPU device-memory P2P is unsupported.
    dm = _make_manager(device_type, SimpleNamespace())
    assert dm.supports_device_p2p() is expected


def test_host_empty_cache_noncuda_uses_gc():
    dm = _make_manager("xpu", SimpleNamespace())
    with patch("checkpoint_engine.device_utils.gc.collect") as gc_collect:
        dm.host_empty_cache()
        gc_collect.assert_called_once()


def test_host_empty_cache_cuda_uses_torch():
    dm = _make_manager("cuda", SimpleNamespace())
    with patch.object(torch._C, "_host_emptyCache", create=True) as host_empty:
        dm.host_empty_cache()
        host_empty.assert_called_once()


# --------------------------------------------------------------------------------------------
# Hardware-gated: real torch.xpu behavior on an Intel GPU host.
# --------------------------------------------------------------------------------------------

_HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_XPU, reason="requires an Intel XPU device")
def test_real_xpu_device_manager():
    dm = DeviceManager()
    assert dm.device_type == "xpu"
    assert dm.backend == "xccl"
    assert dm.device_module is torch.xpu
    # ipc_collect must be a harmless no-op (torch.xpu has no ipc_collect).
    dm.ipc_collect()
    # XPU has no torch-native device-tensor IPC, but checkpoint-engine ships its own
    # native SYCL IPC memory handler; supports_device_ipc() must agree with whether
    # that extension can actually be built/loaded in this environment.
    from checkpoint_engine import xpu_ipc

    assert dm.supports_device_ipc() is xpu_ipc.is_available()
    # Mooncake has no Level Zero backend, so device-memory P2P stays unsupported on XPU.
    assert dm.supports_device_p2p() is False
    # cudaHostRegister-style in-place pinning is CUDA-only.
    assert dm.supports_inplace_pin() is False


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_XPU, reason="requires an Intel XPU device")
def test_real_xpu_device_ipc_available_when_extension_builds():
    """When the SYCL ipc_memory extension builds on real XPU hardware, the
    broadcast path must be reported as supported. This guards the torch>=2.14
    c++20 build regression that silently disabled XPU broadcast."""
    from checkpoint_engine import xpu_ipc

    if not xpu_ipc.is_available():
        pytest.skip("SYCL ipc_memory extension could not be built in this environment")
    dm = DeviceManager()
    assert dm.supports_device_ipc() is True


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_XPU, reason="requires an Intel XPU device")
def test_real_xpu_physical_uuid_matches_worker_format():
    from checkpoint_engine.ps import _get_physical_gpu_id

    dm = DeviceManager()
    uuid0 = _get_physical_gpu_id(dm, 0)
    # Same format the vLLM worker derives from torch.xpu.get_device_properties(idx).uuid.
    expected = f"GPU-{torch.xpu.get_device_properties(0).uuid!s}"
    assert uuid0 == expected
    if dm.device_module.device_count() > 1:
        assert _get_physical_gpu_id(dm, 1) != uuid0, "per-device uuids must be distinct"
