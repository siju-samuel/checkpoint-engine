"""CPU-only parity tests for the XPU support paths that diverge from CUDA/NPU.

These cover device-agnostic logic that would otherwise have no unit coverage:
* The portable-handle contract of ``xpu_ipc.get_handle``/``open_handle`` (bytes
  in, bytes out -- no fd, no offset).
* The custom-distributed backend rejection for XPU.
* ``register_checkpoint`` forcing in-place pinning off on non-CUDA devices.

The zero-copy device handoff itself is exercised by the hardware-gated tests in
``test_xpu_ipc.py``; here we isolate the surrounding logic so it runs on CPU-only
CI (``-m "not gpu"``).
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import checkpoint_engine.distributed as dist
from checkpoint_engine import xpu_ipc
from checkpoint_engine.distributed.base import TorchBackend, use_backend


# ------------------------------------------------------------------------------
# get_handle / open_handle: a SYCL ipc_memory handle is self-contained portable
# bytes -- get_handle returns the raw blob and open_handle passes it straight to
# the extension (offset and any fd are encoded inside the bytes, not carried).
# ------------------------------------------------------------------------------


def test_get_handle_returns_raw_portable_bytes():
    fake_ext = MagicMock()
    blob = list(b"\xab" * 120)  # SYCL handle blob (opaque, self-contained)
    fake_ext.ipc_get_handle.return_value = blob
    with patch("checkpoint_engine.xpu_ipc.load_ext", return_value=fake_ext):
        handle_bytes = xpu_ipc.get_handle(0xDEAD)
    fake_ext.ipc_get_handle.assert_called_once_with(0xDEAD)
    assert handle_bytes == bytes(blob)


def test_open_handle_passes_bytes_and_device_through():
    fake_ext = MagicMock()
    fake_ext.ipc_open_handle.return_value = 0x5000  # mapped ptr (offset already applied)
    with patch("checkpoint_engine.xpu_ipc.load_ext", return_value=fake_ext):
        ptr = xpu_ipc.open_handle(b"\xab" * 120, device=2)
    assert ptr == 0x5000
    (blob_arg, device_arg) = fake_ext.ipc_open_handle.call_args.args
    assert device_arg == 2
    assert bytes(blob_arg) == b"\xab" * 120


# ------------------------------------------------------------------------------
# load_ext: the -fsycl build must force CC/CXX to icx/icpx (a conda/CI env often
# exports CXX=g++, which cannot compile -fsycl) and restore the env afterwards.
# ------------------------------------------------------------------------------


def test_load_ext_forces_icpx_over_existing_gpp_and_restores_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CXX", "/usr/bin/g++")  # what conda gxx_linux-64 exports
    monkeypatch.delenv("CC", raising=False)

    captured: dict[str, str | None] = {}

    def fake_load(**kwargs: object) -> MagicMock:
        # Record what torch.utils.cpp_extension.load would see.
        captured["CC"] = os.environ.get("CC")
        captured["CXX"] = os.environ.get("CXX")
        return MagicMock()

    xpu_ipc.load_ext.cache_clear()
    try:
        with (
            patch("checkpoint_engine.xpu_ipc._find_icpx", return_value="/opt/oneapi/bin/icpx"),
            patch("checkpoint_engine.xpu_ipc._find_sycl_include_dir", return_value=None),
            patch("torch.utils.cpp_extension.load", side_effect=fake_load),
        ):
            xpu_ipc.load_ext()
    finally:
        xpu_ipc.load_ext.cache_clear()

    # During the build the SYCL compiler must win over the inherited g++.
    assert captured["CXX"] == "/opt/oneapi/bin/icpx"
    assert captured["CC"] == "/opt/oneapi/bin/icx"
    # ...and the caller's environment must be restored afterwards.
    assert os.environ["CXX"] == "/usr/bin/g++"
    assert "CC" not in os.environ


def test_find_icpx_falls_back_to_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # A sourced setvars.sh may put icpx on PATH without exporting CMPLR_ROOT, and
    # oneAPI need not live under /opt. Without the PATH fallback the compiler is
    # reported missing and XPU IPC is wrongly declared unavailable.
    monkeypatch.delenv("CMPLR_ROOT", raising=False)
    with (
        patch("checkpoint_engine.xpu_ipc.glob.glob", return_value=[]),
        patch("checkpoint_engine.xpu_ipc.shutil.which", return_value="/custom/bin/icpx") as which,
    ):
        assert xpu_ipc._find_icpx() == "/custom/bin/icpx"
    which.assert_called_once_with("icpx")


def test_is_available_caches_only_success_and_retries_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import checkpoint_engine.xpu_ipc as mod

    monkeypatch.setattr(mod, "_AVAILABLE", False)
    calls = {"n": 0}

    def flaky_load() -> MagicMock:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient: compiler env not warm yet")
        return MagicMock()

    fake_torch = SimpleNamespace(xpu=SimpleNamespace(is_available=lambda: True))
    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch.object(mod, "load_ext", side_effect=flaky_load),
    ):
        assert mod.is_available() is False  # transient failure NOT cached
        assert mod.is_available() is True  # retried, now succeeds
        assert mod.is_available() is True  # success cached (no third load_ext)
    assert calls["n"] == 2


# ------------------------------------------------------------------------------
# Custom distributed backend: XPU must reject custom_dist and fall back to the
# native "xccl" TorchBackend (there is no vLLM PyXcclCommunicator to subclass).
# ------------------------------------------------------------------------------


def test_use_backend_rejects_xpu_custom_dist():
    with pytest.raises(ValueError, match="XPU is not supported here"):
        use_backend("vllm_xccl")


def test_use_backend_none_keeps_default_torch_backend():
    # A falsy backend must leave the (default) TorchBackend in place, which is what
    # XPU relies on for xccl.
    before = dist.is_initialized  # attribute presence sanity
    use_backend(None)
    from checkpoint_engine.distributed.base import _BACKEND_INSTANCE

    assert isinstance(_BACKEND_INSTANCE, TorchBackend)
    assert before is dist.is_initialized


# ------------------------------------------------------------------------------
# register_checkpoint: in-place pinning (cudaHostRegister) is CUDA-only; on XPU
# it must be silently disabled rather than attempted.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# _update_per_bucket: the exported IPC handle is retained until detach(). A
# failure after export but before the broadcast loop (e.g. the ZMQ bind) must
# still release it -- otherwise the exporter handle leaks on every failed update.
# ------------------------------------------------------------------------------


def test_update_per_bucket_detaches_transport_on_early_failure():
    from checkpoint_engine.ps import ParameterServer

    ps = ParameterServer.__new__(ParameterServer)
    ps._rank = 0
    ps.device_manager = SimpleNamespace(
        device_type="cpu",
        supports_device_ipc=lambda: True,
        supports_device_p2p=lambda: False,
    )
    ps._current_global_parameter_metas = {0: object()}
    ps._local_rdma_devices = None
    ps._remote_rdma_devices = None

    fake_transport = MagicMock()
    fake_transport.export.return_value = {"kind": "fake"}

    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch("checkpoint_engine.ps.build_transport", return_value=fake_transport),
        patch("checkpoint_engine.ps._gen_h2d_buckets", return_value=[]),
        patch.object(ps, "_detect_bucket_size", return_value=(16, False)),
        patch.object(ps, "_bind_zmq_socket", side_effect=RuntimeError("bind failed")),
        pytest.raises(RuntimeError, match="bind failed"),
    ):
        ps._update_per_bucket("ckpt", req_func=lambda _paths: None, ranks_group=None, ranks=None)

    # export happened, so the exporter handle is live and must be released even
    # though the failure struck before the broadcast loop's own cleanup.
    fake_transport.export.assert_called_once()
    fake_transport.detach.assert_called_once_with()


def test_register_checkpoint_disables_inplace_pin_on_xpu():
    from checkpoint_engine.ps import ParameterServer

    ps = ParameterServer.__new__(ParameterServer)
    ps._rank = 0
    ps.device_manager = SimpleNamespace(
        device_type="xpu",
        supports_inplace_pin=lambda: False,
    )
    ps._memory_pool = {}
    ps._current_shared_memory_pool_user = ""
    ps._p2p_store = None
    ps.shared_memory_pool_name = ParameterServer.shared_memory_pool_name

    captured = {}

    def fake_register(*, inplace_pin: bool, **kwargs: object) -> list:
        captured["inplace_pin"] = inplace_pin
        return []

    with patch("checkpoint_engine.ps._register_checkpoint", side_effect=fake_register):
        ps.register_checkpoint("ckpt", named_tensors={}, use_inplace_pin_memory=True)
    # Requested True, but XPU cannot in-place pin -> must be forced False.
    assert captured["inplace_pin"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
