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
from pathlib import Path
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
# load_ext: with_sycl=True makes torch shell out to a bare "icpx", so the located
# compiler must be on PATH for the build and PATH restored afterwards.
# ------------------------------------------------------------------------------


def test_load_ext_puts_icpx_on_path_and_restores_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")

    captured: dict[str, object] = {}

    def fake_load(**kwargs: object) -> MagicMock:
        # Record what torch.utils.cpp_extension.load would see.
        captured["PATH"] = os.environ.get("PATH")
        captured["kwargs"] = kwargs
        return MagicMock()

    xpu_ipc.load_ext.cache_clear()
    try:
        with (
            patch("checkpoint_engine.xpu_ipc._find_icpx", return_value="/opt/oneapi/bin/icpx"),
            patch("torch.utils.cpp_extension.load", side_effect=fake_load),
        ):
            xpu_ipc.load_ext()
    finally:
        xpu_ipc.load_ext.cache_clear()

    # torch runs `icpx --version`, so its directory must lead PATH during the build.
    assert captured["PATH"] == "/opt/oneapi/bin:/usr/bin"
    # The SYCL toolchain comes from with_sycl; -O2 must stay or the host object is -O0.
    kwargs = captured["kwargs"]
    assert kwargs["with_sycl"] is True
    assert kwargs["extra_cflags"] == ["-O2"]
    # ...and the caller's environment must be restored afterwards.
    assert os.environ["PATH"] == "/usr/bin"


def test_icpx_version_key_orders_numerically() -> None:
    # Lexicographic sort would rank 2026.9 above 2026.10; the key must compare
    # version parts as numbers so the newest install really wins.
    paths = [f"/opt/intel/oneapi/compiler/{v}/bin/icpx" for v in ("2026.9", "2026.10", "2025.3")]
    ordered = sorted(paths, key=xpu_ipc._icpx_version_key, reverse=True)
    assert [p.split("/")[-3] for p in ordered] == ["2026.10", "2026.9", "2025.3"]
    # The non-numeric `latest` symlink points at the newest install, so it wins.
    with_latest = [*paths, "/opt/intel/oneapi/compiler/latest/bin/icpx"]
    best = max(with_latest, key=xpu_ipc._icpx_version_key)
    assert best.split("/")[-3] == "latest"


def test_find_icpx_falls_back_to_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # A sourced setvars.sh may put icpx on PATH without exporting CMPLR_ROOT, and
    # oneAPI need not live under /opt. Without the PATH fallback the compiler is
    # reported missing and XPU IPC is wrongly declared unavailable.
    monkeypatch.delenv("CMPLR_ROOT", raising=False)
    with (
        patch("checkpoint_engine.xpu_ipc.glob.glob", return_value=[]),
        patch("checkpoint_engine.xpu_ipc.shutil.which", return_value="/custom/bin/icpx") as which,
        patch("checkpoint_engine.xpu_ipc.os.path.exists", return_value=True),
        patch("checkpoint_engine.xpu_ipc._has_ipc_memory", return_value=True),
    ):
        assert xpu_ipc._find_icpx() == "/custom/bin/icpx"
    which.assert_called_once_with("icpx")


def test_find_icpx_skips_compiler_without_ipc_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    # An icpx older than oneAPI 2026.0 builds a device image that torch's libsycl
    # cannot load, aborting the process (SIGABRT) on dlopen -- uncatchable from
    # Python. Such compilers must be rejected up front, before any build.
    monkeypatch.setenv("CMPLR_ROOT", "/opt/intel/oneapi/compiler/2025.3")
    with (
        patch("checkpoint_engine.xpu_ipc.glob.glob", return_value=[]),
        patch("checkpoint_engine.xpu_ipc.shutil.which", return_value=None),
        patch("checkpoint_engine.xpu_ipc.os.path.exists", return_value=True),
        patch("checkpoint_engine.xpu_ipc._has_ipc_memory", return_value=False),
    ):
        assert xpu_ipc._find_icpx() is None


def test_has_ipc_memory_probes_the_header(tmp_path: Path) -> None:
    # The discriminator is the header's presence in the compiler's own tree
    # (<root>/bin/icpx -> <root>/include/sycl/.../ipc_memory.hpp).
    icpx = tmp_path / "bin" / "icpx"
    icpx.parent.mkdir(parents=True)
    icpx.touch()
    assert xpu_ipc._has_ipc_memory(str(icpx)) is False

    header = tmp_path / "include" / "sycl" / "ext" / "oneapi" / "experimental" / "ipc_memory.hpp"
    header.parent.mkdir(parents=True)
    header.touch()
    assert xpu_ipc._has_ipc_memory(str(icpx)) is True


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
# update(): the exported IPC handle is retained until detach(). A failure inside
# _update_per_bucket must still release it -- otherwise the exporter handle leaks
# on every failed weight update.
# ------------------------------------------------------------------------------


def test_update_releases_ipc_handle_when_update_fails():
    from checkpoint_engine.ipc_handler import IPCHandler
    from checkpoint_engine.ps import ParameterServer

    # A real IPCHandler (not a MagicMock) so detach() must be reached through the
    # context manager's __exit__ rather than by mocked attribute access.
    class RecordingHandler(IPCHandler):
        def __init__(self) -> None:
            self.detached = 0

        def export(self, buffer: object) -> dict:
            return {"kind": "fake"}

        def attach(self, handle: object, device_id: int) -> None:
            raise AssertionError("attach is the consumer side; not used here")

        def detach(self) -> None:
            self.detached += 1

    ps = ParameterServer.__new__(ParameterServer)
    ps._rank = 0
    ps._auto_pg = False
    ps.device_manager = SimpleNamespace(
        device_type="cpu",
        device_module=SimpleNamespace(
            empty_cache=lambda: None,
            memory_allocated=lambda: 0,
            memory_reserved=lambda: 0,
        ),
    )
    handler = RecordingHandler()

    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch("checkpoint_engine.ps.build_ipc_handler", return_value=handler),
        patch.object(
            ps, "_update_per_bucket", side_effect=RuntimeError("update failed")
        ) as per_bucket,
        pytest.raises(RuntimeError, match="update failed"),
    ):
        ps.update("ckpt", req_func=lambda _paths: None)

    # The handler is handed to _update_per_bucket and released regardless of outcome.
    assert handler in per_bucket.call_args.args
    assert handler.detached == 1


def test_update_releases_ipc_handle_on_success():
    from checkpoint_engine.ipc_handler import IPCHandler
    from checkpoint_engine.ps import ParameterServer

    class RecordingHandler(IPCHandler):
        def __init__(self) -> None:
            self.detached = 0

        def export(self, buffer: object) -> dict:
            return {"kind": "fake"}

        def attach(self, handle: object, device_id: int) -> None:
            raise AssertionError("attach is the consumer side; not used here")

        def detach(self) -> None:
            self.detached += 1

    ps = ParameterServer.__new__(ParameterServer)
    ps._rank = 0
    ps._auto_pg = False
    ps.device_manager = SimpleNamespace(
        device_type="cpu",
        device_module=SimpleNamespace(
            empty_cache=lambda: None,
            memory_allocated=lambda: 0,
            memory_reserved=lambda: 0,
        ),
    )
    handler = RecordingHandler()

    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch("checkpoint_engine.ps.build_ipc_handler", return_value=handler),
        patch.object(ps, "_update_per_bucket"),
        patch.object(ps, "store_based_barrier"),
    ):
        ps.update("ckpt", req_func=lambda _paths: None)

    assert handler.detached == 1


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
