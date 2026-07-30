"""The P2P update path must reject devices whose memory Mooncake cannot register.

XPU device memory has no Level Zero backend in Mooncake, so a P2P update
(``ranks`` set) must raise a clear error before touching the transfer engine.
CPU-only: we stub the ParameterServer internals up to the guard.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import checkpoint_engine.distributed as dist
from checkpoint_engine.ps import ParameterServer


def _ps_with_device(device_type: str, *, supports_ipc: bool, supports_p2p: bool) -> ParameterServer:
    ps = ParameterServer.__new__(ParameterServer)
    ps._rank = 0
    ps.device_manager = SimpleNamespace(
        device_type=device_type,
        supports_device_ipc=lambda: supports_ipc,
        supports_device_p2p=lambda: supports_p2p,
    )
    # Non-empty metas so the leading assert passes; content is irrelevant (guard fires first).
    ps._current_global_parameter_metas = {0: object()}
    return ps


def test_p2p_update_rejected_on_xpu():
    ps = _ps_with_device("xpu", supports_ipc=True, supports_p2p=False)
    ipc_handler = MagicMock()
    with (
        patch.object(dist, "is_initialized", return_value=True),
        pytest.raises(RuntimeError, match=r"P2P weight update .* is not supported"),
    ):
        ps._update_per_bucket(
            "ckpt",
            req_func=lambda _paths: None,
            ipc_handler=ipc_handler,
            ranks_group=None,
            ranks=[0],
        )
    # The guard must fire before any handle is exported.
    ipc_handler.export.assert_not_called()


def test_ipc_unavailable_rejected():
    ps = _ps_with_device("xpu", supports_ipc=False, supports_p2p=False)
    ipc_handler = MagicMock()
    with (
        patch.object(dist, "is_initialized", return_value=True),
        pytest.raises(RuntimeError, match="cross-process device-tensor IPC"),
    ):
        ps._update_per_bucket(
            "ckpt",
            req_func=lambda _paths: None,
            ipc_handler=ipc_handler,
            ranks_group=None,
            ranks=None,
        )
    ipc_handler.export.assert_not_called()
