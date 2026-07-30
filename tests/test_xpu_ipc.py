"""Hardware-gated tests for the native SYCL IPC memory handler on Intel XPU.

These are skipped unless an Intel GPU is present and the SYCL IPC extension can
be built (needs an oneAPI ``icpx`` with SYCL ipc_memory support). They are marked
``gpu`` so CPU-only CI (``-m "not gpu"``) skips them.
"""

import os

import pytest
import torch


pytestmark = pytest.mark.gpu


def _xpu_ipc_available() -> bool:
    try:
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return False
        from checkpoint_engine import xpu_ipc

        return xpu_ipc.is_available()
    except Exception:  # noqa: BLE001
        return False


skip_no_xpu_ipc = pytest.mark.skipif(
    not _xpu_ipc_available(), reason="Intel XPU with buildable SYCL ipc_memory extension required"
)

_N_TENSORS = 8


def _gen_tensors() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return {
        f"w{i}": (torch.randn(128 + i, 64, generator=gen) * 50).to(torch.bfloat16)
        for i in range(_N_TENSORS)
    }


def _worker_proc(
    device_uuid: str,
    expected: dict[str, torch.Tensor],
    inq: object,
    outq: object,
) -> None:
    # Module-level so the "spawn" start method can pickle it.
    import zmq

    from checkpoint_engine.worker import update_weights_from_ipc

    torch.xpu.set_device(0)
    exp = {k: v.to("xpu:0") for k, v in expected.items()}
    ctx = zmq.Context()
    state = {"n": 0, "ok": True}

    def run(weights: list[tuple[str, torch.Tensor]]) -> None:
        for name, w in weights:
            if name in exp and not torch.equal(w.to(torch.bfloat16), exp[name]):
                state["ok"] = False
            elif name in exp:
                state["n"] += 1

    while True:
        socket_paths = inq.get()
        if socket_paths is None:
            break
        update_weights_from_ipc(
            ctx,
            dict(socket_paths)[device_uuid],
            device_id=0,
            run=run,
            post_hook=lambda: torch.xpu.synchronize(),
        )
    outq.put((state["n"], state["ok"]))


@skip_no_xpu_ipc
def test_sycl_ipc_same_process_roundtrip():
    """get_handle -> open_handle (same process) maps back to the original bytes."""
    from checkpoint_engine import xpu_ipc

    torch.xpu.set_device(0)
    t = torch.arange(256, device="xpu:0", dtype=torch.uint8)
    torch.xpu.synchronize()

    handle_bytes = xpu_ipc.get_handle(t.data_ptr())
    ptr = xpu_ipc.open_handle(handle_bytes, 0)
    wrapped = xpu_ipc.wrap_tensor(ptr, t.numel(), 0)
    try:
        assert torch.equal(wrapped, t)
    finally:
        xpu_ipc.close_handle(ptr)


@skip_no_xpu_ipc
def test_sycl_ipc_interior_pointer_offset_preserved():
    """A SYCL IPC handle for an interior (sub-allocation) pointer must reopen at
    the same offset -- the offset is encoded in the portable handle bytes, not
    carried separately."""
    from checkpoint_engine import xpu_ipc

    torch.xpu.set_device(0)
    big = torch.zeros(4096, device="xpu:0", dtype=torch.uint8)
    view = big[1024:1280]
    view.fill_(0xAB)
    torch.xpu.synchronize()

    handle_bytes = xpu_ipc.get_handle(view.data_ptr())
    ptr = xpu_ipc.open_handle(handle_bytes, 0)
    wrapped = xpu_ipc.wrap_tensor(ptr, view.numel(), 0)
    try:
        assert torch.equal(wrapped, view)
    finally:
        xpu_ipc.close_handle(ptr)


@skip_no_xpu_ipc
def test_sycl_ipc_cross_process_broadcast():
    """Full ParameterServer broadcast -> colocated worker over the XPU SYCL handler."""
    from torch.multiprocessing import get_context

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29570")

    from checkpoint_engine.ps import ParameterServer, _get_physical_gpu_id

    tensors = _gen_tensors()

    torch.xpu.set_device(0)
    ps = ParameterServer(auto_pg=True)
    uuid = _get_physical_gpu_id(ps.device_manager, 0)

    mp = get_context("spawn")
    inq, outq = mp.Queue(), mp.Queue()
    proc = mp.Process(target=_worker_proc, args=(uuid, tensors, inq, outq))
    proc.start()
    try:
        ps.register_checkpoint("ckpt", named_tensors=tensors)
        ps.init_process_group()
        ps.gather_metas("ckpt")
        ps.update("ckpt", inq.put)
        inq.put(None)
        n, ok = outq.get(timeout=60)
        assert ok, "received weights did not match originals"
        assert n == _N_TENSORS, f"expected {_N_TENSORS} tensors checked, got {n}"
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
