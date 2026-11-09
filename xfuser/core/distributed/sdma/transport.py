"""Collective data movement on the SDMA copy engines (ROCm).

RCCL moves data with kernels on the compute units, and a kernel that needs the
data waits for all of it. This transport moves it with ROCr's
`hsa_amd_memory_async_copy_on_engine` instead: each rank pushes into its peers'
buffers on one SDMA engine per peer, all peers at once, and uses no compute
units, so compute that does not need the data runs alongside.

`PeerGroup` implements interface.PeerTransport. A route's copies are registered
once as a `Plan` (per peer: source offsets, destination addresses, sizes and the
flag to set after them) and issued per call with that call's source address, so
the per-call host work is one loop in C.

Buffers that peers write into (`SharedBuffer`) are allocated by the library
with hipMalloc and mapped into every peer with HIP IPC handles, exchanged over
the group's device process group. A signal is a flag word: channel c holds one
word per sender at each rank, in uncached memory, so a kernel polling them sees
the engines' writes. Engines: see sx_engine_plan in transport.hip.

Ordering:
- A copy waits for a trigger that the compute stream releases, so it never
  reads data the stream has not written yet. Each send of a call has a trigger
  of its own.
- Behind a peer's copies, a 4-byte write on the same engine sets this rank's
  flag word at that peer; the engine runs them in order, and the flag lands
  after the data, and after every earlier send's copies to that peer, flagged
  or not (tests/integration/accelerator/core/distributed/test_sdma_ulysses.py
  checks every element each call reads).
- The receiving stream waits for its flag words with a one-thread kernel
  (`Call.wait`), which resets them for their next use.
- A call takes one of _SLOTS slots, whose completion signals its copies take
  as they are issued; the slot's next call first waits for exactly those.

Waits are bounded. After XDIT_SDMA_WAIT_S seconds (default 30) the wait kernel
records which flags never landed. By default (`abort`) the kernel keeps the
stream held, so nothing runs on data that never came, and a watchdog thread logs
what was missing and aborts the process, as NCCL's async error handling does.
Tests can call `set_on_timeout("raise")`: the stream then moves on (that call's
output is wrong) and the next call on this process raises an error naming the
missing flags.

The library is loaded with ctypes, from the first of: XDIT_SDMA_LIB; a library
built into this package's directory from the current source (`python
xfuser/core/distributed/sdma/build.py --out <this directory>`, for image builds:
no GPU and no xfuser import needed); a library built with hipcc for this GPU on
first use and cached. The transport runs on the GPU architectures in
VALIDATED_ARCHS; others need XDIT_SDMA_ALLOW_UNTESTED=1. ROCm only, within one
node.
"""

import ctypes
import fcntl
import hashlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Hashable, List, Tuple

import torch
import torch.distributed as dist

import xfuser.envs as envs
from xfuser.core.distributed.sdma.build import LIB_NAME, SOURCE, build, source_hash
from xfuser.logger import init_logger

logger = init_logger(__name__)

# GPU architectures the transport is validated on; others need XDIT_SDMA_ALLOW_UNTESTED=1.
VALIDATED_ARCHS = ("gfx942", "gfx950")
_SLOTS = 4
_INITIAL_COPIES = 256  # signals per slot to start with; a slot grows when a call needs more
_TRIGGERS = 32  # sends per call (kTriggers in transport.hip)
_CHANNELS = 64
MAX_FLAGS_PER_WAIT = 64

_lib = None
_lib_path = None
_groups: Dict[Tuple[int, ...], "PeerGroup"] = {}
_by_process_group: Dict[dist.ProcessGroup, "PeerGroup"] = {}
_slot = 0
_call = 0
_watchdog = None
_init_lock = threading.Lock()


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with {rc}")


def _env(name: str) -> str:
    return envs.environment_variables[name]().strip()


def wait_seconds() -> float:
    return float(_env("XDIT_SDMA_WAIT_S"))


_on_timeout = "abort"


def on_timeout() -> str:
    """What a flag wait that gives up does: abort the process, or let the next call raise."""
    return _on_timeout


def set_on_timeout(mode: str) -> str:
    """Set `on_timeout` to "abort" (the default) or "raise" (tests); returns the previous mode.
    Set it before the transport's first use in the process: abort mode starts a watchdog."""
    global _on_timeout
    if mode not in ("abort", "raise"):
        raise ValueError(f"on_timeout must be abort or raise; got {mode!r}")
    previous, _on_timeout = _on_timeout, mode
    return previous


def _arch() -> str:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]


def available() -> bool:
    """Whether this process can use the transport at all: a ROCm build with a GPU."""
    return torch.version.hip is not None and torch.cuda.is_available()


def _library_path() -> Path:
    """XDIT_SDMA_LIB if set; else a library built for this GPU, cached by source, arch and HIP version."""
    prebuilt = _env("XDIT_SDMA_LIB")
    if prebuilt:
        if not Path(prebuilt).is_file():
            raise RuntimeError(f"XDIT_SDMA_LIB={prebuilt} is not a file")
        return Path(prebuilt)
    packaged = SOURCE.with_name(LIB_NAME)
    stamp = SOURCE.with_name(LIB_NAME + ".source-sha1")
    if packaged.is_file() and stamp.is_file() and stamp.read_text().strip() == source_hash():
        return packaged
    arch = _arch()
    key = hashlib.sha1(SOURCE.read_bytes() + f" {arch} {torch.version.hip}".encode()).hexdigest()[:12]
    cache = Path(os.environ.get("XDIT_CACHE_DIR", Path.home() / ".cache" / "xfuser")) / f"sdma_{key}"
    so = cache / LIB_NAME
    if not so.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with open(cache / "build.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not so.exists():
                build(cache, [arch])
    return so


def load():
    """Load the transport library (prebuilt or built on first use)."""
    global _lib, _lib_path
    if _lib is not None:
        return _lib
    _lib_path = _library_path()
    lib = ctypes.CDLL(str(_lib_path))
    i, u64, vp, dbl = ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_double
    ip, u64p = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64)
    for name, argtypes in {
        "sx_init": [i, i, i],
        "sx_engine_plan": [i, ip, i, i, ip],
        "sx_bus_of": [i],
        "sx_malloc": [u64, i, u64p],
        "sx_free": [u64],
        "sx_memset": [u64, i, u64],
        "sx_ipc_handle_size": [],
        "sx_ipc_get": [u64, vp],
        "sx_ipc_open": [vp, u64p],
        "sx_ipc_close": [u64],
        "sx_arm": [i],
        "sx_release": [i, i, vp],
        "sx_plan_create": [i, ip, ip, u64p, u64p, u64p, u64p],
        "sx_plan_free": [i],
        "sx_plan_issue": [i, i, i, u64, u64],
        "sx_wait_flags": [vp, u64, i, i, dbl, i, i, i],
        "sx_take_error": [ip, ip, u64p],
        "sx_peek_error": [ip, ip, u64p],
        "sx_debug_drop_flag": [i, i, i],
    }.items():
        getattr(lib, name).argtypes = argtypes
        getattr(lib, name).restype = i
    _lib = lib
    return lib


class _CudaArray:
    """Exposes raw device memory to torch.as_tensor through __cuda_array_interface__."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 2,
            "strides": None,
        }


def device_bytes(ptr: int, nbytes: int, device: torch.device) -> torch.Tensor:
    """A uint8 tensor viewing `nbytes` of device memory at `ptr` (not owned by torch)."""
    return torch.as_tensor(_CudaArray(ptr, nbytes), device=device)


class SharedBuffer:
    """Device memory of the same size on every rank of a group, mapped into every peer
    (interface.SymmetricBuffer).

    `local` is this rank's buffer as a uint8 tensor; `peer_bases` holds the
    peers' base addresses in this process, in `PeerGroup.peers` order. `uncached`
    memory is for flag words. Collective: every rank of the group creates (and
    closes) it together.
    """

    def __init__(self, group: "PeerGroup", nbytes: int, uncached: bool = False):
        lib = load()
        self.nbytes = nbytes
        self.uncached = uncached
        # False once released: routes naming it are rebuilt.
        self.live = True
        ptr = ctypes.c_uint64()
        _check(lib.sx_malloc(nbytes, int(uncached), ctypes.byref(ptr)), f"sx_malloc({nbytes}, uncached={uncached})")
        self.ptr = ptr.value
        _check(lib.sx_memset(self.ptr, 0, nbytes), "sx_memset")
        size = lib.sx_ipc_handle_size()
        handle = (ctypes.c_uint8 * size)()
        _check(lib.sx_ipc_get(self.ptr, handle), "sx_ipc_get")
        mine = torch.tensor(list(handle), dtype=torch.uint8, device=group.device)
        every = torch.empty(group.world * size, dtype=torch.uint8, device=group.device)
        dist.all_gather_into_tensor(every, mine, group=group.process_group)
        every = every.view(group.world, size).cpu()
        self._opened = []
        for p in group.peers:
            theirs = (ctypes.c_uint8 * size)(*every[p].tolist())
            out = ctypes.c_uint64()
            _check(lib.sx_ipc_open(theirs, ctypes.byref(out)), f"sx_ipc_open (peer {p})")
            self._opened.append(out.value)
        self.peer_bases = (ctypes.c_uint64 * len(self._opened))(*self._opened)
        self.local = device_bytes(self.ptr, nbytes, group.device)

    def unmap(self) -> None:
        """Unmap the peers' buffers; every rank must be past its last use of them."""
        self.live = False
        lib = load()
        for p in self._opened:
            lib.sx_ipc_close(p)
        self._opened = []

    def free(self) -> None:
        """Free this rank's buffer; every peer must have unmapped it: a buffer freed while a peer
        still maps it leaves later allocations that HIP cannot export."""
        self.local = None
        load().sx_free(self.ptr)


class PeerGroup:
    """Per process group (interface.PeerTransport): the peers in push order, one SDMA engine per
    peer, named shared buffers that grow on demand, and the routes registered on them."""

    channels = _CHANNELS
    # Copies are queued from the host: a CUDA graph capture would not replay them.
    host_issued = True

    def __init__(self, process_group: dist.ProcessGroup):
        lib = load()
        self.process_group = process_group
        self.rank, self.world = process_group.rank(), process_group.size()
        self.device = torch.device("cuda", torch.cuda.current_device())
        arch = _arch()
        if arch not in VALIDATED_ARCHS and _env("XDIT_SDMA_ALLOW_UNTESTED") != "1":
            raise RuntimeError(
                f"the SDMA transport is validated on {', '.join(VALIDATED_ARCHS)}, not {arch}; set "
                "XDIT_SDMA_ALLOW_UNTESTED=1 to run it anyway"
            )
        rc = lib.sx_init(self.device.index, _INITIAL_COPIES, _SLOTS)
        if rc == -11:
            raise RuntimeError(f"{_lib_path} has no GPU code for {arch}; rebuild it with --offload-arch {arch}")
        _check(rc, "sx_init")
        bus = torch.tensor([lib.sx_bus_of(self.device.index)], dtype=torch.int32, device=self.device)
        buses = torch.empty(self.world, dtype=torch.int32, device=self.device)
        dist.all_gather_into_tensor(buses, bus, group=process_group)
        self.buses = buses.tolist()
        if len(set(self.buses)) != self.world:
            raise RuntimeError(f"the SDMA transport needs one GPU per rank within a node; buses {self.buses}")
        # Peers in step order from this rank, so at each step every rank sends to a different peer.
        self.peers = [(self.rank + step) % self.world for step in range(1, self.world)]
        n = len(self.peers)
        peer_bus = (ctypes.c_int * n)(*[self.buses[p] for p in self.peers])
        self.engine_arr = (ctypes.c_int * n)()
        _check(lib.sx_engine_plan(self.device.index, peer_bus, n, 1, self.engine_arr), "sx_engine_plan")
        self.engines = list(self.engine_arr)
        self.one = torch.ones(1, dtype=torch.int32, device=self.device)
        self._buffers: Dict[str, SharedBuffer] = {}
        self._routes: Dict[tuple, "Plan"] = {}
        # Channel c's words at each rank: [c * world, (c + 1) * world), word p is rank p's.
        self._flags = self._buffer("signals", self.channels * self.world * 4, uncached=True)
        _start_watchdog(lib)
        logger.info(
            f"SDMA transport for group ranks {dist.get_process_group_ranks(process_group)}: engines {self.engine_plan()}"
        )

    def engine_plan(self) -> dict:
        """This rank's SDMA engines: {peer rank: engine}."""
        return dict(zip(self.peers, self.engines))

    def barrier(self) -> None:
        """Every rank's device work, and so every copy it waited for, is done past this point."""
        torch.cuda.synchronize()
        dist.barrier(group=self.process_group)
        torch.cuda.synchronize()

    def buffer(self, name: str, nbytes: int) -> SharedBuffer:
        """The shared buffer `name`, at least `nbytes`; collective when it must be (re)created."""
        return self._buffer(name, nbytes)

    def _buffer(self, name: str, nbytes: int, uncached: bool = False) -> SharedBuffer:
        buf = self._buffers.get(name)
        if buf is not None and buf.nbytes >= nbytes:
            return buf
        if buf is not None:
            self._release([buf])
        buf = SharedBuffer(self, nbytes, uncached)
        self._buffers[name] = buf
        self.barrier()
        return buf

    def route(self, key: Hashable, build, signal) -> "Plan":
        """The route `key` (interface.PeerTransport.route) as a registered plan."""
        cached = self._routes.get((key, signal))
        if cached is not None and all(buf.live for buf in cached.buffers):
            return cached
        if cached is not None:
            cached.free()
        copies = build()
        if len(copies) != len(self.peers):
            raise ValueError(f"a route needs the copies of each of the {len(self.peers)} peers")
        flags = None
        if signal is not None:
            if not 0 <= signal < self.channels:
                raise ValueError(f"channel {signal} is not in [0, {self.channels})")
            word = (signal * self.world + self.rank) * 4
            flags = [self._flags.peer_bases[i] + word for i in range(len(self.peers))]
        plan = Plan(
            self,
            [
                [(src, dst.peer_bases[i] + offset, nbytes) for src, dst, offset, nbytes in peer]
                for i, peer in enumerate(copies)
            ],
            flags,
        )
        plan.buffers = list({id(dst): dst for peer in copies for _, dst, _, _ in peer}.values())
        self._routes[(key, signal)] = plan
        return plan

    def begin(self) -> "Call":
        """Open a call (interface.PeerTransport.begin) on the next slot."""
        return Call(self)

    def _release(self, buffers) -> None:
        # Peers may write into a buffer until every rank is past its last use, and its owner may
        # free it only once every peer has unmapped it.
        self.barrier()
        for buf in buffers:
            buf.unmap()
        self.barrier()
        for buf in buffers:
            buf.free()

    def close(self) -> None:
        for plan in self._routes.values():
            plan.free()
        self._routes = {}
        self._release(list(self._buffers.values()))
        self._buffers = {}


def group_for(process_group: dist.ProcessGroup) -> PeerGroup:
    """The PeerGroup of `process_group`, created (collectively) on first use."""
    g = _by_process_group.get(process_group)
    if g is not None:
        return g
    key = tuple(dist.get_process_group_ranks(process_group))
    with _init_lock:
        if key not in _groups:
            _groups[key] = PeerGroup(process_group)
        _by_process_group[process_group] = _groups[key]
        return _groups[key]


def _timeout_message(count: int, tag: int, call: int, missing: int) -> str:
    words = [i for i in range(MAX_FLAGS_PER_WAIT) if missing >> i & 1]
    plans = {ranks: g.engine_plan() for ranks, g in _groups.items()}
    rank = dist.get_rank() if dist.is_initialized() else "?"
    return (
        f"rank {rank}: {count} SDMA flag wait(s) gave up after {wait_seconds():g} s. First: call {call}, wait "
        f"{tag}, flag words {words} never set (word r is group rank r's). Peer engines per group (peer "
        f"rank: engine): {plans}"
    )


def _start_watchdog(lib) -> None:
    """In abort mode, a thread that aborts the process once a flag wait has given up."""
    global _watchdog
    if _watchdog is not None or on_timeout() != "abort":
        return

    def watch():
        tag, call, missing = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint64()
        while True:
            count = lib.sx_peek_error(ctypes.byref(tag), ctypes.byref(call), ctypes.byref(missing))
            if count:
                message = _timeout_message(count, tag.value, call.value, missing.value) + "; aborting"
                logger.error(message)
                print(message, file=sys.stderr, flush=True)
                os.abort()
            time.sleep(0.1)

    _watchdog = threading.Thread(target=watch, name="xfuser-sdma-watchdog", daemon=True)
    _watchdog.start()


def raise_if_waits_gave_up() -> None:
    """Raise if a bounded flag wait of an earlier call on this process ran out."""
    tag, call, missing = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint64()
    count = load().sx_take_error(ctypes.byref(tag), ctypes.byref(call), ctypes.byref(missing))
    if count:
        raise RuntimeError(_timeout_message(count, tag.value, call.value, missing.value))


class Call:
    """One call on a slot (interface.Call): each send takes the next trigger, and each wait is
    tagged with its order in the call, for the timeout report."""

    def __init__(self, g: PeerGroup):
        global _slot, _call
        lib = load()
        if on_timeout() == "raise":
            raise_if_waits_gave_up()
        self._g = g
        self.slot, _slot = _slot, (_slot + 1) % _SLOTS
        _call += 1
        self.id = _call
        rc = lib.sx_arm(self.slot)
        if rc <= -2000:
            raise RuntimeError(f"flag copy {-rc - 2000} of an earlier call on slot {self.slot} never landed")
        if rc <= -1000:
            raise RuntimeError(f"data copy {-rc - 1000} of an earlier call on slot {self.slot} never landed")
        _check(rc, "sx_arm")
        self._sends = 0
        self._waits = 0

    def send(self, route: "Plan", src: torch.Tensor, stream: torch.cuda.Stream) -> None:
        if self._sends >= _TRIGGERS:
            raise ValueError(f"at most {_TRIGGERS} sends per call")
        k = self._sends
        lib = load()
        # Queued before the trigger is released, so the host issues them while the stream finishes.
        _check(lib.sx_plan_issue(self.slot, k, route.id, src.data_ptr(), self._g.one.data_ptr()), "sx_plan_issue")
        _check(lib.sx_release(self.slot, k, ctypes.c_void_p(stream.cuda_stream)), "sx_release")
        self._sends += 1

    def wait(self, channel: int, stream: torch.cuda.Stream) -> None:
        g = self._g
        if g.world > MAX_FLAGS_PER_WAIT:
            raise ValueError(f"at most {MAX_FLAGS_PER_WAIT} ranks per group; got {g.world}")
        _check(
            load().sx_wait_flags(
                ctypes.c_void_p(stream.cuda_stream),
                g._flags.ptr + channel * g.world * 4,
                g.world,
                g.rank,
                wait_seconds(),
                self._waits,
                self.id,
                int(on_timeout() == "abort"),
            ),
            "sx_wait_flags",
        )
        self._waits += 1

    def end(self) -> None:
        pass


class Plan:
    """A route's copies, registered once and issued per call with that call's source.

    `copies[i]` lists peer `g.peers[i]`'s copies as (source offset, destination address,
    bytes); `flags[i]` is the address of the flag word set behind them. With `flags=None` the
    plan writes no flags, for copies that a later plan's flag on the same engines covers.
    """

    def __init__(self, g: PeerGroup, copies, flags):
        lib = load()
        n = len(g.peers)
        if len(copies) != n or (flags is not None and len(flags) != n):
            raise ValueError(f"a plan needs copies and, unless flagless, a flag for each of the {n} peers")
        flat = [c for peer in copies for c in peer]
        u64 = ctypes.c_uint64 * max(1, len(flat))
        self.id = lib.sx_plan_create(
            n,
            g.engine_arr,
            (ctypes.c_int * n)(*[len(peer) for peer in copies]),
            u64(*[c[0] for c in flat]),
            u64(*[c[1] for c in flat]),
            u64(*[c[2] for c in flat]),
            None if flags is None else (ctypes.c_uint64 * n)(*flags),
        )
        if self.id < 0:
            raise RuntimeError(f"sx_plan_create failed with {self.id}")
        # The buffers the copies write into (set by PeerGroup.route).
        self.buffers: List[SharedBuffer] = []

    def free(self) -> None:
        if self.id >= 0:
            load().sx_plan_free(self.id)
            self.id = -1


def debug_drop_flag(ahead: int, issue: int, peer_index: int) -> None:
    """Test hook: in the `ahead`-th call started from now, the `issue`-th send (from 0) loses its
    flag write to peers[peer_index]."""
    load().sx_debug_drop_flag(ahead, issue, peer_index)
