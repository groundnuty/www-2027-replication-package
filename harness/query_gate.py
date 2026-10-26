"""One query at a time on a store we do not own, across processes (design log, design ruling of 20 Oct).

A private gate file holds the earliest time (epoch seconds) at which the next query may be sent, in the format the
private provenance scripts read and write ({"not_before": <float>}, here with the holder's "owner" token beside it),
so every client of the store shares one gate.

A query holds an exclusive lock on a sibling lock file for its WHOLE request (External review of c00244b, item 6): from
before the gate is checked until the whole answer has arrived, or, after a client time-out, a transport error, a cut
answer or a slow answer, until the server time-out has passed. A second process meanwhile finds the lock taken and is
refused without sending, however long the holder pauses. The gate file is still written, for the processes that do
not take the lock and for a holder that dies mid-query (the lock goes with the process, the reservation stays): on
acquisition, and again immediately before the send, it is set to that time plus the server time-out (and a margin),
so the reservation runs from the send. A gate in the future refuses without sending.

Releasing writes the gate only while it still holds this holder's own token, so a holder never clears a reservation
it does not own.

``acquire`` refuses at once (the scripts that send one query at a time and stop on a refusal). ``wait_acquire`` waits
its turn instead (the agent runner, whose attempts share the store): it polls the lock, then waits out a reservation in
the future, and gives up with ``GateTimeout`` once its bound (on a monotonic clock) has passed, having sent nothing.
Every waiter, a thread of this process or of another, polls the lock alike, at a short interval drawn afresh each time
(at most ``poll_s``): randomised contention with a bounded wait, rather than threads of one process handing the gate on
to each other while another process looks only now and then, which starved that process. The order is not first come,
first served: a query asked for just as the gate is let go can go before one that has been waiting (a session's calls
of one reply run back to back), and a process with more waiting threads has more chances, so a wait is bounded by the
bound, not by a queue position. The gate keeps, per process, when it
was queued for and when held (``GateStats``), so a runner can report the share of its wall time the gate was busy.

A gate of several ``slots`` lets that many queries run at once (a store's description sets it; harness/gated_store.py):
each slot is a gate of its own, a lock file and a reservation file, slot 0 the gate's own two files and slot ``i`` the
same names with ``.slot<i>`` before ``.lock``, so a process with fewer slots shares the first of them, and the queries
at the store across processes never exceed the largest slot count among them. A query takes the first slot it finds
free among the ``allowed`` ones, as a one-slot gate takes its one. ``step_down`` lowers ``allowed`` (never raises it):
new queries then take only the lower slots, and a query holding a higher one (its request, or its wait-out) finishes
as it would have. Every change of ``allowed`` starts a new ``epoch``, which each hold keeps, so a count of what the
queries met can leave out those sent before the change. A one-slot gate runs exactly as before slots existed.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import random
import secrets
import threading
import time
from collections.abc import Callable

GATE_MARGIN_S = 5.0
GATE_POLL_S = 0.05  # the longest pause between two looks at a taken gate


class GateClosed(RuntimeError):
    """The gate holds the next query back: another query may still be running on the store."""

    def __init__(self, not_before: float, now: float):
        super().__init__(f"the query gate is closed for another {max(0.0, not_before - now):.1f} s")
        self.not_before = not_before


class GateTimeout(GateClosed):
    """``wait_acquire`` waited its whole bound and the gate never opened; nothing was sent."""

    def __init__(self, waited_s: float, max_wait_s: float, not_before: float):
        RuntimeError.__init__(self, f"the query gate stayed closed for {waited_s:.1f} s (the wait is bounded at "
                                    f"{max_wait_s:g} s)")
        self.not_before = not_before
        self.waited_s = waited_s


class GateStats:
    """When this process queued for the gate and when it held it (gate clock), for the busy share of a run."""

    def __init__(self):
        self._lock = threading.Lock()
        self.queued: list[tuple[float, float]] = []
        self.held: list[tuple[float, float]] = []

    def add(self, kind: str, start: float, end: float) -> None:
        with self._lock:
            getattr(self, kind).append((start, end))

    def busy_shares(self, since: float, until: float) -> dict:
        """Seconds within [since, until] in which the gate was held by this process, queued for, or either (busy), and
        those as shares of the window."""
        with self._lock:
            queued, held = list(self.queued), list(self.held)
        wall = max(0.0, until - since)
        out = {"wall_s": round(wall, 3), "queries_held": len(held)}
        for name, spans in (("held", held), ("queued", queued), ("busy", held + queued)):
            seconds = _union_s(spans, since, until)
            out[f"{name}_s"] = round(seconds, 3)
            out[f"{name}_share"] = round(seconds / wall, 4) if wall > 0 else None
        return out


def _union_s(spans: list[tuple[float, float]], since: float, until: float) -> float:
    total, end = 0.0, since
    for a, b in sorted((max(a, since), min(b, until)) for a, b in spans):
        if b <= a or b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


class Hold:
    """One query's hold on the gate: the lock and the reservation, until ``release`` or ``wait_out``."""

    def __init__(self, gate: QueryGate, lock_fd: int, server_timeout_s: float, *, slot: int = 0, epoch: int = 0):
        self.gate = gate
        self.slot = slot
        self.epoch = epoch  # the gate's epoch when this hold was taken (``QueryGate.level``)
        self.owner = secrets.token_hex(16)
        self.server_timeout_s = server_timeout_s
        self._lock_fd: int | None = lock_fd
        self.acquired_at = gate.clock()
        self.release_at = 0.0
        self._release_mono = 0.0
        self.sending()

    def sending(self) -> float:
        """Reserve the store from now for the server time-out (and the margin): on acquisition, and again immediately
        before each send, so that the reservation runs from the send. The file holds wall time, for the other
        processes; this holder waits out its own on the monotonic clock, which a change of the wall clock cannot
        shorten."""
        span = self.server_timeout_s + self.gate.margin_s
        self._release_mono = self.gate.monotonic() + span
        self.release_at = self.gate.clock() + span
        self._write({"not_before": self.release_at, "owner": self.owner})
        return self.release_at

    def _read(self) -> dict:  # slot 0 is the gate's own file, read and written as before slots existed
        return self.gate._read(self.slot) if self.slot else self.gate._read()

    def _write(self, state: dict) -> None:
        if self.slot:
            self.gate._write(state, self.slot)
        else:
            self.gate._write(state)

    def release(self) -> None:
        """The whole answer arrived: the server has finished the query. The gate is opened only if it still holds
        this holder's reservation; the lock is let go either way."""
        try:
            if self._lock_fd is not None and self._read().get("owner") == self.owner:
                self._write({"not_before": self.gate.clock()})
        finally:
            self._unlock()

    def wait_out(self) -> None:
        """Sleep, still holding the lock, until the server time-out of the query sent under this hold has passed;
        then release. Interrupted (anything raised in the sleep, KeyboardInterrupt included), it lets the lock go but
        leaves the reservation, so no client sends before the server time-out has passed."""
        try:
            rest = self._release_mono - self.gate.monotonic()
            if rest > 0:
                self.gate.sleep(rest)
        except BaseException:
            self._unlock()
            raise
        self.release()

    def _unlock(self) -> None:
        if self._lock_fd is not None:
            fd, self._lock_fd = self._lock_fd, None
            try:
                os.close(fd)  # closing the descriptor drops its lock
            finally:
                self.gate.stats.add("held", self.acquired_at, self.gate.clock())


def slot_path(path: str, slot: int) -> str:
    """The reservation file of a gate's slot: the gate's own file for slot 0 (its lock file is this plus ".lock")."""
    return path if slot == 0 else f"{path}.slot{slot}"


class QueryGate:
    def __init__(self, path, clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 margin_s: float = GATE_MARGIN_S, jitter: Callable[[], float] = lambda: random.uniform(0.2, 1.0),
                 monotonic: Callable[[], float] | None = None, slots: int = 1, allowed: int | None = None):
        if not (isinstance(slots, int) and not isinstance(slots, bool) and slots >= 1):
            raise ValueError(f"a gate has a positive whole number of slots, got {slots!r}")
        allowed = slots if allowed is None else allowed
        if not (isinstance(allowed, int) and not isinstance(allowed, bool) and 1 <= allowed <= slots):
            raise ValueError(f"a gate of {slots} slot(s) allows 1 to {slots} of them, got {allowed!r}")
        self.path = os.path.expanduser(str(path))
        self.clock = clock
        # the wait's bound is measured on a monotonic clock (the given clock where a test gives one)
        self.monotonic = monotonic if monotonic is not None else (time.monotonic if clock is time.time else clock)
        self.sleep = sleep
        self.margin_s = margin_s
        self.jitter = jitter  # the share of ``poll_s`` of each pause between two looks at a taken lock
        self.stats = GateStats()
        self.slots = slots
        self._allowed = allowed
        self._epoch = 0
        self._level_lock = threading.Lock()

    def level(self) -> tuple[int, int]:
        """(the slots new queries may take now, the epoch): one reading, so the two belong together."""
        with self._level_lock:
            return self._allowed, self._epoch

    def step_down(self, to: int) -> tuple[int, int]:
        """Allow ``to`` slots from now (fewer than now, at least 1) and start a new epoch; returns (from, to). Queries
        holding a higher slot finish as they would have."""
        with self._level_lock:
            if not (isinstance(to, int) and not isinstance(to, bool) and 1 <= to < self._allowed):
                raise ValueError(f"a step down goes from {self._allowed} slot(s) to fewer, at least 1, not {to!r}")
            old, self._allowed = self._allowed, to
            self._epoch += 1
            return old, to

    def new_epoch(self) -> int:
        """Start a new epoch without changing the slots allowed (a trigger at the lowest level); returns it."""
        with self._level_lock:
            self._epoch += 1
            return self._epoch

    def _read(self, slot: int = 0) -> dict:
        try:
            with open(slot_path(self.path, slot)) as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {}

    def not_before(self, slot: int = 0) -> float:
        return float(self._read(slot).get("not_before", 0.0))

    def _write(self, state: dict, slot: int = 0) -> None:
        path = slot_path(self.path, slot)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    def _open_lock(self, slot: int) -> int:
        return os.open(slot_path(self.path, slot) + ".lock", os.O_WRONLY | os.O_CREAT, 0o600)

    def _lock_free_slot(self, fds: dict[int, int]) -> tuple[tuple[int, int] | None, list[float]]:
        """One look at the allowed slots, lowest first: (slot, the epoch) for the first whose lock is free and whose
        reservation has passed, its descriptor locked, else None; and, for each slot not taken, the time it is
        reserved to (now, for a taken lock). Every descriptor opened, the locked one too, stays in ``fds``, which the
        caller closes whatever happens: the locked one leaves it only once a hold has it (``_hold``)."""
        allowed, _ = self.level()
        closed = []
        for slot in range(allowed):
            if slot not in fds:
                fds[slot] = self._open_lock(slot)
            try:
                fcntl.flock(fds[slot], fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                closed.append(max(self.not_before(slot), self.clock()))
                continue
            now, held = self.clock(), self.not_before(slot)
            allowed_now, epoch = self.level()
            if now < held or slot >= allowed_now:  # a reservation left behind, or stepped down meanwhile
                fcntl.flock(fds[slot], fcntl.LOCK_UN)
                closed.append(max(held, now))
                continue
            return (slot, epoch), closed
        return None, closed

    def _hold(self, fds: dict[int, int], taken: tuple[int, int], server_timeout_s: float) -> Hold:
        """A hold on the slot ``taken`` names, whose locked descriptor is in ``fds``: it leaves ``fds`` only once the
        hold has it, so until then the caller's closing of ``fds`` lets the lock go on any exception."""
        slot, epoch = taken
        hold = Hold(self, fds[slot], server_timeout_s, slot=slot, epoch=epoch)
        del fds[slot]
        return hold

    @staticmethod
    def _close_all(fds: dict[int, int], hold: Hold | None = None) -> None:
        """Close every descriptor in ``fds`` (all of them, even when one fails). If that fails or is interrupted after
        ``hold`` was made, the hold never reaches its caller: its lock goes too (its reservation stays, as on any
        interruption), then the failure is raised."""
        try:
            first: BaseException | None = None
            for fd in list(fds.values()):
                try:
                    os.close(fd)
                except BaseException as exc:  # noqa: BLE001 - the rest are closed first, then it is raised
                    first = first or exc
            fds.clear()
            if first is not None:
                raise first
        except BaseException:
            if hold is not None:
                hold._unlock()
            raise

    def acquire(self, server_timeout_s: float) -> Hold:
        """Take the gate for one query: its lock (held until the hold is released or waited out) and a reservation.
        Refuse (``GateClosed``), changing nothing, when another process holds the lock or the gate is in the future.
        A gate of several slots takes the first free one, and refuses when none is."""
        if self.slots > 1:
            fds: dict[int, int] = {}
            hold = None
            try:
                taken, closed = self._lock_free_slot(fds)
                if taken is None:
                    now = self.clock()
                    raise GateClosed(min(closed, default=now), now)
                hold = self._hold(fds, taken, server_timeout_s)
                return hold
            finally:
                self._close_all(fds, hold)
        lock = os.open(self.path + ".lock", os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                now = self.clock()
                raise GateClosed(max(self.not_before(), now), now) from None
            now = self.clock()
            held = self.not_before()
            if now < held:
                raise GateClosed(held, now)
            return Hold(self, lock, server_timeout_s)
        except BaseException:
            os.close(lock)
            raise

    def wait_acquire(self, server_timeout_s: float, *, max_wait_s: float, poll_s: float = GATE_POLL_S) -> Hold:
        """Take the gate for one query as ``acquire`` does, waiting for it instead of refusing: for the lock (looked
        at again after a pause of at most ``poll_s``, drawn afresh each time), then for a reservation in the future.
        ``GateTimeout``, with nothing taken or changed, once ``max_wait_s`` has passed (or a reservation reaches past
        it)."""
        if not (isinstance(max_wait_s, (int, float)) and math.isfinite(max_wait_s) and max_wait_s >= 0):
            raise ValueError(f"the wait for the gate needs a finite bound, got {max_wait_s!r}")
        if self.slots > 1:
            return self._wait_acquire_slots(server_timeout_s, max_wait_s, poll_s)
        since = self.clock()
        start = self.monotonic()
        deadline = start + max_wait_s

        def left() -> float:
            return deadline - self.monotonic()

        def timed_out(not_before: float = 0.0) -> GateTimeout:
            self.stats.add("queued", since, self.clock())
            return GateTimeout(self.monotonic() - start, max_wait_s, max(not_before, self.not_before()))

        lock = os.open(self.path + ".lock", os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if left() <= 0:
                        raise timed_out() from None
                    self.sleep(min(poll_s * self.jitter(), left()))
            while True:  # a reservation by a holder that died, or by a process that does not take the lock
                now, held = self.clock(), self.not_before()
                if now >= held:
                    break
                if held - now > left():
                    raise timed_out(held)
                self.sleep(min(poll_s, held - now))
            if left() < 0:  # resumed past the bound (a late wake-up): give the gate up rather than send late
                raise timed_out()
            self.stats.add("queued", since, self.clock())
            return Hold(self, lock, server_timeout_s)
        except BaseException:
            os.close(lock)
            raise

    def _wait_acquire_slots(self, server_timeout_s: float, max_wait_s: float, poll_s: float) -> Hold:
        """``wait_acquire`` on a gate of several slots: look at the allowed slots, lowest first, until one is free and
        its reservation has passed, pausing at most ``poll_s`` (drawn afresh) between looks; ``GateTimeout``, with
        nothing taken or changed, once ``max_wait_s`` has passed."""
        since = self.clock()
        start = self.monotonic()
        deadline = start + max_wait_s
        fds: dict[int, int] = {}
        hold = None
        try:
            while True:
                taken, closed = self._lock_free_slot(fds)
                if taken is not None:  # its locked descriptor stays in ``fds`` until the hold has it
                    if deadline - self.monotonic() < 0:  # resumed past the bound: give it up rather than send
                        raise GateTimeout(self.monotonic() - start, max_wait_s, 0.0)
                    self.stats.add("queued", since, self.clock())
                    hold = self._hold(fds, taken, server_timeout_s)
                    return hold
                left = deadline - self.monotonic()
                if left <= 0:
                    raise GateTimeout(self.monotonic() - start, max_wait_s, min(closed, default=0.0))
                self.sleep(min(poll_s * self.jitter(), left))
        except GateTimeout:
            self.stats.add("queued", since, self.clock())
            raise
        finally:
            self._close_all(fds, hold)
