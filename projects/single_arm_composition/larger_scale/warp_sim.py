"""Batch the existing Aster controllers over an isolated MuJoCo Warp backend."""

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from queue import Queue
from threading import Event

import mujoco as mj
import numpy as np

from .workcell import ANGLE, Workcell


@dataclass
class Step:
    remaining: int
    reset: bool
    done: Future


class WarpCell(Workcell):
    def __init__(self, layout_id, batch, slot):
        self.batch = None
        self.observed_qpos = None
        self.world_index = slot
        super().__init__(layout_id)
        self.batch = batch

    def reset(self, *args, **kwargs):
        self.reset_pending = True
        super().reset(*args, **kwargs)

    def advance(self, ticks, *, panel=False):
        self._locations = None
        if self.batch is None:
            return  # Workcell constructor. The first real reset is on Warp.
        if self.trace is not None:
            raise ValueError("Warp traces use separate physics provenance")
        if not panel:
            self._step(ticks)
            return
        for _ in range(ticks):
            self._step(1)
            depth = self.j("button")
            self.button_peak = max(depth, self.button_peak)
            pressed = depth > .007
            if pressed and not self._pressed and self.j("lid") < .007 and self.j("bolt") > .022:
                self._rotor_command = ANGLE if self._rotor_command < 1 else 0.
                self.data.ctrl[9] = self._rotor_command
                self.index_count += 1
            self._pressed = pressed

    def _step(self, ticks):
        done = Future()
        edited = self.observed_qpos is not None and not np.array_equal(self.data.qpos, self.observed_qpos)
        self.batch.requests.put((self.world_index, Step(ticks, self.reset_pending or edited, done)))
        self.reset_pending = False
        while not self.batch.failed.is_set():
            try:
                done.result(timeout=.2)
                return
            except TimeoutError:
                continue
        raise RuntimeError("Warp batch stopped")


class WarpBatch:
    def __init__(self, layout_id, size, device):
        import mujoco_warp as mjw
        import warp as wp
        from mujoco_warp._src.types import ContactType

        self.mjw, self.wp = mjw, wp
        self.device = wp.get_device(device)
        self.contact_mask = int(ContactType.CONSTRAINT | ContactType.PASSIVE)
        self.requests = Queue()
        self.failed = Event()
        self.cells = [WarpCell(layout_id, self, slot) for slot in range(size)]
        model = self.cells[0].model
        self.controls = np.zeros((size, model.nu), dtype=np.float32)
        self.active = np.zeros((size, model.neq), dtype=bool)
        self.equalities = np.stack([cell.model.eq_data for cell in self.cells]).astype(np.float32)
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(model, batch_sizes={"eq_data": size})
            self.data = mjw.make_data(model, nworld=size, nconmax=256, njmax=1024)
            self.graph = None

    def run(self, jobs):
        if len(jobs) != len(self.cells):
            raise ValueError("provide one controller call per world")

        def worker(cell, job):
            try:
                return job(cell)
            finally:
                self.requests.put((cell.world_index, None))

        with ThreadPoolExecutor(max_workers=len(jobs)) as workers:
            results = [workers.submit(worker, cell, job) for cell, job in zip(self.cells, jobs)]
            try:
                with self.wp.ScopedDevice(self.device):
                    self._schedule()
            finally:
                self.failed.set()
            return [result.result() for result in results]

    def _schedule(self):
        waiting = set(range(len(self.cells)))
        pending = {}
        while waiting or pending:
            resets = []
            while waiting:
                slot, request = self.requests.get()
                waiting.remove(slot)
                if request is None:
                    continue
                pending[slot] = request
                cell = self.cells[slot]
                self.controls[slot] = cell.data.ctrl
                self.active[slot] = cell.data.eq_active
                self.equalities[slot] = cell.model.eq_data
                if request.reset:
                    resets.append(slot)
            if not pending:
                break
            if resets:
                self._reset_worlds(resets)
            self.data.ctrl.assign(self.controls)
            self.data.eq_active.assign(self.active)
            self.model.eq_data.assign(self.equalities)
            if self.graph is None and self.device.is_cuda:
                with self.wp.ScopedCapture(device=self.device) as capture:
                    self.mjw.step(self.model, self.data)
                self.graph = capture.graph
            elapsed = min(request.remaining for request in pending.values())
            for _ in range(elapsed):
                if self.graph is None:
                    self.mjw.step(self.model, self.data)
                else:
                    self.wp.capture_launch(self.graph)
            waiting = {slot for slot, request in pending.items() if request.remaining == elapsed}
            self._observe(waiting)
            for slot in waiting:
                pending.pop(slot).done.set_result(None)
            for request in pending.values():
                request.remaining -= elapsed

    def _reset_worlds(self, slots):
        # Reset integrator state before setting the seeded host positions. All
        # worlds in a collection cohort normally reset together at the start.
        d = self.data
        for field in ("qpos", "qvel", "qacc_warmstart", "time"):
            values = getattr(d, field).numpy()
            for slot in slots:
                values[slot] = getattr(self.cells[slot].data, field)
            getattr(d, field).assign(values)

    def _observe(self, slots):
        d = self.data
        overflow = d.overflow.numpy()
        if any(overflow[slot] for slot in slots):
            raise RuntimeError(f"MuJoCo Warp buffer overflow: {overflow.tolist()}")
        # Transfer only the state the original controller uses. The optical
        # ray reads these Warp-computed geom transforms through the host API.
        fields = ("qpos", "qvel", "time", "xpos", "xmat", "geom_xpos",
                  "geom_xmat", "site_xpos")
        arrays = {field: getattr(d, field).numpy() for field in fields}
        count = int(d.nacon.numpy()[0])
        world = d.contact.worldid.numpy()[:count]
        contact_type = d.contact.type.numpy()[:count]
        geoms = d.contact.geom.numpy()[:count]
        for slot in slots:
            host = self.cells[slot].data
            for field, values in arrays.items():
                if field == "time":
                    host.time = float(values[slot])
                else:
                    out = getattr(host, field)
                    out[:] = values[slot].reshape(out.shape)
            if not np.isfinite(host.qpos).all() or not np.isfinite(host.qvel).all():
                raise RuntimeError(f"nonfinite MuJoCo Warp state in world {slot}")
            self.cells[slot].observed_qpos = host.qpos.copy()
            pairs = geoms[(world == slot) & ((contact_type & self.contact_mask) != 0)]
            if len(pairs) != host.ncon:
                mj._functions._realloc_con_efc(host, len(pairs), 0, 0)
            host.contact.geom[:len(pairs)] = pairs
