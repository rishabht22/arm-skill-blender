"""Physical fixture counterexamples and isolated primitive regression checks."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import time

import mujoco as mj
import numpy as np

from .scene import LAYOUTS, SCAN, WORK
from .workcell import LOCK, OPEN, PHASES, PRIMITIVES, Workcell


def controls(cell):
    def evidence(name, blocked, **metrics):
        return {"case": name, "passed": bool(blocked), **metrics}

    result = []
    identity = (0, 1, 2, 3)
    cell.reset(31001, identity, 5)
    unlocked = cell.press()  # Exercise the panel independently of execute().
    result.append(evidence("index_open", not unlocked.ok and cell.index_count == 0 and cell.button_peak > .007,
                           angle=cell.j("rotor"), button_peak=cell.button_peak))
    cell.reset(31002, identity, 5)
    assert cell.execute("load", 0).ok and cell.execute("close", 0).ok
    unlocked = cell.press()
    result.append(evidence("index_unlatched", not unlocked.ok and cell.index_count == 0 and cell.button_peak > .007,
                           angle=cell.j("rotor"), button_peak=cell.button_peak))
    assert cell.execute("latch", 0).ok
    blocked = cell.move_slide("lid", OPEN)
    result.append(evidence("opening_locked_lid", blocked.reason == "contact:strike_bolt" and cell.j("lid") < .024,
                           lid_travel=cell.j("lid"), bolt_travel=cell.j("bolt")))
    cell.reset(31003, identity, 5)
    invalid = cell.transfer(0, "feed0", SCAN, malformed=True)
    position = cell.position(cell.carrier(0))
    target, _ = cell.slot(SCAN)
    result.append(evidence("reverse_key_insertion", not invalid.ok and not cell.seated(0, SCAN)
                           and abs(position[0] - target[0]) < .025,
                           seat_height_error=float(position[2] - target[2])))
    drops = []
    for quat in ((1, 0, 0, 0), (0, 0, 0, 1)):
        cell.reset(310031, identity, 5)
        target, _ = cell.slot(SCAN)
        joint = cell.data.joint(cell.carrier(0))
        joint.qpos[:] = [*(target + [0, 0, .024]), *quat]
        joint.qvel[:] = 0
        mj.mj_forward(cell.model, cell.data)
        cell.advance(80)
        drops.append((cell.touches(f"{cell.carrier(0)}_stem", f"{SCAN}_floor"),
                      float(cell.position(cell.carrier(0))[2] - target[2])))
    result.append(evidence("paired_key_clearance", drops[0][0] and not drops[1][0]
                           and drops[1][1] > .003, aligned_bottom=drops[0][0],
                           reversed_bottom=drops[1][0], reversed_height_error=drops[1][1]))
    cell.reset(31004, identity, 5)
    assert cell.move_slide("lid", 0).ok
    invalid = cell.transfer(0, "feed0", WORK)
    result.append(evidence("insertion_through_closed_shield", not invalid.ok and not cell.seated(0, WORK),
                           lid_travel=cell.j("lid")))
    cell.reset(31005, identity, 5)
    result.append(evidence("routing_without_read", not cell.execute("reject", 0).ok))
    assert cell.execute("read_load", 0).ok and cell.execute("read", 0).ok
    assert cell.execute("read_return", 0).ok
    result.append(evidence("misrouting_reject", not cell.execute("release", 0).ok,
                           measured_grade=cell.reads[0]))
    return result


def exercise(layout_id, repetitions):
    workcell = Workcell(layout_id)
    failures = []
    totals = Counter()
    for r in range(repetitions):
        for job in range(4):
            grade = (r + job + layout_id) % 2
            workcell.reset(30101 + 31 * r + job, (1, 3, 2, 0), grade << job, (job,))
            for kind in ("reseat", *PHASES[(r + job) % 2], *PHASES[(r + job + 1) % 2],
                         "reject" if grade else "release"):
                outcome = workcell.execute(kind, job)
                if not outcome.ok:
                    failures.append({"layout": layout_id, "repetition": r, "job": job,
                                     "primitive": kind, "reason": outcome.reason})
                    break
                totals[kind] += 1
    return totals, failures


def validate(layout_count=LAYOUTS, repetitions=2, workers=1):
    start = time.monotonic()
    counterexamples = controls(Workcell(0))
    totals = Counter()
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, errors in pool.map(exercise, range(layout_count), [repetitions] * layout_count):
            totals.update(count)
            failures.extend(errors)
    return {"successes_by_primitive": dict(totals), "failures": failures,
            "layout_count": layout_count, "repetitions": repetitions,
            "counterexamples": counterexamples, "seconds": time.monotonic() - start,
            "all_primitives_exercised": set(totals) == set(PRIMITIVES)}
