"""Read outcome goals for interactive rollouts."""

import json
from pathlib import Path

from larger_scale.scene import CARRIERS, LAYOUTS

from .planner import CLOSED, FEED, LATCHED, OPEN, PROCESS, READER, RECOVERY, SORTED, Goal, State, Target, at
from .tasks import Task


CARRIERS_BY_NAME = {chr(65 + job): job for job in range(CARRIERS)}
DESTINATIONS = {"feed": FEED, "sorted": SORTED}
STARTS = {"feed": FEED, "process": PROCESS, "reader": READER, "recovery": RECOVERY}
ENCLOSURES = {"open": OPEN, "closed": CLOSED, "latched": LATCHED}


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer between {low} and {high}")
    return value


def read_goal(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if data.keys() - {"carriers", "read_before_index", "start", "layout", "seed", "grades"}:
        raise ValueError("unknown goal file field")
    targets = [Target()] * CARRIERS
    for name, spec in data["carriers"].items():
        if spec.keys() - {"cycles", "inspect", "destination"}:
            raise ValueError(f"unknown carrier field for {name}")
        cycles = integer(spec.get("cycles", 0), 0, 2, "cycles")
        inspect = spec.get("inspect", False)
        if type(inspect) is not bool:
            raise ValueError("inspect must be true or false")
        destination = DESTINATIONS[spec.get("destination", "feed")]
        if destination == SORTED and not inspect:
            raise ValueError("a sorted destination requires inspection")
        targets[CARRIERS_BY_NAME[name]] = Target(cycles, inspect, destination)
    rule = None
    if "read_before_index" in data:
        reader, process = (CARRIERS_BY_NAME[name] for name in data["read_before_index"])
        if reader == process or not targets[reader].inspect or not targets[process].cycles:
            raise ValueError("ordering requires two carriers with inspection and indexing goals")
        rule = (reader, process)
    state = State()
    if "start" in data:
        spec = data["start"]
        if spec.keys() - {"carrier", "location", "enclosure"}:
            raise ValueError("unknown start field")
        job = CARRIERS_BY_NAME[spec["carrier"]]
        where = STARTS[spec["location"]]
        enclosure = ENCLOSURES[spec.get("enclosure", "open")]
        if where != PROCESS and enclosure != OPEN:
            raise ValueError("a closed or latched start requires a loaded carrier")
        state = State(locations=at(state.locations, job, where), enclosure=enclosure)
    layout = integer(data.get("layout", 0), 0, LAYOUTS - 1, "layout")
    seed = integer(data.get("seed", 18011), 0, 2**31 - 1, "seed")
    grades = integer(data.get("grades", 0), 0, 2**CARRIERS - 1, "grades")
    ids = tuple(range(CARRIERS))
    return Task(path.stem, Goal(tuple(targets), rule), state, layout, seed, ids, ids, grades)
