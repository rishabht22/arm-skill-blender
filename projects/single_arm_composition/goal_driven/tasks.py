"""Seeded outcome goals and separate benchmark suites."""

from dataclasses import dataclass
import random

from larger_scale.scene import CARRIERS, FAMILIAR_LAYOUTS

from .planner import CLOSED, FEED, LATCHED, OPEN, PROCESS, READER, RECOVERY, SORTED, Goal, State, Target, at


PROFILES = {
    "inspect": Target(inspect=True, destination=FEED),
    "process": Target(cycles=1, destination=FEED),
    "sort": Target(inspect=True, destination=SORTED),
    "process_sort": Target(cycles=1, inspect=True, destination=SORTED),
    "double": Target(cycles=2, destination=FEED),
}
BASE = tuple(PROFILES)
SUITES = ("novel_goal", "more_carriers", "partial", "ordering")
SEEDS = {"train": 52011, "dev": 74677, "engineering": 19697, "novel_goal": 88321,
         "more_carriers": 16747, "partial": 58943, "ordering": 42221}
HELD_PAIRS = (("process_sort", "process_sort"), ("double", "process_sort"),
              ("double", "sort"), ("inspect", "inspect"))


@dataclass(frozen=True)
class Task:
    name: str
    goal: Goal
    start: State
    layout: int
    seed: int
    binding: tuple[int, ...]
    destinations: tuple[int, ...]
    grades: int


def showcase(job=1, loaded=False):
    targets = at((Target(),) * CARRIERS, job, PROFILES["process"])
    start = State(locations=at(State().locations, job, PROCESS)) if loaded else State()
    return Task("showcase-loaded" if loaded else "showcase-feed", Goal(targets), start,
                0, 18011, tuple(range(CARRIERS)), tuple(range(CARRIERS)), 0)


def tasks(suite, count):
    rng = random.Random(SEEDS[suite])
    for i in range(count):
        size = 3 if suite == "more_carriers" else rng.choice((1, 2))
        if suite in ("dev", "novel_goal", "ordering"):
            size = 2
        jobs = rng.sample(range(CARRIERS), size)
        profiles = rng.choices(BASE, k=size)
        if suite == "novel_goal":
            profiles = list(rng.choice(HELD_PAIRS))
        elif suite == "dev":
            profiles = ["double", "double"]
        elif tuple(sorted(profiles)) in (*HELD_PAIRS, ("double", "double")):
            profiles = ["inspect", "process"]
        if suite == "partial" and not any(PROFILES[name].cycles for name in profiles):
            profiles[-1] = "process"
        rule = None
        if suite == "ordering":
            profiles = ["sort", rng.choice(("process", "double", "process_sort"))]
            rule = tuple(jobs)
        elif suite != "novel_goal" and size >= 2 and rng.random() < .30:
            possible = [(a, b) for a, pa in zip(jobs, profiles) for b, pb in zip(jobs, profiles)
                        if a != b and pa != "sort" and PROFILES[pa].inspect and PROFILES[pb].cycles]
            if possible:
                rule = rng.choice(possible)
        target = [Target()] * CARRIERS
        for job, name in zip(jobs, profiles):
            target[job] = PROFILES[name]
        locations = [FEED] * CARRIERS
        enclosure = OPEN
        if suite == "partial" or (suite not in ("ordering",) and rng.random() < .50):
            job = rng.choice(jobs)
            kinds = [RECOVERY]
            if target[job].inspect:
                kinds.append(READER)
            if target[job].cycles:
                kinds.append(PROCESS)
            where = rng.choice(kinds)
            if suite == "partial" and any(target[j].cycles for j in jobs):
                job = next(j for j in jobs if target[j].cycles)
                where = PROCESS
            locations[job] = where
            if where == PROCESS:
                enclosure = rng.choice((CLOSED, LATCHED)) if suite == "partial" else OPEN
        yield Task(f"{suite}-{i:05d}", Goal(tuple(target), rule),
                   State(locations=tuple(locations), enclosure=enclosure),
                   rng.randrange(FAMILIAR_LAYOUTS), rng.randrange(2**31),
                   tuple(rng.sample(range(CARRIERS), CARRIERS)),
                   tuple(rng.sample(range(CARRIERS), CARRIERS)), rng.randrange(16))
