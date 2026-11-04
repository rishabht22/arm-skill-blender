"""Goal graphs and deterministic train/test composition splits."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    name: str
    split: str
    sequence: tuple
    initial_state: dict
    final_checks: tuple

    def prerequisites(self, skill):
        # These experiments specify a chain, expressed to the head as a graph.
        index = self.sequence.index(skill)
        return set(self.sequence[:index])


def load_tasks(path):
    config = path if isinstance(path, dict) else json.loads(Path(path).read_text())
    active = config["active_skills"]
    if not active or len(set(active)) != len(active):
        raise ValueError("active_skills must be nonempty and unique")
    tasks = []
    for item in config["tasks"]:
        state = item["initial_state"]
        expected = {"slider", "drawer", "lightbulb", "led", "red_block", "blue_block", "pink_block"}
        if set(state) != expected or state["slider"] not in ("left", "right") or state["drawer"] not in ("open", "closed"):
            raise ValueError(f"Invalid initial state for {item['name']}")
        if any(state[key] not in (0, 1) for key in ("lightbulb", "led")):
            raise ValueError("Initial lights must be 0 or 1")
        locations = [state[color + "_block"] for color in ("red", "blue", "pink")]
        if any(location not in ("table", "slider_left", "slider_right") for location in locations):
            raise ValueError("Unsupported initial block location")
        if locations.count("table") > 2 or any(locations.count(side) > 1 for side in ("slider_left", "slider_right")):
            raise ValueError("Initial block placements overlap")
        sequence = tuple(item["sequence"])
        if not sequence or len(set(sequence)) != len(sequence):
            raise ValueError("Each task must contain unique skill events")
        if set(sequence) - set(active):
            raise ValueError(f"{item['name']} uses inactive skills")
        if not item["final_checks"] or set(item["final_checks"]) - set(sequence):
            raise ValueError("final_checks must refer to task skills")
        if item["split"] not in ("train", "test"):
            raise ValueError("split must be train or test")
        tasks.append(Task(item["name"], item["split"], sequence, item["initial_state"], tuple(item["final_checks"])))
    if len({t.name for t in tasks}) != len(tasks):
        raise ValueError("Task names must be unique")
    if not all(any(t.split == split for t in tasks) for split in ("train", "test")):
        raise ValueError("Both training and held-out test tasks are required")
    train_chains = {t.sequence for t in tasks if t.split == "train"}
    if any(t.sequence in train_chains for t in tasks if t.split == "test"):
        raise ValueError("Test compositions must differ from training compositions")
    return tuple(active), tasks
