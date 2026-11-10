"""State-aware goal focus and call filters for the trained service selector."""

import numpy as np

from .environment import CONTACT_PART, KEEPER_OPEN, SENSORS
from .jobs import PORT, satisfied


HELD_MOTIONS = {
    "swing_keeper", "extract_sensor", "align_sensor", "seat_sensor", "align_cartridge",
    "feed_cartridge", "seat_cartridge_stop", "withdraw_cartridge", "orient_for_inspection", "seat_in_nest",
}
DESCENTS = {"descend_to_sensor", "descend_to_grasp"}
HANDOFF_HEIGHT = .92


class NoFeasibleCall(Exception):
    """No ranked call passed the current state checks."""


class GuardedSelector:
    """Focus on one component at a time while retaining the model's call ranking."""

    def __init__(self, selector, arm):
        self.selector = selector
        self.arm = arm
        self.active = None
        self.last_skill = None

    def priority(self, goal):
        obj = goal["object"]
        if obj not in SENSORS:
            return 2
        if goal["support"] != PORT and self.arm.contact(CONTACT_PART[obj], PORT):
            return 0
        return 1

    def focused_goals(self, goals):
        arm = self.arm
        if self.active is not None and satisfied(arm, self.active) and arm.position[2] >= HANDOFF_HEIGHT:
            self.active = None

        fixture = next((goal for goal in goals if "joint" in goal), None)
        if self.active is None:
            components = [goal for goal in goals if "xy" in goal]
            pending = [goal for goal in components if not satisfied(arm, goal)]
            if pending:
                self.active = min(pending, key=self.priority)
            elif fixture is not None and not satisfied(arm, fixture):
                self.active = next((goal for goal in components if goal.get("support") == PORT), None)
                if self.active is None:
                    self.active = next((goal for goal in components if goal["object"] in SENSORS), None)
            if self.active is None:
                return goals

        goal = self.active
        if goal["object"] not in SENSORS or (fixture is None and goal["support"] != PORT):
            return [goal]
        # A removal needs open access even when the final job calls for a locked port.
        target = fixture["joint"] if goal["support"] == PORT and fixture is not None else KEEPER_OPEN
        return [goal, {"object": "keeper", "joint": target, "tolerance": .10}]

    def applicable(self, skill, args, objects, held):
        arm = self.arm
        obj = args.get("obj") or args.get("held")
        if (obj is not None and obj not in objects) or (held is not None and obj != held):
            return False
        if skill in HELD_MOTIONS or args.get("held"):
            return held is not None
        if skill == "open_gripper":
            return held is not None
        if skill == "raise_by":
            return self.last_skill == "open_gripper"
        if held is not None or obj is None:
            return False

        delta = arm.location(obj) - arm.position
        horizontal = np.linalg.norm(delta[:2])
        if skill == "close_gripper":
            return np.linalg.norm(delta) <= .025
        if skill == "hover_over":
            return horizontal >= .012 or arm.position[2] <= .98
        if skill == "approach_keeper":
            height = args["height"]
            if height < .02 and horizontal > .027:
                return False
            return np.linalg.norm(delta + [0, 0, height]) >= .012
        if skill in DESCENTS:
            return horizontal <= .027 and np.linalg.norm(delta + [0, 0, .003]) >= .012
        return False

    def choose(self, observation, goals):
        done = all(satisfied(self.arm, goal) for goal in goals)
        if done and self.arm.position[2] >= HANDOFF_HEIGHT:
            return None
        local = self.focused_goals(goals)
        objects = {goal["object"] for goal in local}
        held = next((name for name in self.arm.env.entities if self.arm.holding(name)), None)
        for call in self.selector.ranked_calls(observation, local):
            if call is None:
                if done:
                    return None
                continue
            skill, args = call
            if self.applicable(skill, args, objects, held):
                self.last_skill = skill
                return call
        raise NoFeasibleCall
