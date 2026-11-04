"""CALVIN-D evaluation poses, adapted from the upstream evaluation utility.

Source: mees/calvin, calvin_agent/evaluation/utils.py (MIT, Oier Mees 2021).
License: experiments/configs/CALVIN_LICENSE.txt. Uses an explicit random seed
instead of the upstream pyhash dependency; not the official 1000-sequence panel.
"""

import numpy as np


def initial_observations(state, *, seed=0):
    robot = np.array([
        .02586889, -.2313129, .5712808, 3.09045411, -.02908596,
        1.50013585, .07999963, -1.21779124, 1.03987629, 2.11978254,
        -2.34205014, -.87015899, 1.64119093, .55344928, 1.,
    ])
    rng = np.random.RandomState(seed)
    table = [np.array([.0500000896, -.120000177, .459990009]),
             np.array([.229995412, -.11999514, .459990010])]
    rng.shuffle(table)
    locations = {
        "slider_left": [-.240851662, .0924044687, .460990009],
        "slider_right": [.070341633, .0924044687, .460990009],
    }
    scene = np.zeros(24)
    scene[0] = .28 if state["slider"] == "left" else 0.
    scene[1] = .22 if state["drawer"] == "open" else 0.
    scene[3:6] = [.088 * state["lightbulb"], state["lightbulb"], state["led"]]
    for index, color in enumerate(("red", "blue", "pink")):
        location = state[f"{color}_block"]
        table_index = 0 if index == 0 else 1 if index == 2 or state["red_block"] == "table" else 0
        scene[6 + index * 6:9 + index * 6] = table[table_index] if location == "table" else locations[location]
        scene[11 + index * 6] = rng.uniform(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    return robot, scene
