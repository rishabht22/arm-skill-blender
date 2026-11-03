"""Create a scene robot arms using robosuite and MuJoCo."""


def make_env(*, seed=0):
    """Create Sawyer (arm 1) and Panda (arm 2) facing each other across a table.

    The object is robosuite's pot with two handles. Call reset() to place
    the pot, then step() to move the robots and advance physics.
    """
    import numpy as np
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    from .sawyer import Sawyer

    # In robosuite 1.5.1, TwoArmLift has no seed argument and its object
    # placement sampler uses NumPy's global random generator.
    np.random.seed(seed)

    # Load the matching controller configuration for each robot.
    robots = ["UR5e", "Jaco"]
    controllers = [load_composite_controller_config(robot=robot) for robot in robots]

    return robosuite.make(
        "TwoArmLift",
        robots=robots,
        env_configuration="opposed",
        controller_configs=controllers,
        initialization_noise=None,
        use_object_obs=False,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        reward_shaping=False,
        hard_reset=False,
    )
