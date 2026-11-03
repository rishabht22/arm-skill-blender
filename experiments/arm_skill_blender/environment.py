"""Create a two-arm scene using robosuite and MuJoCo."""


def make_env(*, seed=0):
    """Create two Panda arms facing each other across a table.

    The object is robosuite's pot with two handles. The preview only displays
    the initial scene; it does not advance physics or send robot commands.
    """
    import numpy as np
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    # In robosuite 1.5.1, TwoArmLift has no seed argument and its object
    # placement sampler uses NumPy's global random generator.
    np.random.seed(seed)

    # robosuite initializes controllers when creating the robots, even
    # though the static preview does not use them.
    controller = load_composite_controller_config(robot="Panda")

    return robosuite.make(
        "TwoArmLift",
        robots=["Panda", "Panda"],
        env_configuration="opposed",
        controller_configs=controller,
        initialization_noise=None,
        use_object_obs=False,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        reward_shaping=False,
        hard_reset=False,
    )
