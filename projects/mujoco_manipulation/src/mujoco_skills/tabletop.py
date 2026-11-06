"""An unmodified robosuite Stack scene with a Panda and two physical cubes."""

def make_tabletop(seed=0, *, video=False):
    import numpy as np
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from robosuite.utils.placement_samplers import UniformRandomSampler
    class SpacedCubes(UniformRandomSampler):
        def sample(self, *args, **kwargs):
            for _ in range(200):
                placements = super().sample(*args, **kwargs)
                positions = [value[0] for value in placements.values()]
                if np.linalg.norm(np.asarray(positions[0])[:2] - np.asarray(positions[1])[:2]) > .14:
                    return placements
            raise RuntimeError("Could not sample cubes with safe gripper clearance")
    np.random.seed(seed)
    return robosuite.make(
        "Stack", robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        initialization_noise=None, use_object_obs=True, use_camera_obs=False,
        placement_initializer=SpacedCubes(
            name="SeparatedCubes", x_range=[-.12, .12], y_range=[-.12, .12], rotation=0.,
            reference_pos=[0., 0., .8], z_offset=.01,
            ensure_object_boundary_in_range=False, ensure_valid_placement=True,
        ),
        has_renderer=False, has_offscreen_renderer=False,
        hard_reset=False, horizon=3000, control_freq=20,
    )
