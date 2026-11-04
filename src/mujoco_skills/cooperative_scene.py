"""Two independent Panda arms sharing one physical container and block."""

import numpy as np
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.models.objects import BoxObject, PotWithHandlesObject
from robosuite.models.tasks import ManipulationTask


class HoldAndPlaceScene(TwoArmLift):
    def _load_model(self):
        super()._load_model()
        # Reuse the native two-arm arena, robot models and controller interface.
        arena = self.model.mujoco_arena
        self.pot = PotWithHandlesObject(
            "container", body_half_size=(.065, .065, .04), handle_length=.055,
            handle_width=.07, handle_radius=.015, density=80, handle_friction=2.,
            use_texture=False, rgba_body=[.2, .4, .8, 1.],
        )
        self.block = BoxObject("payload", size=[.014, .014, .014], rgba=[.9, .15, .1, 1.], density=400)
        self.model = ManipulationTask(arena, [r.robot_model for r in self.robots], [self.pot, self.block])

    def _setup_references(self):
        super()._setup_references()
        self.block_body_id = self.sim.model.body_name2id(self.block.root_body)

    def _reset_internal(self):
        # The native sampler references the original pot; reset placements below.
        deterministic = self.deterministic_reset
        self.deterministic_reset = True
        try:
            super()._reset_internal()
        finally:
            self.deterministic_reset = deterministic


def make_cooperative_scene():
    from robosuite.controllers import load_composite_controller_config
    configs = [load_composite_controller_config(robot="Panda") for _ in range(2)]
    for config in configs:
        config["body_parts"]["right"]["input_ref_frame"] = "world"
    return HoldAndPlaceScene(
        robots=["Panda", "Panda"], env_configuration="opposed", controller_configs=configs,
        initialization_noise=None, use_object_obs=False, use_camera_obs=False,
        has_renderer=False, has_offscreen_renderer=False, hard_reset=False,
        table_full_size=(.8, 1.2, .05), control_freq=20, horizon=2500,
    )


def reset_scene(env, seed):
    env.reset()
    rng = np.random.default_rng(seed)
    pot_position = np.array([rng.uniform(-.015, .015), rng.uniform(-.015, .015), .845])
    block_position = np.array([rng.uniform(.09, .15), rng.uniform(.19, .25), .824])
    env.sim.data.set_joint_qpos(env.pot.joints[0], np.r_[pot_position, [0., 0., 0., 1.]])
    env.sim.data.set_joint_qpos(env.block.joints[0], np.r_[block_position, [1., 0., 0., 0.]])
    env.sim.forward()
    return rng
