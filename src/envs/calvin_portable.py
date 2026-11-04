"""CALVIN's native transforms with selectable EGL or software rendering."""

from pathlib import Path

import gym
import hydra
from omegaconf import OmegaConf
from calvin_agent.wrappers.calvin_env_wrapper import CalvinEnvWrapper
from calvin_env.envs.play_table_env import PlayTableSimEnv


class ManagedPlayTableSimEnv(PlayTableSimEnv):
    def close(self):
        # Upstream __del__ calls close() again after explicit shutdown.
        if getattr(self, "_composition_closed", False):
            return
        self._composition_closed = True
        if getattr(self, "ownsPhysicsClient", False) and getattr(self, "cid", -1) >= 0:
            super().close()
        self.cid = -1


class PortableCalvinEnv(CalvinEnvWrapper):
    def __init__(self, inputs, device, *, renderer="tiny", show_gui=False):
        cfg = OmegaConf.load(Path(inputs.abs_datasets_dir) / ".hydra/merged_config.yaml")
        cfg.env._target_ = "src.envs.calvin_portable.ManagedPlayTableSimEnv"
        required_cameras = {
            key.split("_", 1)[1]
            for key in list(inputs.observation_space.rgb_obs) + list(inputs.observation_space.depth_obs)
        }
        for name in list(cfg.cameras):
            if name not in required_cameras:
                del cfg.cameras[name]
        if renderer == "egl":
            self.set_egl_device(device)
        env = hydra.utils.instantiate(
            cfg.env, use_vr=False, use_scene_info=True,
            show_gui=show_gui, use_egl=renderer == "egl",
        )
        gym.Wrapper.__init__(self, env)
        self.observation_space_keys = inputs.observation_space
        self.transforms = inputs.transforms
        self.proprio_state = inputs.proprio_state
        self.device = device
        self.relative_actions = "rel_actions" in inputs.observation_space["actions"]
        self._has_reset = False

    def reset(self, *args, **kwargs):
        # Clear contacts, solver warm starts and controller state between episodes.
        # Skill boundaries never call this method.
        if self._has_reset:
            self.env.load()
            for camera in self.env.cameras:
                if hasattr(camera, "robot_uid"):
                    camera.robot_uid = self.env.robot.robot_uid
        self._has_reset = True
        return super().reset(*args, **kwargs)
