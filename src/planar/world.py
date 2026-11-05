"""A small collection task using the source simulator and frozen experts."""
import contextlib
import io
import numpy as np

from .experts import simulator_module

DISPATCH = ('push', 'lift', 'both')


class CollectionWorld:
    """A dispatch decision starts one or both native spatial macro actions.

    Decisions occur when all dispatched actions finish. Physics continues for
    both robots throughout. The native controller, collision model, maps,
    manipulation mechanics, and expert spatial targets are preserved.
    """
    def __init__(self, experts, seed=0, cubes=2, max_decisions=60, frame_callback=None):
        native = simulator_module()

        class DispatchEnv(native.VectorEnv):
            tracking_resets = False
            recovery_resets = 0

            def _set_awaiting_new_action(self):
                # Synchronous boundary, exposing each robot's native map.
                for robot in self.robots:
                    robot.awaiting_new_action = robot.is_idle()

            def _execute_actions(self):
                steps = 0
                while any(not robot.is_idle() for robot in self.robots):
                    if steps >= 4000:
                        raise RuntimeError('Native macro exceeded 4000 physics steps')
                    self.step_simulation()
                    for robot in self.robots:
                        if not robot.is_idle():
                            robot.step()
                    steps += 1
                return steps

            def reset_cube_pose(self, *args):
                if self.tracking_resets:
                    self.recovery_resets += 1
                return super().reset_cube_pose(*args)

            def step_simulation(self):
                super().step_simulation()
                if (self.tracking_resets and frame_callback is not None
                        and self.step_simulation_count % 24 == 0):
                    frame_callback(self)

        self.experts = experts
        self.max_decisions = max_decisions
        self.decisions = 0
        self.trace = []
        self.done = False
        with contextlib.redirect_stdout(io.StringIO()):
            self.env = DispatchEnv(
                robot_config=[{'pushing_robot': 1}, {'lifting_robot': 1}],
                num_cubes=cubes, env_name='small_empty', room_length=1., room_width=.5,
                random_seed=int(seed), use_egl_renderer=False, show_gui=False,
                use_intention_map=False, use_partial_observations=True,
            )
            self.state = self.env.reset()
        self.env.tracking_resets = True
        self.timestep = float(self.env.p.getPhysicsEngineParameters()['fixedTimeStep'])

    def features(self):
        e = self.env
        positions = np.array([e.get_cube_position(c)[:2] for c in e.cube_ids
                              if c not in e.removed_cube_ids_set])
        remaining = len(positions)
        mean = positions.mean(axis=0) if remaining else np.zeros(2)
        nearest = min((np.linalg.norm(p - np.array(e.receptacle_position[:2]))
                       for p in positions), default=0.)
        features = [remaining / e.num_cubes, *mean, nearest]
        for robot in e.robots:
            features.extend([*robot.get_position()[:2], np.cos(robot.get_heading()),
                             np.sin(robot.get_heading()), float(getattr(robot, 'cube_id', None) is not None)])
        features.extend([r.cumulative_cubes / e.num_cubes for r in e.robots])
        features.append(self.decisions / self.max_decisions)
        return np.asarray(features, dtype=np.float32)

    def potential(self):
        e = self.env
        return -sum(np.linalg.norm(np.array(e.get_cube_position(c)[:2]) - e.receptacle_position[:2])
                    for c in e.cube_ids if c not in e.removed_cube_ids_set)

    def step(self, dispatch):
        if self.done:
            raise RuntimeError('Episode already ended')
        if dispatch not in (0, 1, 2):
            raise ValueError('Dispatch must be push, lift, or both')
        e = self.env
        before = len(e.removed_cube_ids_set)
        before_steps = e.simulation_steps
        potential = self.potential()
        actions = [[None], [None]]
        for robot in (0, 1):
            if dispatch in (robot, 2):
                actions[robot][0] = self.experts.action(robot, self.state[robot][0])
        # Avoid carrying native per-action rewards from an inactive robot.
        for robot in e.robots:
            robot.cubes = 0
            robot.cubes_with_reward = 0
            robot.distance = 0
            robot.prev_waypoint_position = robot.get_position()
            robot.collided_with_obstacle = False
            robot.collided_with_robot = False
            if hasattr(robot, 'cube_dist_closer'):
                robot.cube_dist_closer = 0
            if hasattr(robot, 'pointless_cube_drop'):
                robot.pointless_cube_drop = False
        self.state, _, native_done, info = e.step(actions)
        self.decisions += 1
        self.done = bool(native_done or self.decisions >= self.max_decisions or e.recovery_resets)
        elapsed = (info['simulation_steps'] - before_steps) * self.timestep
        info['decision_seconds'] = elapsed
        delivered = len(e.removed_cube_ids_set) - before
        reward = 5. * delivered + .5 * (self.potential() - potential) - .02 * elapsed - .02
        if e.recovery_resets:
            reward -= 5.
        self.trace.append({'decision': self.decisions, 'dispatch': DISPATCH[dispatch],
                           'native_actions': actions, 'seconds': elapsed,
                           'delivered': delivered, 'reward': float(reward)})
        return self.features(), float(reward), self.done, info

    def report(self):
        e = self.env
        delivered = len(e.removed_cube_ids_set)
        return {'success': delivered == e.num_cubes and e.recovery_resets == 0,
                'delivered': delivered, 'cubes': e.num_cubes, 'decisions': self.decisions,
                'simulated_seconds': e.simulation_steps * self.timestep,
                'recovery_resets': e.recovery_resets,
                'obstacle_collisions': sum(r.cumulative_obstacle_collisions for r in e.robots),
                'robot_collisions': sum(r.cumulative_robot_collisions for r in e.robots),
                'trace': self.trace}

    def close(self):
        self.env.close()
