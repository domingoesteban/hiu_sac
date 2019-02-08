import numpy as np
from gym.spaces.box import Box


def get_env_and_params(env_name):
    if env_name.lower() == 'navigation2d':
        from robolearn_gym_envs.simple_envs import Navigation2dEnv
        env_params = dict(
            goal_reward=0,
            actuation_cost_coeff=5.0e+0,
            distance_cost_coeff=1.0e+0,
            log_distance_cost_coeff=2.0e+0,
            alpha=1e-1,
            # Initial Condition
            init_position=(4., 4.),
            init_sigma=1.00,
            # Goal
            goal_position=(-2.0, -2.0),  # TODO: Make this a script param
            goal_threshold=0.10,
            # Others
            dynamics_sigma=0.1,
            # dynamics_sigma=0.0,
            # horizon=PATH_LENGTH,
            horizon=None,
            subtask=None,
            seed=610,
        )
        env_fcn = Navigation2dEnv
    elif env_name.lower() == 'reacher':
        from robolearn_gym_envs.pybullet import Reacher2D3DofGoalCompoEnv
        env_params = dict(
            is_render=False,
            # obs_distances=False,
            obs_distances=True,
            obs_with_img=False,
            # obs_with_ori=True,
            obs_with_ori=False,
            obs_with_goal=True,
            # obs_with_goal=False,
            # goal_pose=(0.65, 0.65),
            goal_pose=(0.65, 0.35),
            # rdn_goal_pos=True,
            rdn_goal_pos=False,
            robot_config=None,
            rdn_robot_config=True,
            goal_cost_weight=4.0e0,
            ctrl_cost_weight=5.0e-1,
            goal_tolerance=0.01,
            use_log_distances=True,
            log_alpha=1e-6,
            # max_time=PATH_LENGTH*DT,
            max_time=None,
            sim_timestep=1.e-3,
            frame_skip=10,
            half_env=True,
            subtask=None,
            seed=610,
        )
        env_fcn = Reacher2D3DofGoalCompoEnv
    elif env_name.lower() == 'pusher':
        from robolearn_gym_envs.pybullet import Pusher2D3DofGoalCompoEnv

        env_params = dict(
            is_render=False,
            # obs_distances=False,
            obs_distances=True,
            obs_with_img=False,
            # obs_with_ori=True,
            obs_with_ori=False,
            goal_pose=(0.65, 0.65),
            rdn_goal_pose=True,
            tgt_pose=(0.5, 0.25, 1.4660),
            rdn_tgt_object_pose=True,
            robot_config=None,
            rdn_robot_config=True,
            tgt_cost_weight=5.0,
            goal_cost_weight=3.0,
            ctrl_cost_weight=1.0e-3,
            no_task_weight=1.0,
            goal_tolerance=0.01,
            # max_time=PATH_LENGTH*DT,
            max_time=None,
            sim_timestep=1.e-3,
            frame_skip=10,
            subtask=None,
            seed=610,
        )
        env_fcn = Pusher2D3DofGoalCompoEnv
    elif env_name.lower() == 'centauro':
        from robolearn_gym_envs.pybullet import CentauroTrayEnv
        env_params = dict(
            is_render=False,
            # obs_distances=False,
            obs_distances=True,
            obs_with_img=False,
            # obs_with_ori=True,
            active_joints='RA',
            control_mode='joint_tasktorque',
            # _control_mode='torque',
            balance_cost_weight=1.0,
            fall_cost_weight=1.0,
            tgt_cost_weight=3.0,
            # tgt_cost_weight=50.0,
            balance_done_cost=0.,  # 2.0*PATH_LENGTH,  # TODO: dont forget same balance weight
            tgt_done_reward=0.,  # 20.0,
            ctrl_cost_weight=1.0e-1,
            use_log_distances=True,
            log_alpha_pos=1e-4,
            log_alpha_ori=1e-4,
            goal_tolerance=0.05,
            min_obj_height=0.60,
            max_obj_height=1.20,
            max_obj_distance=0.20,
            max_time=None,
            sim_timestep=0.01,
            frame_skip=1,
            subtask=None,
            random_init=True,
            seed=610,
        )
        env_fcn = CentauroTrayEnv
    else:
        raise ValueError("Wrong environment name '%s'" % env_name)

    return env_fcn, env_params


def get_normalized_env(env_name, subtask=None, seed=610, render=False,
                       new_env_params=None):
    env_fcn, env_params = get_env_and_params(env_name)
    if new_env_params is not None:
        env_params.update(new_env_params)
    env_params['subtask'] = subtask
    env_params['seed'] = seed
    if not env_name.lower() == 'navigation2d':
        env_params['is_render'] = render

    return NormalizedEnv(env_fcn(**env_params)), env_params


class NormalizedEnv:
    def __init__(
            self,
            env,
    ):
        """
        Normalize action to in [-1, 1].
        :param env:
        """
        self._wrapped_env = env

        self._is_action_box = isinstance(self._wrapped_env.action_space, Box)

        # Action Space
        if isinstance(self._wrapped_env.action_space, Box):
            ub = np.ones(self._wrapped_env.action_space.shape)
            self.action_space = Box(-1 * ub, ub, dtype=np.float32)
        else:
            self.action_space = self._wrapped_env.action_space

    def reset(self, *args, **kwargs):
        obs = self._wrapped_env.reset(*args, **kwargs)

        return obs

    def step(self, action):
        if self._is_action_box:
            # Scale Action
            lb = self._wrapped_env.action_space.low
            ub = self._wrapped_env.action_space.high
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action

        # Interact with Environment
        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step

        return next_obs, reward, done, info

    def seed(self, *args, **kwargs):
        self._wrapped_env.seed(*args, **kwargs)

    def render(self, *args, **kwargs):
        self._wrapped_env.render(*args, **kwargs)

    @property
    def n_subgoals(self):
        return self._wrapped_env.n_subgoals

    @property
    def obs_dim(self):
        return self._wrapped_env.obs_dim

    @property
    def action_dim(self):
        return self._wrapped_env.action_dim

    @property
    def name(self):
        return type(self._wrapped_env).__name__

    def set_subtask(self, subtask):
        self._wrapped_env.set_subtask(subtask)
