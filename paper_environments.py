# This python module provides the environments used in the paper.


def get_env(env_name, subtask=None, seed=610, render=False,
            new_env_params=None):
    """Get Robolearn environment with some specific parameters.

    Args:
        env_name (str): Environment name
        subtask (int or None): Environment subtask index. None: Compound task
        seed (int): Seed value
        render (bool): Run environment with GUI.
        new_env_params (dict): Dictionary of new environment parameters.

    Returns:
        Env: Environment object
        dict: Environment parameters used in the environment object.

    """
    # Get environment function and default values
    env_fcn, env_params = get_env_and_params(env_name)

    # Update environment parameters.
    if new_env_params is not None:
        env_params.update(new_env_params)
    else:
        env_params['subtask'] = subtask
    env_params['seed'] = seed
    if not env_name.lower() == 'navigation2d':
        env_params['is_render'] = render

    return env_fcn(**env_params), env_params


def get_env_and_params(env_name):
    """Get environment function and default parameters.

    Args:
        env_name (str): Environment's name.
            Available options:
            - navigation2d
            - reacher
            - pusher
            - centauro

    Returns:
        environment function
        dict: Environment default parameters.

    """
    if env_name.lower() == 'navigation2d':
        from robolearn_envs.simple_envs import Navigation2dEnv
        env_params = dict(
            goal_reward=0,
            actuation_cost_coeff=5.0e+0,
            distance_cost_coeff=1.0e+0,
            log_distance_cost_coeff=2.0e+0,
            alpha=1e-1,
            # Initial Condition
            init_position=(4., 4.),
            init_sigma=0.10,
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
        from robolearn_envs.pybullet import Reacher2DGoalCompoEnv
        env_params = dict(
            is_render=False,
            # obs_distances=False,
            obs_distances=True,
            only_position=True,
            # only_position=False,
            obs_with_goal=True,
            # obs_with_goal=False,
            # goal_pose=(0.65, 0.65),
            goal_pose=(0.65, 0.35),
            rdn_goal_pos=True,
            # rdn_goal_pos=False,
            robot_config=None,
            rdn_robot_config=True,
            # goal_cost_weight=4.0e0,  # Antes de 27/02 9.21pm
            goal_cost_weight=3.0e0,
            ctrl_cost_weight=5.0e-1,
            goal_tolerance=0.01,
            use_log_distances=True,
            log_alpha=1e-6,
            # max_time=PATH_LENGTH*DT,
            max_time=None,
            sim_timestep=1.e-3,
            frame_skip=50,
            half_env=True,
            subtask=None,
            seed=610,
        )
        env_fcn = Reacher2DGoalCompoEnv
    elif env_name.lower() == 'pusher':
        from robolearn_envs.pybullet import Pusher2DGoalCompoEnv

        env_params = dict(
            is_render=False,
            # obs_distances=False,
            obs_distances=True,
            only_position=True,
            # only_position=False,
            goal_pose=(0.65, 0.65),
            rdn_goal_pose=True,
            tgt_pose=(0.5, 0.25, 1.4660),
            rdn_tgt_object_pose=True,
            robot_config=None,
            rdn_robot_config=True,
            tgt_cost_weight=0.5,
            # goal_cost_weight=1.5,
            goal_cost_weight=8,  # Desde 01-03 a las 4.30pm
            goal_tolerance=0.01,
            # ctrl_cost_weight=1.0e-5,  # Paper
            ctrl_cost_weight=1.0e-3,  # Desde 05-03
            # max_time=PATH_LENGTH*DT,
            max_time=None,
            sim_timestep=1.e-3,
            # frame_skip=10,  # Antes de 26-02
            frame_skip=50,
            half_env=True,
            subtask=None,
            seed=610,
        )
        env_fcn = Pusher2DGoalCompoEnv
    elif env_name.lower() == 'centauro':
        from robolearn_envs.pybullet import CentauroTrayEnv
        env_params = dict(
            is_render=False,
            # obs_distances=False,
            obs_distances=True,
            active_joints='RA',
            control_mode='joint_tasktorque',
            # _control_mode='torque',
            balance_cost_weight=0.3,
            fall_cost_weight=0.6,
            tgt_cost_weight=15.0,
            # tgt_cost_weight=50.0,
            balance_done_cost=0.,  # 2.0*PATH_LENGTH,  # TODO: dont forget same balance weight
            tgt_done_reward=0.,  # 20.0,
            ctrl_cost_weight=1.0e-2,
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
            random_tgt=True,
            random_config=False,
            # random_init=True,
            seed=610,
        )
        env_fcn = CentauroTrayEnv
    else:
        raise ValueError("Wrong environment name '%s'!" % env_name)

    return env_fcn, env_params
