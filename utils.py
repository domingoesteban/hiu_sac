import torch
import numpy as np
import time


def interaction(env, policy, obs, device='cpu', **pol_kwargs):
    # Get action from policy
    action, pol_info = policy(torch_ify(obs[None], dtype=torch.float32,
                                        device=device),
                              **pol_kwargs)

    # Interact with the environment
    next_obs, reward, done, env_info = env.step(np_ify(action[0, :]))

    reward_vector = np.array(env_info.get('reward_multigoal', None))
    done_vector = np.array(env_info.get('terminal_multigoal', None)).astype(np.float32)

    interaction_info = dict(
        next_obs=next_obs,
        action=action,
        reward=reward,
        done=float(done),
        reward_vector=reward_vector,
        done_vector=done_vector,
    )

    return interaction_info


def rollout(env, policy, max_horizon=100, fixed_horizon=False,
            render=False, return_info=False,
            device='cpu',
            **pol_kwargs):

    rollout_obs = list()
    rollout_action = list()
    rollout_next_obs = list()
    rollout_reward = list()
    rollout_done = list()
    rollout_reward_vector = list()
    rollout_done_vector = list()

    obs = env.reset()
    if render:
        env.render()
    for step in range(max_horizon):
        # start_time = time.time()
        interaction_info = interaction(
            env, policy, obs,
            device=device,
            **pol_kwargs
        )
        # elapsed_time = time.time() - start_time
        # print(elapsed_time)
        if render:
            env.render()

        if return_info:
            rollout_obs.append(obs)
            rollout_action.append(interaction_info['action'])
            rollout_next_obs.append(interaction_info['next_obs'])
            rollout_reward.append(interaction_info['reward'])
            rollout_done.append(interaction_info['done'])
            rollout_reward_vector.append(interaction_info['reward_vector'])
            rollout_done_vector.append(interaction_info['done_vector'])

        obs = interaction_info['next_obs']

        if not fixed_horizon and interaction_info['done']:
            print("The rollout has finished because the environment is done!")
            break
    # print("The rollout has finished because the maximum horizon is reached!")

    return dict(
        obs=rollout_obs,
        action=rollout_action,
        next_obs=rollout_next_obs,
        reward=rollout_reward,
        done=rollout_done,
        reward_vector=rollout_reward_vector,
        done_vector=rollout_done_vector,
    )


def np_ify(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.to('cpu').data.numpy()
    else:
        return np.array(tensor)


def torch_ify(ndarray, dtype=None, device=None):
    # return torch.from_numpy(ndarray).float().to(device).requires_grad_(requires_grad)
    if isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray).to(device=device, dtype=dtype)
    elif isinstance(ndarray, torch.Tensor):
        return ndarray
    else:
        return torch.as_tensor(ndarray, device=device, dtype=dtype)


def string_between(s, a, b):
    return s.split(a)[1].split(b)[0]
