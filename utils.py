# This python script contains useful functions.

import torch
import numpy as np
import time


def interaction(env, policy, obs, scale_pol_output=True, device='cpu',
                **pol_kwargs):
    """Perform one interaction step.
    Performs one interaction step in the environment `env` with an action
    :math:`a_t \sim \pi(\cdot | o_t)` sampled from the policy :math:`\pi`
    given the current observation `o_t`.

    Args:
        env (Env): Environment.
        policy (torch.nn.module): Policy :math:`\pi`.
        obs (np.array or torch.tensor): Current observation :math`o_t`.
        scale_pol_output (bool): Scale policy output to (act_min, act_max)
        device (torch.device or None or str): 'cpu' or 'cuda' device.
        pol_kwargs: Policy's keyword arguments. E.g. deterministic or
          intention.

    Returns:
        dict: A dictionary with data from the interaction.
    """
    # Get action from policy
    with torch.no_grad():  # Because it is off-policy
        action, pol_info = policy(torch_ify(obs[None], device=device),
                                  **pol_kwargs)
    action = np_ify(action[0, :])

    if scale_pol_output:
        act_low = env.action_space.low
        act_high = env.action_space.high
        real_action = act_low + (action + 1.) * 0.5 * (act_high - act_low)
    else:
        real_action = action

    # Interact with the environment
    next_obs, reward, done, env_info = env.step(real_action)

    # Get multitask/multigoal data
    reward_vector = np.array(env_info.get('reward_multigoal', 0))
    termination_vector = \
        np.array(env_info.get('done_multigoal', False)).astype(np.float32)

    interaction_info = {
        'obs': obs,
        'next_obs': next_obs,
        'action': action,
        'reward': reward,
        'termination': float(done),
        'reward_vector': reward_vector,
        'termination_vector': termination_vector,
    }

    return interaction_info


def rollout(env, policy, max_horizon=100, fixed_horizon=True,
            render=False, return_info_dict=False,
            scale_pol_output=True,
            device='cpu',
            record_video_name=None,
            **pol_kwargs):
    """Perform an interaction over a finite time.
    Performs a rollout in the environment `env` following a policy :math:`\pi`
    over a finite horizon :math:`H` or until the environment is 'done'.

    Args:
        env (Env): Environment.
        policy (torch.nn.module): Policy :math:`\pi`.
        max_horizon (int): Maximum number of interactions.
        fixed_horizon (bool): Force the rollout to have a fixed horizon.
        render (bool): Rendering or not the environment.
        return_info_dict (bool): The function will return a dictionary with
            the resulted interaction data.
        device (torch.device or None or str): 'cpu' or 'cuda' device.
        record_video_name (str or None):
        pol_kwargs: Policy's keyword arguments. E.g. deterministic or
            intention.

    Returns:
        None or dict: If return_info_dict=True, it returns a dictionary with
                      data from the rollout.

    """
    rollout_info = {
        'obs': [],
        'next_obs': [],
        'action': [],
        'reward': [],
        'termination': [],
        'reward_vector': [],
        'termination_vector': [],
    }

    intention = pol_kwargs.get('intention', None)

    obs = env.reset()
    if render:
        env.render()
    if record_video_name is not None:
        env.start_recording_video(file_name=record_video_name)
        time.sleep(1.0)  # Wait a little
    for step in range(max_horizon):
        with torch.no_grad():
            interaction_info = interaction(env, policy, obs,
                                           scale_pol_output=scale_pol_output,
                                           device=device,
                                           **pol_kwargs)

        if render:
            env.render()

        if return_info_dict:
            for key in rollout_info.keys():
                rollout_info[key].append(interaction_info[key])

        obs = interaction_info['next_obs']
        is_done = interaction_info['termination'] if intention is None \
            else interaction_info['termination_vector'][intention]
        if not fixed_horizon and is_done:
            # print("The rollout has finished because the environment is done!")
            break

    if record_video_name is not None:
        env.stop_recording_video()

    if return_info_dict:
        return rollout_info


def np_ify(tensor):
    """Converts the given tensor to a numpy array object (np.ndarray)

    Args:
        tensor (torch.tensor): Tensor

    Returns:
        np.ndarray: Numpy array object

    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to('cpu').data.numpy()
    else:
        return np.array(tensor)


def torch_ify(ndarray, device=None, dtype=torch.float32):
    """Creates a torch.Tensor from a given tensor/array.

    Args:
        ndarray (array_like):
        device (torch.device, optional): Desired device of returned tensor.
            Default: if ``None``, uses default pytorch tensor type, usually
            'cpu'.
        dtype (torch.dtype, optional): Desired data type of returned tensor.
            Default: torch.float32

    Returns:
        torch.Tensor

    """
    if isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray).to(device=device, dtype=dtype)
    elif isinstance(ndarray, torch.Tensor):
        return ndarray
    else:
        return torch.as_tensor(ndarray, device=device, dtype=dtype)


def soft_param_update_from_to(source, target, tau):
    """Soft update of two torch Modules' parameters.

    Args:
        source (torch.nn.Module): Torch module from which to copy the
            parameters.
        target (torch.nn.Module): Torch module to copy the parameters.
        tau (float):

    Returns:
        None

    """
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )


def hard_buffer_update_from_to(source, target):
    """Hard update of two torch Modules' buffers.

    Args:
        source (torch.nn.Module): Torch module from which to copy the buffers.
        target (torch.nn.Module): Torch module to copy the buffers.

    Returns:
        None

    """
    # Buffers should be hard copy
    for target_buff, source_buff in zip(target.buffers(), source.buffers()):
        target_buff.data.copy_(source_buff.data)


def string_between(string, a, b):
    """Get a string between two substrings.

    Args:
        string (str): complete string.
        a (str): first substring.
        b (str): last substring.

    Returns:
        str: The resulting string.

    """
    return string.split(a)[1].split(b)[0]
