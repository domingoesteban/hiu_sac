import os
import numpy as np
import torch
import math
from models import MultiPolicyNet, MultiQNet, MultiVNet
from itertools import chain
import logger.logger as logger
import gtimer as gt
import tqdm

from utils import interaction, rollout, np_ify, torch_ify
from utils import soft_param_update_from_to, hard_buffer_update_from_to


class HIUSAC(object):
    def __init__(
            self,
            env,

            # Learning models
            nets_hidden_size=64,
            nets_nonlinear_op='relu',
            use_q2=True,
            explicit_vf=False,

            # RL algorithm behavior
            total_episodes=10,
            train_steps=100,
            eval_rollouts=10,
            max_horizon=100,
            fixed_horizon=True,

            # Target models update
            soft_target_tau=5e-3,
            target_update_interval=1,

            # Replay Buffer
            replay_buffer_size=1e6,
            batch_size=64,

            # Values
            discount=0.99,

            # Optimization
            optimization_steps=1,
            optimizer='adam',
            optimizer_kwargs=None,
            policy_lr=3e-4,
            qf_lr=3e-4,
            policy_weight_decay=1.e-5,
            q_weight_decay=1.e-5,

            # Entropy
            i_entropy_scale=1.,
            u_entropy_scale=None,
            auto_alpha=True,
            max_alpha=10,
            min_alpha=0.01,
            i_tgt_entro=None,
            u_tgt_entros=None,

            # Multitask
            multitask=True,
            combination_method='convex',

            # Others
            norm_input_pol=False,
            norm_input_vfs=False,
            seed=610,
            render=False,
            gpu_id=-1,

    ):
        """Hierarchical Intentional-Unitentional Soft Actor-Critic algorithm.
        Args:
            env (Env):  OpenAI-Gym-like environment with multigoal option.
            nets_hidden_size (int): Number of units in hidden layers for all
                the networks.
            use_q2 (bool): Use two parameterized Q-functions.
            explicit_vf (bool): Use a parameterized soft state value function.
            total_episodes (int): Number of episodes (iterations) to run the
                algorithm.
            train_steps (int): Number of training steps per episode.
            eval_rollouts (int): Number of rollouts to perform by the policy
                at the end of each episode.
            max_horizon (int): Maximum length of each rollout.
            fixed_horizon (bool):
            soft_target_tau (float): Interpolation factor for the target
                networks.
            target_update_interval (int): How often (gap between training
                steps) the target networks are updated. Training steps
            replay_buffer_size (int): Maximum length of the replay buffer.
            batch_size (int):  Minibatch size for SGD.
            discount (float): Discount factor (between 0 and 1).
            optimization_steps (int): Number of optimization steps after each
                interaction.
            optimizer (str): name of the (mini-batch) SGD optimizer.
                Options: 'adam' or 'rmsprop'.
            optimizer_kwargs (dict): Keyword arguments for the optimizer.
            policy_lr (float): Policy learning rate.
            qf_lr (float): State-action and state value functions learning rate
            policy_weight_decay (float): Weight decay (L2 penalty) in policy
                network.
            q_weight_decay (float): Weight decay (L2 penalty) in value
                networks.
            i_entropy_scale (float): Scale value for entropy in the compound
                task.
            u_entropy_scale (list or tuple of float): Scale value for the
                entropies in the composable tasks.
            auto_alpha (int): Compute entropy regularization term automatically
            max_alpha (float): Maximum entropy regularization value.
            min_alpha (float): Minimum entropy regularization value.
            i_tgt_entro (float): Target entropy value in the compound policy.
            u_tgt_entros (list or tuple of float): Target entropy value in the
                composable policy.
            multitask (bool): If False a single-task process is carried out,
                resulting in the SAC algorithm.
            combination_method (str): Combination method of the policy.
            norm_input_pol (bool): Normalize the input of the policy.
            norm_input_vfs (bool): Normalize the input of the value functions.
            seed (int): Seed value for the random number generators.
            render (bool): Rendering the interaction.
            gpu_id (int): GPU ID. For CPU use value -1
        """
        self.seed = seed
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        self.env = env
        self.env.seed(seed)

        if multitask:
            self.num_intentions = self.env.n_subgoals
        else:
            self.num_intentions = 0

        # Algorithm hyperparameters
        self.obs_dim = np.prod(env.observation_space.shape).item()
        self.action_dim = np.prod(env.action_space.shape).item()
        self.total_episodes = total_episodes
        self.train_steps = train_steps
        self.eval_rollouts = eval_rollouts
        self.max_horizon = max_horizon
        self.fixed_horizon = fixed_horizon
        self.render = render

        self.discount = discount

        self.soft_target_tau = soft_target_tau
        self.target_update_interval = target_update_interval

        self.norm_input_pol = norm_input_pol
        self.norm_input_vfs = norm_input_vfs

        # Policy Network
        self.policy = MultiPolicyNet(
            num_intentions=max(self.num_intentions, 1),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            shared_sizes=(nets_hidden_size,),
            intention_sizes=(nets_hidden_size, nets_hidden_size),
            shared_non_linear=nets_nonlinear_op,
            shared_batch_norm=False,
            intention_non_linear=nets_nonlinear_op,
            intention_final_non_linear='linear',
            intention_batch_norm=False,
            input_normalization=norm_input_pol,
            combination_method=combination_method,
        )

        # Value Function Networks
        self.qf1 = MultiQNet(
            num_intentions=self.num_intentions + 1,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            shared_sizes=(nets_hidden_size,),
            intention_sizes=(nets_hidden_size, nets_hidden_size),
            shared_non_linear=nets_nonlinear_op,
            shared_batch_norm=False,
            intention_non_linear=nets_nonlinear_op,
            intention_final_non_linear='linear',
            intention_batch_norm=False,
            input_normalization=norm_input_vfs,
        )
        if use_q2:
            self.qf2 = MultiQNet(
                num_intentions=self.num_intentions + 1,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                shared_sizes=(nets_hidden_size,),
                intention_sizes=(nets_hidden_size, nets_hidden_size),
                shared_non_linear='relu',
                shared_batch_norm=False,
                intention_non_linear='relu',
                intention_final_non_linear='linear',
                intention_batch_norm=False,
                input_normalization=norm_input_vfs,
            )
        else:
            self.qf2 = None

        if explicit_vf:
            self.vf = MultiVNet(
                num_intentions=self.num_intentions + 1,
                obs_dim=self.obs_dim,
                shared_sizes=(nets_hidden_size,),
                intention_sizes=(nets_hidden_size, nets_hidden_size),
                shared_non_linear='relu',
                shared_batch_norm=False,
                intention_non_linear='relu',
                intention_final_non_linear='linear',
                intention_batch_norm=False,
                input_normalization=norm_input_vfs,
            )
            self.target_vf = MultiVNet(
                num_intentions=self.num_intentions + 1,
                obs_dim=self.obs_dim,
                shared_sizes=(nets_hidden_size,),
                intention_sizes=(nets_hidden_size, nets_hidden_size),
                shared_non_linear='relu',
                shared_batch_norm=False,
                intention_non_linear='relu',
                intention_final_non_linear='linear',
                intention_batch_norm=False,
                input_normalization=norm_input_vfs,
            )
            self.target_vf.load_state_dict(self.vf.state_dict())
            self.target_vf.eval()
            self.target_qf1 = None
            self.target_qf2 = None
        else:
            self.vf = None
            self.target_vf = None
            self.target_qf1 = MultiQNet(
                num_intentions=self.num_intentions + 1,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                shared_sizes=(nets_hidden_size,),
                intention_sizes=(nets_hidden_size, nets_hidden_size),
                shared_non_linear='relu',
                shared_batch_norm=False,
                intention_non_linear='relu',
                intention_final_non_linear='linear',
                intention_batch_norm=False,
                input_normalization=norm_input_vfs,
            )
            self.target_qf1.load_state_dict(self.qf1.state_dict())
            self.target_qf1.eval()
            if use_q2:
                self.target_qf2 = MultiQNet(
                    num_intentions=self.num_intentions + 1,
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    shared_sizes=(nets_hidden_size,),
                    intention_sizes=(nets_hidden_size, nets_hidden_size),
                    shared_non_linear='relu',
                    shared_batch_norm=False,
                    intention_non_linear='relu',
                    intention_final_non_linear='linear',
                    intention_batch_norm=False,
                    input_normalization=norm_input_vfs,
                )
                self.target_qf2.load_state_dict(self.qf2.state_dict())
                self.target_qf2.eval()
            else:
                self.target_qf2 = None

        # Replay Buffer
        self.replay_buffer = MultiGoalReplayBuffer(
            max_size=int(replay_buffer_size),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_intentions=self.num_intentions,
        )
        self.batch_size = batch_size

        # Move models to GPU (if applicable)
        self.torch_device = \
            torch.device("cuda:" + str(gpu_id) if gpu_id >= 0 else "cpu")

        for model in self.trainable_models + self.non_trainable_models:
            model.to(device=self.torch_device)

        # Ensure non trainable models have fixed parameters
        for model in self.non_trainable_models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        # Entropy regularization coefficients (Alphas).
        if u_entropy_scale is None:
            u_entropy_scale = [i_entropy_scale
                               for _ in range(self.num_intentions)]
        self.entropy_scales = torch.tensor(u_entropy_scale+[i_entropy_scale],
                                           device=self.torch_device)
        if i_tgt_entro is None:
            i_tgt_entro = float(-self.action_dim)
        if u_tgt_entros is None:
            u_tgt_entros = [i_tgt_entro for _ in range(self.num_intentions)]
        self.tgt_entros = torch.tensor(u_tgt_entros + [i_tgt_entro],
                                       device=self.torch_device)
        self._auto_alpha = auto_alpha
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.log_alphas = torch.zeros(self.num_intentions+1,
                                      device=self.torch_device,
                                      requires_grad=True)

        # Select optimizer function and hyperparameters
        self.optimization_steps = optimization_steps
        if optimizer.lower() == 'adam':
            optimizer_class = torch.optim.Adam
            if optimizer_kwargs is None:
                optimizer_kwargs = dict(
                    amsgrad=True,
                    # amsgrad=False,
                )
        elif optimizer.lower() == 'rmsprop':
            optimizer_class = torch.optim.RMSprop
            if optimizer_kwargs is None:
                optimizer_kwargs = dict(

                )
        else:
            raise ValueError('Wrong optimizer')

        # Values optimizer
        qvals_params = self.qf1.parameters()
        if self.qf2 is not None:
            qvals_params = chain(qvals_params, self.qf2.parameters())
        self.qvalues_optimizer = optimizer_class(
            qvals_params,
            lr=qf_lr,
            weight_decay=q_weight_decay,
            **optimizer_kwargs
        )
        if self.vf is not None:
            self.vvalues_optimizer = optimizer_class(
                self.vf.parameters(),
                lr=qf_lr,
                weight_decay=q_weight_decay,
                **optimizer_kwargs
            )
        else:
            self.vvalues_optimizer = None

        # Policy optimizer
        self._policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            weight_decay=policy_weight_decay,
            **optimizer_kwargs
        )

        # Alpha optimizers
        self._alphas_optimizer = optimizer_class(
            [self.log_alphas],
            lr=policy_lr,
            **optimizer_kwargs
        )

        # Internal variables
        self.num_train_interactions = 0
        self.num_train_steps = 0
        self.num_eval_interactions = 0
        self.num_episodes = 0

        # Log variables
        self.logging_qvalues_error = 0
        self.logging_vvalues_error = 0
        self.logging_policies_error = 0
        self.logging_entros = torch.zeros((
            self.batch_size, self.num_intentions + 1
        ))
        self.logging_means = torch.zeros((
            self.batch_size, self.num_intentions + 1, self.action_dim
        ))
        self.logging_stds = torch.zeros((
            self.batch_size, self.num_intentions + 1, self.action_dim
        ))
        self.logging_weights = torch.zeros((
            self.batch_size, self.num_intentions, self.action_dim
        ))
        self.logging_eval_rewards = np.zeros((
            self.eval_rollouts, self.num_intentions + 1
        ))
        self.logging_eval_returns = np.zeros((
            self.eval_rollouts, self.num_intentions + 1
        ))

    @property
    def trainable_models(self):
        models = [
            self.policy,
            self.qf1
        ]
        if self.qf2 is not None:
            models.append(self.qf2)

        if self.vf is not None:
            models.append(self.vf)

        return models

    @property
    def non_trainable_models(self):
        models = [
            self.target_qf1
        ]
        if self.target_qf2 is not None:
            models.append(self.target_qf2)
        if self.target_vf is not None:
            models.append(self.target_vf)
        return models

    def train(self, init_episode=0):
        """Train the HIU policy with HIU algorithm.

        Args:
            init_episode (int): Initial iteration.

        Returns:
            np.ndarray: Array with the expected accumulated reward obtained
                with the HIU policy during the learning process.

        """

        if init_episode == 0:
            # Eval and log
            self.eval()
            self.log(write_table_header=True)

        gt.reset()
        gt.set_def_unique(False)

        expected_accum_rewards = np.zeros(self.total_episodes)

        episodes_iter = range(init_episode, self.total_episodes)
        if not logger.get_log_stdout():
            # Fancy iterable bar
            episodes_iter = tqdm.tqdm(episodes_iter)

        for iter in gt.timed_for(episodes_iter, save_itrs=True):
            # Put models in training mode
            for model in self.trainable_models:
                model.train()

            obs = self.env.reset()
            rollout_steps = 0
            for step in range(self.train_steps):
                if self.render:
                    self.env.render()
                interaction_info = interaction(
                    self.env, self.policy, obs,
                    device=self.torch_device,
                    intention=None, deterministic=False,
                )
                self.num_train_interactions += 1
                rollout_steps += 1
                gt.stamp('sample')

                # Add data to replay_buffer
                self.replay_buffer.add_interaction(**interaction_info)

                # Only train when there are enough samples from buffer
                if self.replay_buffer.available_samples() > self.batch_size:
                    for ii in range(self.optimization_steps):
                        self.learn()
                gt.stamp('train')

                # Reset environment if it is done
                if interaction_info['termination'] \
                        or rollout_steps > self.max_horizon:
                    obs = self.env.reset()
                    rollout_steps = 0
                else:
                    obs = interaction_info['next_obs']

            # Evaluate current policy to check performance
            expected_accum_rewards[iter] = self.eval()

            # Log the episode data
            self.log()

            self.num_episodes += 1

        return expected_accum_rewards

    def eval(self):
        """Evaluate deterministically the HIU policy.

        Returns:
            np.array: Expected accumulated reward

        """
        # Put models in evaluation mode
        for model in self.trainable_models:
            model.eval()

        env_subtask = self.env.get_active_subtask()
        for ii in range(-1, self.num_intentions):
            for rr in range(self.eval_rollouts):
                if self.num_intentions > 0:
                    self.env.set_active_subtask(None if ii == -1 else ii)

                rollout_info = rollout(self.env, self.policy,
                                       max_horizon=self.max_horizon,
                                       fixed_horizon=self.fixed_horizon,
                                       render=self.render,
                                       return_info_dict=True,
                                       device=self.torch_device,
                                       deterministic=True,
                                       intention=None if ii == -1 else ii
                                       )

                if ii == -1:
                    rewards = np.array(rollout_info['reward'])
                else:
                    rewards = np.array(rollout_info['reward_vector'])[:, ii]

                self.logging_eval_rewards[rr, ii] = rewards.mean()
                self.logging_eval_returns[rr, ii] = rewards.sum()

                self.num_eval_interactions += rewards.size

        # Set environment to training subtask.
        self.env.set_active_subtask(env_subtask)

        gt.stamp('eval')

        return self.logging_eval_returns[-1].mean().item()

    def learn(self):
        """Improve the HIU policy with HIU algorithm.

        The method computes a 'training step' of the algorithm.

        Returns:
            None

        """
        # Get batch from the replay buffer
        batch = self.replay_buffer.random_batch(self.batch_size,
                                                device=self.torch_device)
        # Get common data from batch
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # Concatenate all (sub)task rewards
        i_rewards = batch['rewards'].unsqueeze(-1)
        if self.num_intentions > 0:
            u_rewards = batch['reward_vectors'].unsqueeze(-1)
            hiu_rewards = torch.cat((u_rewards, i_rewards), dim=1)
        else:
            hiu_rewards = i_rewards

        # Concatenate all (sub)task terminations
        i_terminals = batch['terminations'].unsqueeze(-1)
        if self.num_intentions > 0:
            u_terminals = batch['termination_vectors'].unsqueeze(-1)
            hiu_terminations = torch.cat((u_terminals, i_terminals), dim=1)
        else:
            hiu_terminations = i_terminals

        policy_prior_log_probs = 0.0  # Uniform prior  # TODO: Normal prior

        # Alphas
        alphas = self.entropy_scales*self.log_alphas.exp()
        alphas.unsqueeze_(dim=-1)

        # Actions for batch observation
        i_new_actions, policy_info = self.policy(
            obs,
            deterministic=False,
            intention=None,
            log_prob=True,
        )
        i_new_log_pi = policy_info['i_log_prob']
        i_new_mean = policy_info['i_mean']
        i_new_std = policy_info['i_std']
        # Unintentional policy info
        if self.num_intentions > 0:
            u_new_actions = policy_info['u_actions']
            u_new_log_pi = policy_info['u_log_probs']
            u_new_means = policy_info['u_means']
            u_new_stds = policy_info['u_stds']
            activation_weights = policy_info['activation_weights']
        else:
            u_new_actions = None
            u_new_log_pi = None
            u_new_means = None
            activation_weights = None

        # Actions for batch next_observation
        with torch.no_grad():
            i_next_actions, policy_info = self.policy(
                next_obs,
                deterministic=False,
                intention=None,
                log_prob=True,
            )
        i_next_log_pi = policy_info['i_log_prob']
        # i_next_mean = policy_info['i_mean']
        # i_next_std = policy_info['i_std']
        # Unintentional policy info
        if self.num_intentions > 0:
            u_next_actions = policy_info['u_actions']
            u_next_log_pi = policy_info['u_log_probs']
            # u_next_means = policy_info['u_means']
            # u_next_stds = policy_info['u_stds']
        else:
            u_next_actions = None
            u_next_log_pi = None
            # u_next_means = None
            # u_next_stds = None

        # Intention Mask
        intention_mask = torch.eye(self.num_intentions + 1,
                                   device=self.torch_device,
                                   ).unsqueeze(-1)

        hiu_obs = obs.unsqueeze(-2).expand(
            self.batch_size, self.num_intentions + 1, self.obs_dim
        )
        hiu_actions = actions.unsqueeze(-2).expand(
            self.batch_size, self.num_intentions + 1, self.action_dim
        )
        hiu_next_obs = next_obs.unsqueeze(-2).expand(
            self.batch_size, self.num_intentions + 1, self.obs_dim
        )

        if u_new_actions is None:
            hiu_new_actions = i_new_actions.unsqueeze(-2)
        else:
            hiu_new_actions = torch.cat(
                (u_new_actions, i_new_actions.unsqueeze(-2)),
                dim=-2
            )

        if u_new_log_pi is None:
            hiu_new_log_pi = i_new_log_pi.unsqueeze(-2)
        else:
            hiu_new_log_pi = torch.cat(
                (u_new_log_pi, i_new_log_pi.unsqueeze(-2)),
                dim=-2
            )

        if u_next_actions is None:
            hiu_next_actions = i_next_actions.unsqueeze(-2)
        else:
            hiu_next_actions = torch.cat(
                (u_next_actions, i_next_actions.unsqueeze(-2)),
                dim=-2
            )

        if u_next_log_pi is None:
            hiu_next_log_pi = i_next_log_pi.unsqueeze(-2)
        else:
            hiu_next_log_pi = torch.cat(
                (u_next_log_pi, i_next_log_pi.unsqueeze(-2)),
                dim=-2
            )

        # ###################### #
        # Policy Evaluation Step #
        # ###################### #

        if self.target_vf is None:
            with torch.no_grad():
                # Estimate from target Q-value(s)
                # Q1_target(s', a')
                hiu_next_q1 = self.target_qf1(hiu_next_obs, hiu_next_actions)

                if self.target_qf2 is not None:
                    # Q2_target(s', a')
                    hiu_next_q2 = self.target_qf2(hiu_next_obs, hiu_next_actions)

                    # Minimum Unintentional Double-Q
                    hiu_next_q = torch.min(hiu_next_q1, hiu_next_q2)
                else:
                    hiu_next_q = hiu_next_q1

                # Get only the corresponding intentional values
                next_q_intention_mask = intention_mask.expand_as(hiu_next_q)
                hiu_next_q = torch.sum(hiu_next_q*next_q_intention_mask, dim=-2)

                # Vtarget(s')
                hiu_next_v = hiu_next_q - alphas*hiu_next_log_pi
        else:
            with torch.no_grad():
                # Vtarget(s')
                hiu_next_v = self.target_vf(hiu_next_obs)

                # Get only the corresponding intentional values
                next_v_intention_mask = intention_mask.expand_as(hiu_next_v)
                hiu_next_v = torch.sum(hiu_next_v*next_v_intention_mask, dim=-2)

        # Calculate Bellman Backup for Q-values
        hiu_q_backup = hiu_rewards + (1. - hiu_terminations) * self.discount * hiu_next_v

        # Predictions Q(s,a)
        hiu_q1_pred = self.qf1(obs, actions, intention=None)
        # Critic loss: Mean Squared Bellman Error (MSBE)
        hiu_qf1_loss = \
            0.5*torch.mean((hiu_q1_pred - hiu_q_backup)**2, dim=0).squeeze(-1)
        hiu_qf1_loss = torch.sum(hiu_qf1_loss)

        if self.qf2 is not None:
            hiu_q2_pred = self.qf2(obs, actions, intention=None)
            # Critic loss: Mean Squared Bellman Error (MSBE)
            hiu_qf2_loss = \
                0.5*torch.mean((hiu_q2_pred - hiu_q_backup)**2, dim=0).squeeze(-1)
            hiu_qf2_loss = torch.sum(hiu_qf2_loss)
        else:
            hiu_qf2_loss = 0

        self.qvalues_optimizer.zero_grad()
        qvalues_loss = (hiu_qf1_loss + hiu_qf2_loss)
        qvalues_loss.backward()
        self.qvalues_optimizer.step()

        # ############################## #
        # Policy Update/Improvement Step #
        # ############################## #

        # TODO: Decide if use the minimum btw q1 and q2. Using new_q1 for now
        hiu_new_q1 = self.qf1(hiu_obs, hiu_new_actions)
        hiu_new_q = hiu_new_q1

        next_q_intention_mask = intention_mask.expand_as(hiu_new_q)
        hiu_new_q = torch.sum(hiu_new_q*next_q_intention_mask, dim=-2)

        # Policy KL loss: - (E_a[Q(s, a) + H(.)])
        policy_kl_loss = -torch.mean(
            hiu_new_q - alphas*hiu_new_log_pi
            + policy_prior_log_probs,
            dim=0,
            )
        policy_loss = torch.sum(policy_kl_loss)

        # Update both Intentional and Unintentional Policies at the same time
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # ################################# #
        # (Optional) V-fcn improvement step #
        # ################################# #
        if self.vf is not None:
            hiu_v_pred = self.vf(hiu_obs)
            # Calculate Bellman Backup for Q-values
            hiu_v_backup = hiu_new_q - alphas*hiu_new_log_pi + policy_prior_log_probs
            hiu_v_backup.detach_()

            # Critic loss: Mean Squared Bellman Error (MSBE)
            hiu_vf_loss = \
                0.5*torch.mean((hiu_v_pred - hiu_v_backup)**2, dim=0).squeeze(-1)
            hiu_vf_loss = torch.sum(hiu_vf_loss)

        # ####################### #
        # Entropy Adjustment Step #
        # ####################### #
        if self._auto_alpha:
            # NOTE: In SAC formula is alphas and not log_alphas
            alphas_loss = - (self.log_alphas *
                             (hiu_new_log_pi.squeeze(-1) + self.tgt_entros
                              ).mean(dim=0).detach()
                             )
            hiu_alphas_loss = alphas_loss.sum()
            self._alphas_optimizer.zero_grad()
            hiu_alphas_loss.backward()
            self._alphas_optimizer.step()
            self.log_alphas.data.clamp_(min=math.log(self.min_alpha),
                                        max=math.log(self.max_alpha))

        # ########################### #
        # Target Networks Update Step #
        # ########################### #
        if self.num_train_steps % self.target_update_interval == 0:
            if self.target_vf is None:
                soft_param_update_from_to(
                    source=self.qf1,
                    target=self.target_qf1,
                    tau=self.soft_target_tau
                )
                if self.target_qf2 is not None:
                    soft_param_update_from_to(
                        source=self.qf2,
                        target=self.target_qf2,
                        tau=self.soft_target_tau
                    )
            else:
                soft_param_update_from_to(
                    source=self.vf,
                    target=self.target_vf,
                    tau=self.soft_target_tau
                )
        # Always hard_update of input normalizer (if active)
        if self.norm_input_vfs:
            if self.target_vf is None:
                hard_buffer_update_from_to(
                    source=self.qf1,
                    target=self.target_qf1,
                )
                if self.target_qf2 is not None:
                    hard_buffer_update_from_to(
                        source=self.qf2,
                        target=self.target_qf2,
                    )
            else:
                hard_buffer_update_from_to(
                    source=self.vf,
                    target=self.target_vf,
                )

        # Increase internal counter
        self.num_train_steps += 1

        # ######## #
        # Log data #
        # ######## #
        self.logging_policies_error = policy_loss.item()
        self.logging_qvalues_error = qvalues_loss.item()
        self.logging_vvalues_error = hiu_vf_loss.item() \
            if self.target_vf is not None else 0.
        self.logging_entros.data.copy_(-hiu_new_log_pi.squeeze(dim=-1).data)
        self.logging_means.data[:, -1].copy_(i_new_mean.data)
        self.logging_stds.data[:, -1].copy_(i_new_std.data)
        if self.num_intentions > 0:
            self.logging_means.data[:, :self.num_intentions].copy_(u_new_means.data)
            self.logging_stds.data[:, :self.num_intentions].copy_(u_new_stds.data)
            self.logging_weights.data.copy_(activation_weights.data)

    def save_training_state(self):
        """Save models

        Returns:
            None

        """
        models_dict = {
            'policy': self.policy,
            'qf1': self.qf1,
            'qf2': self.qf2,
            'target_qf1': self.target_qf1,
            'target_qf2': self.target_qf2,
            'vf': self.vf,
        }
        replaceable_models_dict = {
            'replay_buffer', self.replay_buffer,
        }
        logger.save_torch_models(self.num_episodes, models_dict,
                                 replaceable_models_dict)

    def load_training_state(self):
        pass

    def log(self, write_table_header=False):
        logger.log("Logging data in directory: %s" % logger.get_snapshot_dir())

        logger.record_tabular("Episode", self.num_episodes)

        logger.record_tabular("Accumulated Training Steps",
                              self.num_train_interactions)

        logger.record_tabular("Policy Error", self.logging_policies_error)
        logger.record_tabular("Q-Value Error", self.logging_qvalues_error)
        logger.record_tabular("V-Value Error", self.logging_vvalues_error)

        for intention in range(self.num_intentions):
            logger.record_tabular("Alpha [U-%02d]" % intention,
                                  np_ify(self.log_alphas[intention].exp()).item())
        logger.record_tabular("Alpha", np_ify(self.log_alphas[-1].exp()).item())

        for intention in range(self.num_intentions):
            logger.record_tabular(
                "Entropy [U-%02d]" % intention,
                np_ify(self.logging_entros[intention].mean(dim=0))
            )
        logger.record_tabular("Entropy",
                              np_ify(self.logging_entros[-1].mean(dim=0)))

        act_means = np_ify(self.logging_means.mean(dim=0))
        act_stds = np_ify(self.logging_stds.mean(dim=0))
        for aa in range(self.action_dim):
            for intention in range(self.num_intentions):
                logger.record_tabular(
                    "Mean Action %02d [U-%02d]" % (aa, intention),
                    act_means[intention, aa]
                )
                logger.record_tabular(
                    "Std Action %02d [U-%02d]" % (aa, intention),
                    act_stds[intention, aa]
                )
            logger.record_tabular("Mean Action %02d" % aa, act_means[-1, aa])
            logger.record_tabular("Std Action %02d" % aa, act_stds[-1, aa])

        for aa in range(self.action_dim):
            for intention in range(self.num_intentions):
                logger.record_tabular(
                    "Activation Weight Action %02d [U-%02d]" % (aa, intention),
                    np_ify(self.logging_weights.mean(dim=(0,))[intention, aa])
                )

        # Evaluation Stats to plot
        for ii in range(-1, self.num_intentions):
            if ii > -1:
                uu_str = ' [%02d]' % ii
            else:
                uu_str = ''
            logger.record_tabular(
                "Test Rewards Mean"+uu_str,
                np_ify(self.logging_eval_rewards[:, ii].mean())
            )

            logger.record_tabular(
                "Test Rewards Std"+uu_str,
                self.logging_eval_rewards[:, ii].std()
            )
            logger.record_tabular(
                "Test Returns Mean"+uu_str,
                self.logging_eval_returns[:, ii].mean()
            )
            logger.record_tabular(
                "Test Returns Std"+uu_str,
                self.logging_eval_returns[:, ii].std()
            )

        # Add the previous times to the logger
        times_itrs = gt.get_times().stamps.itrs
        train_time = times_itrs.get('train', [0])[-1]
        sample_time = times_itrs.get('sample', [0])[-1]
        eval_time = times_itrs.get('eval', [0])[-1]
        epoch_time = train_time + sample_time + eval_time
        total_time = gt.get_times().total
        logger.record_tabular('Train Time (s)', train_time)
        logger.record_tabular('(Previous) Eval Time (s)', eval_time)
        logger.record_tabular('Sample Time (s)', sample_time)
        logger.record_tabular('Epoch Time (s)', epoch_time)
        logger.record_tabular('Total Train Time (s)', total_time)

        # Dump the logger data
        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=write_table_header)
        # Save pytorch models
        self.save_training_state()
        logger.log("----")


class MultiGoalReplayBuffer(object):
    """Multigoal Replay Buffer

    """
    def __init__(self, max_size, obs_dim, action_dim, num_intentions):
        """

        Args:
            max_size (int): Maximum buffersize.
            obs_dim (int): Observation space dimension.
            action_dim (int): Action space dimension.
            num_intentions (int):
        """
        if not max_size > 1:
            raise ValueError("Invalid Maximum Replay Buffer Size: {}".format(
                max_size)
            )
        if not num_intentions >= 0:
            raise ValueError("Invalid Num Intentions Size: {}".format(
                num_intentions)
            )

        max_size = int(max_size)
        num_intentions = int(num_intentions)

        self.obs_buffer = torch.zeros((max_size, obs_dim))
        self.acts_buffer = torch.zeros((max_size, action_dim))
        self.rewards_buffer = torch.zeros((max_size, 1))
        self.termination_buffer = torch.zeros((max_size, 1))
        self.next_obs_buffer = torch.zeros((max_size, obs_dim))

        # Update reward vector and terminal vector buffers if applicable
        if num_intentions > 0:
            self.rew_vects_buffer = torch.zeros((max_size, num_intentions))
            self.term_vects_buffer = torch.zeros((max_size, num_intentions))
        else:
            self.rew_vects_buffer = None
            self.term_vects_buffer = None

        # self.to(device=device)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._max_size = max_size
        self._top = 0
        self._size = 0

    def add_interaction(self, obs, action, reward, termination, next_obs,
                        reward_vector=None, termination_vector=None):
        """Add a new sample to the buffer.

        Args:
            obs (np.ndarray or torch.Tensor): observation
            action (np.ndarray or torch.Tensor): action
            reward (np.ndarray or torch.Tensor): reward
            termination (np.ndarray or torch.Tensor): termination or 'done'
            next_obs (np.ndarray or torch.Tensor): next observation
            reward_vector (np.ndarray or torch.Tensor): reward multitask
            termination_vector (np.ndarray or torch.Tensor): termination multitask

        Returns:
            None

        """
        self.obs_buffer[self._top] = torch_ify(obs)
        self.acts_buffer[self._top] = torch_ify(action)
        self.rewards_buffer[self._top] = torch_ify(reward)
        self.termination_buffer[self._top] = torch_ify(termination)
        self.next_obs_buffer[self._top] = torch_ify(next_obs)

        # Update reward vector and terminal vector buffers if applicable
        if self.rew_vects_buffer is not None:
            self.rew_vects_buffer[self._top] = torch_ify(reward_vector)
        if self.term_vects_buffer is not None:
            self.term_vects_buffer[self._top] = torch_ify(termination_vector)

        self._advance()

    def _advance(self):
        self._top = (self._top + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def random_batch(self, batch_size, device=None):
        """Get a random batch

        Args:
            batch_size (int):
            device (torch.device):

        Returns:
            dict:

        """
        if batch_size > self._size:
            raise AttributeError('Not enough samples to get. %d bigger than '
                                 'current %d!' % (batch_size, self._size))

        indices = torch.randint(0, self._size, (batch_size,))
        batch_dict = {
            'observations': self.obs_buffer[indices].to(device),
            'actions': self.acts_buffer[indices].to(device),
            'rewards': self.rewards_buffer[indices].to(device),
            'terminations': self.termination_buffer[indices].to(device),
            'next_observations': self.next_obs_buffer[indices].to(device),
            'reward_vectors': self.rew_vects_buffer[indices].to(device)
            if self.rew_vects_buffer is not None else None,
            'termination_vectors': self.term_vects_buffer[indices].to(device)
            if self.term_vects_buffer is not None else None,
        }

        return batch_dict

    def available_samples(self):
        """Returns the current size of the buffer.

        Returns:
            int: Current size

        """
        return self._size

    @property
    def size(self):
        return self._size


if __name__ == '__main__':
    import tempfile
    from robolearn_envs.simple_envs import Navigation2dEnv
    from logger import setup_logger
    env = Navigation2dEnv()
    with tempfile.TemporaryDirectory() as log_dir:
        log_dir = setup_logger(
            log_dir=log_dir,
        )
        algo = HIUSAC(env)

        algo.train()
