import os
import numpy as np
import torch
import math
from networks import MultiPolicyNet, MultiQNet, MultiVNet
from itertools import chain
import logger.logger as logger
from collections import OrderedDict
import gtimer as gt

from utils import (
rollout,
np_ify,
torch_ify,
interaction
)

# MAX_LOG_ALPHA = 9.21034037  # Alpha=10000  Before 01/07
MAX_LOG_ALPHA = 6.2146080984  # Alpha=500  From 09/07


class HIUSAC:
    def __init__(
            self,
            env,

            # Models
            net_size=64,
            use_q2=True,
            explicit_vf=False,

            # RL algorithm behavior
            total_iterations=10,
            train_rollouts=10,
            eval_rollouts=10,
            max_horizon=50,
            fixed_horizon=True,
            render=False,
            gpu_id=-1,
            seed=610,

            # Target models update
            soft_target_tau=5e-3,
            target_update_interval=1,

            # Replay Buffer
            batch_size=64,
            replay_buffer_size=1e6,

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
            max_alpha=10,
            min_alpha=0.1,
            # auto_alpha=True,
            auto_alpha=False,
            i_tgt_entro=None,
            u_tgt_entros=None,

            log_dir=None,

    ):
        self.use_gpu = gpu_id >= 0
        global torch_device
        torch_device = torch.device("cuda:" + str(gpu_id) if self.use_gpu
                                    else "cpu")
        self.seed = seed
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        self.env = env
        self.env.seed(seed)

        self.num_intentions = self.env.n_subgoals

        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        self.total_iterations = total_iterations
        self.train_rollouts = train_rollouts
        self.eval_rollouts = eval_rollouts
        self.max_horizon = max_horizon
        self.fixed_horizon = fixed_horizon
        self.render = render

        self.discount = discount

        self.soft_target_tau = soft_target_tau
        self.target_update_interval = target_update_interval

        # Policy Network
        self.policy = MultiPolicyNet(
            num_intentions=self.num_intentions,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            shared_sizes=(net_size,),
            intention_sizes=(net_size, net_size),
            shared_non_linear='relu',
            # shared_non_linear='elu',
            shared_batch_norm=False,
            intention_non_linear='relu',
            # intention_non_linear='elu',
            intention_final_non_linear='linear',
            intention_batch_norm=False,
        )

        # Value Function Networks
        self.qf1 = MultiQNet(
            num_intentions=self.num_intentions + 1,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            shared_sizes=(net_size,),
            intention_sizes=(net_size, net_size),
            shared_non_linear='relu',
            shared_batch_norm=False,
            intention_non_linear='relu',
            intention_final_non_linear='linear',
            intention_batch_norm=False,
        )
        if use_q2:
            self.qf2 = MultiQNet(
                num_intentions=self.num_intentions + 1,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                shared_sizes=(net_size,),
                intention_sizes=(net_size, net_size),
                shared_non_linear='relu',
                shared_batch_norm=False,
                intention_non_linear='relu',
                intention_final_non_linear='linear',
                intention_batch_norm=False,
            )
        else:
            self.qf2 = None

        if explicit_vf:
            self.vf = MultiVNet(
                num_intentions=self.num_intentions + 1,
                obs_dim=self.obs_dim,
                shared_sizes=(net_size,),
                intention_sizes=(net_size, net_size),
                shared_non_linear='relu',
                shared_batch_norm=False,
                intention_non_linear='relu',
                intention_final_non_linear='linear',
                intention_batch_norm=False,
            )
            self.target_vf = MultiVNet(
                num_intentions=self.num_intentions + 1,
                obs_dim=self.obs_dim,
                shared_sizes=(net_size,),
                intention_sizes=(net_size, net_size),
                shared_non_linear='relu',
                shared_batch_norm=False,
                intention_non_linear='relu',
                intention_final_non_linear='linear',
                intention_batch_norm=False,
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
                shared_sizes=(net_size,),
                intention_sizes=(net_size, net_size),
                shared_non_linear='relu',
                shared_batch_norm=False,
                intention_non_linear='relu',
                intention_final_non_linear='linear',
                intention_batch_norm=False,
            )
            self.target_qf1.load_state_dict(self.qf1.state_dict())
            self.target_qf1.eval()
            if use_q2:
                self.target_qf2 = MultiQNet(
                    num_intentions=self.num_intentions + 1,
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    shared_sizes=(net_size,),
                    intention_sizes=(net_size, net_size),
                    shared_non_linear='relu',
                    shared_batch_norm=False,
                    intention_non_linear='relu',
                    intention_final_non_linear='linear',
                    intention_batch_norm=False,
                )
                self.target_qf2.load_state_dict(self.qf2.state_dict())
                self.target_qf2.eval()
            else:
                self.target_qf2 = None

        # Move models to GPU
        if self.use_gpu:
            self.policy.cuda(torch_device)
            self.qf1.cuda(torch_device)
            if self.qf2 is not None:
                self.qf2.cuda(torch_device)
            if self.vf is not None:
                self.vf.cuda(torch_device)
            if self.target_qf1 is not None:
                self.target_qf1.cuda(torch_device)
            if self.target_qf2 is not None:
                self.target_qf2.cuda(torch_device)
            if self.target_vf is not None:
                self.target_vf.cuda(torch_device)

        # Replay Buffer
        self.replay_buffer = MultiGoalReplayBuffer(
            max_size=int(replay_buffer_size),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_intentions=self.num_intentions,
        )
        self.batch_size = batch_size

        self.num_train_steps = 0
        self.num_train_interactions = 0
        self.num_eval_interactions = 0
        self.num_iters = 0

        # ###### #
        # Alphas #
        # ###### #
        if u_entropy_scale is None:
            u_entropy_scale = [i_entropy_scale
                               for _ in range(self.num_intentions)]
        self.entropy_scales = torch.tensor(u_entropy_scale+[i_entropy_scale],
                                           dtype=torch.float32,
                                           device=torch_device)
        if i_tgt_entro is None:
            i_tgt_entro = -self.env.action_dim
        if u_tgt_entros is None:
            u_tgt_entros = [i_tgt_entro for _ in range(self.num_intentions)]
        self.tgt_entros = torch.tensor(u_tgt_entros + [i_tgt_entro],
                                       dtype=torch.float32,
                                       device=torch_device)
        self._auto_alphas = auto_alpha
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.log_alphas = torch.zeros(self.num_intentions+1,
                                      device=torch_device,
                                      requires_grad=True)

        # ########## #
        # Optimizers #
        # ########## #
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
        vals_params_list = [self.qf1.parameters()]
        if self.qf2 is not None:
            vals_params_list.append(self.qf2.parameters())
        if self.vf is not None:
            vals_params_list.append(self.vf.parameters())
        vals_params = chain(*vals_params_list)
        self.values_optimizer = optimizer_class(
            vals_params,
            lr=qf_lr,
            weight_decay=q_weight_decay,
            **optimizer_kwargs
        )
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

        # Logs
        self.log_dir = logger.setup_logger(
            exp_prefix=self.env.name,
            seed=seed,
            log_dir=log_dir,
        )
        self.first_log = True
        self.log_values_error = 0
        self.log_policies_error = 0
        self.log_means = torch.zeros(self.num_intentions + 1, self.action_dim)
        self.log_stds = torch.zeros(self.num_intentions + 1, self.action_dim)
        self.log_weights = torch.zeros(self.num_intentions, self.action_dim)
        self.log_eval_rewards = np.zeros((
            self.eval_rollouts, self.num_intentions + 1, self.max_horizon
        ))

    def train(self, init_iteration=0):

        gt.reset()
        gt.set_def_unique(False)

        for iter in gt.timed_for(
                range(self.total_iterations),
                save_itrs=True,
        ):
            for rollout in range(self.train_rollouts):
                obs = self.env.reset()
                obs = torch_ify(obs, device=torch_device, dtype=torch.float32)
                for step in range(self.max_horizon):
                    interaction_info = interaction(
                        self.env, self.policy, obs,
                        render=self.render, device=torch_device,
                        intention=None, deterministic=False,
                    )
                    self.num_train_interactions += 1
                    gt.stamp('sample')

                    # Get useful info from interaction
                    next_obs = torch_ify(
                        interaction_info['next_obs'],
                        device=torch_device, dtype=torch.float32
                    )
                    action = interaction_info['action']
                    reward = torch_ify(
                        interaction_info['reward'],
                        device=torch_device, dtype=torch.float32
                    )
                    done = torch_ify(
                        interaction_info['done'],
                        device=torch_device, dtype=torch.float32
                    )
                    reward_vector = torch_ify(
                        interaction_info['reward_vector'],
                        device=torch_device, dtype=torch.float32
                    )
                    done_vector = torch_ify(
                        interaction_info['done_vector'],
                        device=torch_device, dtype=torch.float32
                    )

                    # Add data to replay_buffer
                    self.replay_buffer.add_sample(obs,
                                                  action.detach(),
                                                  reward,
                                                  done,
                                                  next_obs,
                                                  reward_vector,
                                                  done_vector
                                                  )

                    # Only train when there are enough samples from buffer
                    if self.replay_buffer.available_samples() > self.batch_size:
                        for ii in range(self.optimization_steps):
                            self.learn()
                    gt.stamp('train')

                    # Reset environment if it is done
                    if done:
                        obs = self.env.reset()
                        obs = torch_ify(obs, device=torch_device,
                                        dtype=torch.float32)
                    else:
                        obs = next_obs

            self.eval()

            self.log()

            self.num_iters += 1

    def eval(self):
        for rr in range(self.eval_rollouts):
            rollout_info = rollout(self.env, self.policy,
                                   max_horizon=self.max_horizon,
                                   fixed_horizon=self.fixed_horizon,
                                   render=self.render,
                                   return_info=True,
                                   device=torch_device,
                                   deterministic=True,
                                   intention=None)

            for step in range(len(rollout_info['reward'])):
                self.log_eval_rewards[rr, -1, step] = \
                    rollout_info['reward'][step]
                self.log_eval_rewards[rr, :self.num_intentions, step] = \
                    rollout_info['reward_vector'][step]
        gt.stamp('eval')

    def learn(self):
        # Get batch from the replay buffer
        batch = self.replay_buffer.random_batch(self.batch_size)

        # Get common data from batch
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # Concatenate all rewards
        i_reward = batch['rewards'].unsqueeze(-1)
        u_rewards = batch['reward_vectors'].unsqueeze(-1)
        hiu_rewards = torch.cat((u_rewards, i_reward), dim=1)

        # Concatenate all terminals
        i_terminals = batch['terminals'].unsqueeze(-1)
        u_terminals = batch['terminal_vectors'].unsqueeze(-1)
        hiu_terminals = torch.cat((u_terminals, i_terminals), dim=1)

        # One pass for both s and s' instead of two
        obs_combined = torch.cat((obs, next_obs), dim=0)
        i_all_actions, policy_info = self.policy(
            obs_combined,
            deterministic=False,
            intention=None,
            log_prob=True,
        )

        # Intentional policy info
        i_new_actions = i_all_actions[:self.batch_size]
        i_next_actions = i_all_actions[self.batch_size:].detach()
        i_new_log_pi = policy_info['log_prob'][:self.batch_size]
        i_next_log_pi = policy_info['log_prob'][self.batch_size:].detach()

        # Unintentional policy info
        u_new_actions = policy_info['action_vect'][:self.batch_size]
        u_next_actions = policy_info['action_vect'][self.batch_size:].detach()
        u_new_log_pi = policy_info['log_probs'][:self.batch_size]
        u_next_log_pi = policy_info['log_probs'][self.batch_size:].detach()

        # Additional info
        i_new_mean = policy_info['mean'][:self.batch_size]
        u_new_means = policy_info['means'][:self.batch_size]
        i_new_log_std = policy_info['log_std'][:self.batch_size]
        u_new_log_stds = policy_info['log_stds'][:self.batch_size]
        activation_weights = policy_info['activation_weights'][:self.batch_size]

        # Alphas
        # alphas = self.entropy_scales*torch.clamp(self.log_alphas,
        #                                          max=MAX_LOG_ALPHA).exp()
        alphas = self.entropy_scales*torch.clamp(self.log_alphas.exp(),
                                                 max=self.max_alpha)
        alphas.unsqueeze_(dim=-1)

        intention_mask = torch.eye(self.num_intentions + 1,
                                   dtype=torch.float32, device=torch_device
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

        # New actions (from current obs)
        hiu_new_actions = torch.cat(
            (u_new_actions, i_new_actions.unsqueeze(-2)),
            dim=-2
        )

        hiu_next_actions = torch.cat(
            (u_next_actions, i_next_actions.unsqueeze(-2)),
            dim=-2
        )

        hiu_next_log_pi = torch.cat(
            (u_next_log_pi, i_next_log_pi.unsqueeze(-2)),
            dim=-2
        )

        hiu_new_log_pi = torch.cat(
            (u_new_log_pi, i_new_log_pi.unsqueeze(-2)),
            dim=-2
        )

        # ####################### #
        # Policy Improvement Step #
        # ####################### #
        hiu_new_q1 = self.qf1(hiu_obs, hiu_new_actions)

        # TODO: Decide if use the minimum btw q1 and q2. Using new_q1 for now
        hiu_new_q = hiu_new_q1

        next_q_intention_mask = intention_mask.expand_as(hiu_new_q)
        hiu_new_q = torch.sum(hiu_new_q*next_q_intention_mask, dim=-2)

        policy_prior_log_probs = 0.0  # Uniform prior  # TODO: Normal prior

        # Policy KL loss: - (E_a[Q(s, a) + H(.)])
        policy_kl_loss = -torch.mean(
            hiu_new_q - alphas*hiu_new_log_pi
            + policy_prior_log_probs,
            dim=0,
        )
        policy_regu_loss = 0
        policy_loss = torch.sum(policy_kl_loss + policy_regu_loss)

        # Update both Intentional and Unintentional Policies at the same time
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # ###################### #
        # Policy Evaluation Step #
        # ###################### #
        if self.target_vf is None:
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
            # Vtarget(s')
            hiu_next_v = self.target_vf(hiu_next_obs)

            # Get only the corresponding intentional values
            next_v_intention_mask = intention_mask.expand_as(hiu_next_v)
            hiu_next_v = torch.sum(hiu_next_v*next_v_intention_mask, dim=-2)

        hiu_next_v.detach_()

        # Calculate Bellman Backup for Q-values
        hiu_q_backup = hiu_rewards + (1. - hiu_terminals) * self.discount * hiu_next_v
        # hiu_q_backup.detach_()

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

        if self.vf is not None:
            hiu_v_pred = self.vf(obs)
            # Calculate Bellman Backup for Q-values
            hiu_v_backup = hiu_new_q - alphas*hiu_new_log_pi + policy_prior_log_probs
            hiu_v_backup.detach_()

            # Critic loss: Mean Squared Bellman Error (MSBE)
            hiu_vf_loss = \
                0.5*torch.mean((hiu_v_pred - hiu_q_backup)**2, dim=0).squeeze(-1)
            hiu_vf_loss = torch.sum(hiu_vf_loss)
        else:
            hiu_vf_loss = 0

        self.values_optimizer.zero_grad()
        values_loss = (hiu_qf1_loss + hiu_qf2_loss + hiu_vf_loss)
        values_loss.backward()
        self.values_optimizer.step()

        # ####################### #
        # Entropy Adjustment Step #
        # ####################### #
        if self._auto_alphas:
            # NOTE: In formula is alphas and not log_alphas
            # log_alphas = self.log_alphas.clamp(max=MAX_LOG_ALPHA)
            alphas_loss = - (self.log_alphas *
                             (hiu_new_log_pi.squeeze(-1) + self.tgt_entros
                              ).detach()
                             ).mean()
            self._alphas_optimizer.zero_grad()
            alphas_loss.backward()
            self._alphas_optimizer.step()
            self.log_alphas.data.clamp_(min=math.log(self.max_alpha),
                                        max=math.log(self.max_alpha))

        # ########################### #
        # Target Networks Update Step #
        # ########################### #
        if self.num_train_steps % self.target_update_interval == 0:
            if self.target_vf is None:
                soft_update_from_to(
                    source=self.qf1,
                    target=self.target_qf1,
                    tau=self.soft_target_tau
                )
                if self.target_qf2 is not None:
                    soft_update_from_to(
                        source=self.qf2,
                        target=self.target_qf2,
                        tau=self.soft_target_tau
                    )
            else:
                soft_update_from_to(
                    source=self.vf,
                    target=self.target_vf,
                    tau=self.soft_target_tau
                )

        # Increase internal counter
        self.num_train_steps += 1

        # ######## #
        # Log data #
        # ######## #
        self.log_policies_error = policy_loss.item()
        self.log_values_error = values_loss.item()
        self.log_means[:self.num_intentions] = u_new_means.mean(dim=0)
        self.log_means[-1] = i_new_mean.mean(dim=0)
        self.log_stds[:self.num_intentions] = u_new_log_stds.exp().mean(dim=0)
        self.log_stds[-1] = i_new_log_std.exp().mean(dim=0)
        self.log_weights = activation_weights.mean(dim=0)

    def save(self):
        snapshot_gap = logger.get_snapshot_gap()
        snapshot_dir = logger.get_snapshot_dir()
        snapshot_mode = logger.get_snapshot_mode()

        save_full_path = os.path.join(
            self.log_dir,
            'models'
        )

        if snapshot_mode == 'all':
            models_dirs = list((
                os.path.join(
                    save_full_path,
                    str('itr_%03d' % self.num_iters)
                ),
            ))
        elif snapshot_mode == 'last':
            models_dirs = list((
                os.path.join(
                    save_full_path,
                    str('last_itr')
                ),
            ))
        elif snapshot_mode == 'gap':
            if self.num_iters % snapshot_gap == 0:
                models_dirs = list((
                    os.path.join(
                        save_full_path,
                        str('itr_%03d' % self.num_iters)
                    ),
                ))
            else:
                return
        elif snapshot_mode == 'gap_and_last':
            models_dirs = list((
                os.path.join(
                    save_full_path,
                    str('itr_%03d' % self.num_iters)
                ),
                os.path.join(
                    save_full_path,
                    str('last_itr')
                ),
            ))
        else:
            return

        for save_path in models_dirs:
            # logger.log('Saving models to %s' % save_full_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.policy, save_path + '/policy.pt')
            torch.save(self.qf1, save_path + '/qf1.pt')
            torch.save(self.qf2, save_path + '/qf2.pt')
            torch.save(self.target_qf1, save_path + '/target_qf1.pt')
            torch.save(self.target_qf2, save_path + '/target_qf2.pt')
            torch.save(self.vf, save_path + '/vf.pt')

        if not os.path.exists(save_full_path):
            os.makedirs(save_full_path)
        torch.save(self.replay_buffer, save_full_path + '/replay_buffer.pt')

    def load(self):
        pass

    def log(self):
        logger.log("Logging data in directory: %s" % self.log_dir)
        # Statistics dictionary
        statistics = OrderedDict()

        statistics["Iteration"] = self.num_iters
        statistics["Accumulated Training Steps"] = self.num_train_interactions

        # Training Stats to plot
        statistics["Total Policy Error"] = self.log_policies_error
        statistics["Total Value Error"] = self.log_values_error
        for intention in range(self.num_intentions):
            statistics["Alpha [U-%02d]" % intention] = \
                self.log_alphas[intention].exp().detach().cpu().numpy()
        statistics["Alpha"] = \
            self.log_alphas[-1].exp().detach().cpu().numpy()

        for aa in range(self.action_dim):
            for intention in range(self.num_intentions):
                statistics["Mean Action %02d [U-%02d]" % (aa, intention)] = \
                    self.log_means[intention, aa].detach().cpu().numpy()
                statistics["Std Action %02d [U-%02d]" % (aa, intention)] = \
                    self.log_stds[intention, aa].detach().cpu().numpy()
            statistics["Mean Action %02d" % aa] = \
                self.log_means[-1, aa].detach().cpu().numpy()
            statistics["Std Action %02d" % aa] = \
                self.log_stds[-1, aa].detach().cpu().numpy()

        for aa in range(self.action_dim):
            for intention in range(self.num_intentions):
                statistics["Activation Weight %02d [U-%02d]" % (aa, intention)] = \
                    self.log_weights[intention, aa].detach().cpu().numpy()

        # Evaluation Stats to plot
        statistics["Test Rewards Mean"] = \
            self.log_eval_rewards[:, -1, :].mean()
        statistics["Test Rewards Std"] = \
            self.log_eval_rewards[:, -1, :].std()
        statistics["Test Returns Mean"] = \
            np.sum(self.log_eval_rewards[:, -1, :], axis=-1).mean(axis=0)
        statistics["Test Returns Std"] = \
            np.sum(self.log_eval_rewards[:, -1, :], axis=-1).std(axis=0)

        # Add Tabular data to logger
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        # Add the previous times to the logger
        times_itrs = gt.get_times().stamps.itrs
        train_time = times_itrs['train'][-1]
        sample_time = times_itrs['sample'][-1]
        # eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
        eval_time = times_itrs['eval'][-1] if 'eval' in times_itrs else 0
        epoch_time = train_time + sample_time + eval_time
        total_time = gt.get_times().total
        logger.record_tabular('Train Time (s)', train_time)
        logger.record_tabular('(Previous) Eval Time (s)', eval_time)
        logger.record_tabular('Sample Time (s)', sample_time)
        logger.record_tabular('Epoch Time (s)', epoch_time)
        logger.record_tabular('Total Train Time (s)', total_time)

        # Dump the logger data
        if self.first_log:
            logger.dump_tabular(with_prefix=False, with_timestamp=False,
                                write_header=True)
            self.first_log = False
        else:
            logger.dump_tabular(with_prefix=False, with_timestamp=False,
                                write_header=False)
        # Save Pytorch models
        self.save()
        logger.log("----")


class MultiGoalReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim, num_intentions):
        if not max_size > 1:
            raise ValueError("Invalid Maximum Replay Buffer Size: {}".format(
                max_size)
            )
        if not num_intentions > 0:
            raise ValueError("Invalid Num Intentions Size: {}".format(
                num_intentions)
            )

        max_size = int(max_size)
        multi_size = int(num_intentions)

        self.obs_buffer = torch.zeros((max_size, obs_dim),
                                      dtype=torch.float32,
                                      device=torch_device)
        self.next_obs_buffer = torch.zeros((max_size, obs_dim),
                                           dtype=torch.float32,
                                           device=torch_device)
        self.acts_buffer = torch.zeros((max_size, action_dim),
                                       dtype=torch.float32,
                                       device=torch_device)
        self.rewards_buffer = torch.zeros((max_size, 1),
                                          dtype=torch.float32,
                                          device=torch_device)
        self.terminals_buffer = torch.zeros((max_size, 1),
                                            dtype=torch.float32,
                                            device=torch_device)
        self.rew_vects_buffer = torch.zeros((max_size, multi_size),
                                            dtype=torch.float32,
                                            device=torch_device)
        self.term_vects_buffer = torch.zeros((max_size, multi_size),
                                             dtype=torch.float32,
                                             device=torch_device)

        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._max_size = max_size
        self._top = 0
        self._size = 0

    def add_sample(self, obs, action, reward, terminal,
                   next_obs, rew_vector, term_vector):
        self.obs_buffer[self._top] = obs
        self.acts_buffer[self._top] = action
        self.rewards_buffer[self._top] = reward
        self.terminals_buffer[self._top] = terminal
        self.next_obs_buffer[self._top] = next_obs
        self.rew_vects_buffer[self._top] = rew_vector
        self.term_vects_buffer[self._top] = term_vector
        self._advance()

    def _advance(self):
        self._top = (self._top + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def random_batch(self, batch_size):
        if batch_size > self._size:
            raise AttributeError('Not enough samples to get. %d bigger than '
                                 'current %d!' % (batch_size, self._size))

        indices = torch.randint(0, self._size, (batch_size,), dtype=torch.long,
                                device=torch_device)
        return dict(
            observations=self.buffer_index(self.obs_buffer, indices),
            actions=self.buffer_index(self.acts_buffer, indices),
            rewards=self.buffer_index(self.rewards_buffer, indices),
            terminals=self.buffer_index(self.terminals_buffer, indices),
            next_observations=self.buffer_index(self.next_obs_buffer, indices),
            reward_vectors=self.buffer_index(self.rew_vects_buffer, indices),
            terminal_vectors=self.buffer_index(self.term_vects_buffer, indices),
        )

    def available_samples(self):
        return self._size

    @staticmethod
    def buffer_index(buffer, indices):
        return torch.index_select(buffer, dim=0, index=indices)


def soft_update_from_to(source, target, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )
