import torch
import math

LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
LOG_SIG_MIN = -6.907755  # SIGMA 0.001
EPS = 1e-8


class Intention(torch.nn.Module):
    """
    Base NN class of an Intention head.
    """
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 non_linear,
                 final_non_linear,
                 batch_norm=False
                 ):
        super(Intention, self).__init__()
        self.batch_norm = batch_norm
        self.non_linear_name = non_linear
        self.output_non_linear_name = final_non_linear

        self.non_linear = get_non_linear_op(self.non_linear_name)
        self.output_non_linear = get_non_linear_op(self.output_non_linear_name)

        # Network
        self.layers = list()
        self.layer_norms = list()
        i_size = input_size
        for ll, o_size in enumerate(hidden_sizes):
            layer = torch.nn.Linear(i_size, o_size)
            self.layers.append(layer)
            self.__setattr__("layer{}".format(ll), layer)
            if self.batch_norm:
                bn = torch.nn.BatchNorm1d(o_size)
                self.layer_norms.append(bn)
                self.__setattr__("layer{}_norm".format(ll), bn)
            i_size = o_size

        self.olayer = torch.nn.Linear(i_size, output_size)

        # Initialize weights
        self.init_weights('uniform')

    def init_weights(self, init_fcn='uniform'):
        if init_fcn.lower() == 'uniform':
            init_fcn = torch.nn.init.xavier_uniform_
        else:
            init_fcn = torch.nn.init.xavier_normal_

        # Initialize hidden layers
        gain_name = self.non_linear_name
        if gain_name == 'elu':
            gain_name = 'relu'
        gain = torch.nn.init.calculate_gain(gain_name)
        for layer in self.layers:
            init_fcn(layer.weight.data, gain=gain)
            torch.nn.init.constant_(layer.bias.data, 0)

        # Initialize output layer
        gain_name = self.output_non_linear_name
        if gain_name == 'elu':
            gain_name = 'relu'
        gain = torch.nn.init.calculate_gain(gain_name)
        init_fcn(self.olayer.weight.data, gain=gain)
        torch.nn.init.constant_(self.olayer.bias.data, 0)

    def forward(self, x):
        for ll in range(len(self.layers)):
            x = self.non_linear(self.layers[ll](x))
            if self.batch_norm:
                x = self.layer_norms[ll](x)
        x = self.output_non_linear(self.olayer(x))
        return x


class MultiValueNet(torch.nn.Module):
    """
    Multi-head Value network
    """
    def __init__(self,
                 num_intentions,
                 input_dim,
                 shared_sizes,
                 intention_sizes,
                 shared_non_linear='relu',
                 shared_batch_norm=False,
                 intention_non_linear='relu',
                 intention_final_non_linear='linear',
                 intention_batch_norm=False,
                 ):
        super(MultiValueNet, self).__init__()

        self.shared_non_linear_name = shared_non_linear
        self.shared_batch_norm = shared_batch_norm

        self.shared_non_linear = get_non_linear_op(self.shared_non_linear_name)

        # Shared Layers
        self.shared_layers = list()
        self.shared_layer_norms = list()
        i_size = input_dim
        for ll, o_size in enumerate(shared_sizes):
            layer = torch.nn.Linear(i_size, o_size)
            self.shared_layers.append(layer)
            self.__setattr__("slayer{}".format(ll), layer)
            if self.shared_batch_norm:
                bn = torch.nn.BatchNorm1d(o_size)
                self.shared_layer_norms.append(bn)
                self.__setattr__("slayer{}_norm".format(ll), bn)
            i_size = o_size

        self.critic_nets = list()
        for ii in range(num_intentions):
            critic_net = Intention(
                input_size=i_size,
                hidden_sizes=intention_sizes,
                output_size=1,
                non_linear=intention_non_linear,
                final_non_linear=intention_final_non_linear,
                batch_norm=intention_batch_norm,
            )
            self.critic_nets.append(critic_net)
            self.add_module('intention{}'.format(ii), critic_net)

    def init_weights(self, init_fcn='uniform'):
        if init_fcn.lower() == 'uniform':
            init_fcn = torch.nn.init.xavier_uniform_
        else:
            init_fcn = torch.nn.init.xavier_normal_

        # Initialize shared layers
        gain_name = self.shared_non_linear_name
        if gain_name == 'elu':
            gain_name = 'relu'
        gain = torch.nn.init.calculate_gain(gain_name)
        for layer in self.shared_layers:
            init_fcn(layer.weight.data, gain=gain)
            torch.nn.init.constant_(layer.bias.data, 0)

    def forward(self, x, intention=None):
        for ll in range(len(self.shared_layers)):
            x = self.shared_non_linear(self.shared_layers[ll](x))
            if self.shared_batch_norm:
                x = self.shared_layer_norms[ll](x)

        if intention is None:
            critic_nets = self.critic_nets  # All critics
        else:
            critic_nets = [self.critic_nets[intention]]

        values = list()
        for critic in critic_nets:
            value = critic(x).unsqueeze(dim=-2)
            values.append(value)

        values_vect = torch.cat(values, dim=-2)

        return values_vect


class MultiQNet(MultiValueNet):
    def __init__(self,
                 num_intentions,
                 obs_dim,
                 action_dim,
                 shared_sizes,
                 intention_sizes,
                 shared_non_linear='relu',
                 shared_batch_norm=False,
                 intention_non_linear='relu',
                 intention_final_non_linear='linear',
                 intention_batch_norm=False,
                 ):
        self.input_dim = obs_dim + action_dim
        super(MultiQNet, self).__init__(
            num_intentions,
            self.input_dim,
            shared_sizes,
            intention_sizes,
            shared_non_linear=shared_non_linear,
            shared_batch_norm=shared_batch_norm,
            intention_non_linear=intention_non_linear,
            intention_final_non_linear=intention_final_non_linear,
            intention_batch_norm=intention_batch_norm,
        )

    def forward(self, observation, action, intention=None):
        x = torch.cat((observation, action), dim=-1)
        return super(MultiQNet, self).forward(x, intention=intention)


class MultiVNet(MultiValueNet):
    def __init__(self,
                 num_intentions,
                 obs_dim,
                 shared_sizes,
                 intention_sizes,
                 shared_non_linear='relu',
                 shared_batch_norm=False,
                 intention_non_linear='relu',
                 intention_final_non_linear='linear',
                 intention_batch_norm=False,
                 ):
        self.input_dim = obs_dim
        super(MultiVNet, self).__init__(
            num_intentions,
            self.input_dim,
            shared_sizes,
            intention_sizes,
            shared_non_linear=shared_non_linear,
            shared_batch_norm=shared_batch_norm,
            intention_non_linear=intention_non_linear,
            intention_final_non_linear=intention_final_non_linear,
            intention_batch_norm=intention_batch_norm,
        )

    def forward(self, observation, intention=None):
        return super(MultiVNet, self).forward(observation, intention=intention)


class MultiPolicyNet(torch.nn.Module):
    """
    Multi-head network
    """
    def __init__(self,
                 num_intentions,
                 obs_dim,
                 action_dim,
                 shared_sizes,
                 intention_sizes,
                 shared_non_linear='relu',
                 shared_batch_norm=False,
                 intention_non_linear='relu',
                 intention_final_non_linear='linear',
                 intention_batch_norm=False,
                 combination_method='convex',
                 ):
        super(MultiPolicyNet, self).__init__()

        self.action_dim = action_dim
        self.num_intentions = num_intentions
        self.shared_non_linear_name = shared_non_linear
        self.shared_batch_norm = shared_batch_norm

        self.shared_non_linear = get_non_linear_op(self.shared_non_linear_name)
        self.combination_method = combination_method

        # self.noise_loc = torch.nn.Parameter()
        self.register_buffer('noise_loc', torch.zeros(action_dim, dtype=torch.float32))
        self.register_buffer('noise_scale', torch.ones(action_dim, dtype=torch.float32))
        self.noise_dist = torch.distributions.Normal(
            loc=self.noise_loc,
            scale=self.noise_scale,
        )

        self._pols_idxs = torch.arange(self.num_intentions)

        # Shared Layers
        self.shared_layers = list()
        self.shared_layer_norms = list()
        i_size = obs_dim
        for ll, o_size in enumerate(shared_sizes):
            layer = torch.nn.Linear(i_size, o_size)
            self.shared_layers.append(layer)
            self.__setattr__("slayer{}".format(ll), layer)
            if self.shared_batch_norm:
                bn = torch.nn.BatchNorm1d(o_size)
                self.shared_layer_norms.append(bn)
                self.__setattr__("slayer{}_norm".format(ll), bn)
            i_size = o_size

        # Intentional Policies
        self.policy_nets = list()
        for ii in range(num_intentions):
            policy_net = Intention(
                input_size=i_size,
                hidden_sizes=intention_sizes,
                output_size=action_dim*2,  # Output means and log_stds
                non_linear=intention_non_linear,
                final_non_linear=intention_final_non_linear,
                batch_norm=intention_batch_norm,
            )
            self.policy_nets.append(policy_net)
            self.add_module('intention{}'.format(ii), policy_net)

        self.combination_net = Intention(
            input_size=i_size,
            hidden_sizes=intention_sizes,
            output_size=num_intentions*self.action_dim,  # Output means and log_stds
            non_linear=intention_non_linear,
            final_non_linear=intention_final_non_linear,
            batch_norm=intention_batch_norm,
        )

        if self.combination_method == 'convex':
            self.combination_non_linear = torch.nn.Softmax(dim=-2)
        else:
            self.combination_non_linear = get_non_linear_op('linear')

    def init_weights(self, init_fcn='uniform'):
        if init_fcn.lower() == 'uniform':
            init_fcn = torch.nn.init.xavier_uniform_
        else:
            init_fcn = torch.nn.init.xavier_normal_

        # Initialize shared layers
        gain_name = self.shared_non_linear_name
        if gain_name == 'elu':
            gain_name = 'relu'
        gain = torch.nn.init.calculate_gain(gain_name)
        for layer in self.shared_layers:
            init_fcn(layer.weight.data, gain=gain)
            torch.nn.init.constant_(layer.bias.data, 0)

    def forward(self, observation, deterministic=False, log_prob=False,
                intention=None):

        if log_prob and deterministic:
            raise ValueError("It is not possible to calculate log_probs in"
                             "deterministic policies")

        batch_size = observation.shape[0]
        x = observation
        for ll in range(len(self.shared_layers)):
            x = self.shared_non_linear(self.shared_layers[ll](x))
            if self.shared_batch_norm:
                x = self.shared_layer_norms[ll](x)

        means = list()
        log_stds = list()
        for policy in self.policy_nets:
            pol_params = policy(x).unsqueeze(dim=-2)
            means.append(pol_params[:, :, :self.action_dim])
            log_stds.append(pol_params[:, :, self.action_dim:])

        activation_weights = self.combination_net(x)
        activation_weights = activation_weights.reshape(-1,
                                                        self.num_intentions,
                                                        self.action_dim)
        activation_weights = self.combination_non_linear(activation_weights)

        means = torch.cat(means, dim=-2)

        real_log_prob = None
        log_probs = None

        # It is not efficient to calculate log_std, stds, and variances with
        # the deterministic option but it is easier to organize everything
        log_stds = torch.cat(log_stds, dim=-2)
        # # Method 1:
        # log_stds = torch.tanh(log_stds)
        # log_stds = \
        #     LOG_SIG_MIN + 0.5 * (LOG_SIG_MAX - LOG_SIG_MIN)*(log_stds + 1)
        # Methods 2:
        log_stds = torch.clamp(log_stds, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        stds = log_stds.exp()
        variances = stds**2

        if intention is None:
            # New Variance
            variance = torch.sum(
                variances.detach()*(activation_weights**2),
                dim=-2,
                keepdim=False,
                )

            std = torch.sqrt(variance)
            std = torch.clamp(std,
                              min=math.exp(LOG_SIG_MIN),
                              max=math.exp(LOG_SIG_MAX))
            log_std = torch.log(std)

            # New Mean
            mean = torch.sum(
                means.detach()*activation_weights,
                dim=-2,
                keepdim=False,
                )
        else:
            intention = self._pols_idxs[intention]
            mean = torch.index_select(means, dim=-2, index=intention).squeeze(-2)

            std = \
                torch.index_select(stds, dim=1, index=intention).squeeze(-2)
            log_std = \
                torch.index_select(log_stds, dim=1, index=intention).squeeze(-2)
            variance = \
                torch.index_select(variances, dim=1, index=intention).squeeze(-2)

        if deterministic:
            action = mean
            actions_vect = means
        else:
            # Sample from Gaussian distribution
            # noise = self.noise_dist.sample((batch_size,))
            # actions_vect = stds*noise.unsqueeze(1) + means
            # action = std*noise + mean
            # actions_vect = stds*torch.randn_like(stds) + means
            # action = std*torch.randn_like(std) + mean
            noise = torch.randn_like(std)
            actions_vect = stds*noise.unsqueeze(1) + means
            action = std*noise + mean

            if log_prob:
                log_probs = -0.5*(((actions_vect - means) / (stds + EPS))**2
                                  + 2*log_stds + math.log(2*math.pi))
                log_probs = log_probs.sum(dim=-1, keepdim=True)

                real_log_prob = -0.5*(((action - mean) / (std + EPS))**2
                                      + 2*log_std + math.log(2*math.pi))
                real_log_prob = real_log_prob.sum(dim=-1, keepdim=True)

        # Action between -1 and 1
        action = torch.tanh(action)
        actions_vect = torch.tanh(actions_vect)
        if log_prob:
            log_probs -= torch.log(
                # torch.clamp(1. - actions_vect**2, 0, 1)
                clip_but_pass_gradient(1. - actions_vect**2, 0, 1)
                + 1.e-6
            ).sum(dim=-1, keepdim=True)
            real_log_prob -= torch.log(
                # torch.clamp(1. - action**2, 0, 1)
                clip_but_pass_gradient(1. - action**2, 0, 1)
                + 1.e-6
            ).sum(dim=-1, keepdim=True)

        pol_info = dict()
        pol_info['action_vect'] = actions_vect
        pol_info['log_probs'] = log_probs
        pol_info['log_prob'] = real_log_prob
        pol_info['activation_weights'] = activation_weights
        pol_info['means'] = means
        pol_info['mean'] = mean
        pol_info['log_stds'] = log_stds
        pol_info['log_std'] = log_std
        pol_info['stds'] = stds
        pol_info['std'] = std

        return action, pol_info

    def cuda(self, *args, **kwargs):
        super(MultiPolicyNet, self).cuda(*args, **kwargs)
        self.noise_dist.loc = self.noise_loc
        self.noise_dist.scale = self.noise_scale

    def cpu(self):
        fcn_output = super(MultiPolicyNet, self).cpu()
        self.noise_dist.loc = self.noise_loc
        self.noise_dist.scale = self.noise_scale
        return fcn_output


def get_non_linear_op(name):
    if name.lower() == 'relu':
        activation = torch.nn.ReLU()
    elif name.lower() == 'elu':
        activation = torch.nn.ELU()
    elif name.lower() == 'leaky_relu':
        activation = torch.nn.LeakyReLU()
    elif name.lower() == 'selu':
        activation = torch.nn.SELU()
    elif name.lower() == 'sigmoid':
        activation = torch.nn.Sigmoid()
    elif name.lower() == 'tanh':
        activation = torch.nn.Tanh()
    elif name.lower() in ['linear', 'identity']:
        activation = torch.nn.Sequential()
    else:
        raise AttributeError("Pytorch does not have activation '%s'",
                             name)
    return activation


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = (x > u).to(dtype=torch.float32)
    clip_low = (x < l).to(dtype=torch.float32)
    return x + ((u - x)*clip_up + (l - x)*clip_low).detach()


if __name__ == '__main__':
    print('&&&\n'*2)
    print("Check Intention Network")
    intention = Intention(
        input_size=6,
        hidden_sizes=(5, 3, 4),
        output_size=2,
        non_linear='relu',
        final_non_linear='linear',
        batch_norm=False,
    )
    print('Architecture:\n', intention)
    input_tensor = torch.rand(6).unsqueeze(0)
    print('input', input_tensor)
    output = intention(input_tensor)
    print('output', output)

    batch = 5
    state_dim = 4
    action_dim = 2
    num_intentions = 3

    print('&&&\n'*2)
    print("Check Critic Network")
    critic = MultiQNet(
        num_intentions=num_intentions,
        obs_dim=state_dim,
        action_dim=action_dim,
        shared_sizes=(2, 4),
        intention_sizes=(10, 15, 9),
        shared_non_linear='relu',
        shared_batch_norm=False,
        intention_non_linear='relu',
        intention_final_non_linear='linear',
        intention_batch_norm=False,
    )
    print('Architecture:\n', critic)
    print('Named parameters:')
    for name, param in critic.named_parameters():
        print(name, param.shape, param.is_cuda)
    print('^^^^')
    for name, module in critic.named_children():
        print(name, module)
    state_tensor = torch.rand(batch, state_dim)
    action_tensor = torch.rand(batch, action_dim)
    print('input', state_tensor.shape, action_tensor.shape)
    output = critic(state_tensor, action_tensor)
    print('output', output, output.shape)
    print('one_output', output[:, 0].shape)

    print('&&&\n'*2)
    print("Check Policy Network")
    policy = MultiPolicyNet(
        num_intentions=num_intentions,
        obs_dim=state_dim,
        action_dim=action_dim,
        shared_sizes=(2, 4),
        intention_sizes=(10, 15, 9),
        shared_non_linear='relu',
        shared_batch_norm=False,
        intention_non_linear='relu',
        intention_final_non_linear='linear',
        intention_batch_norm=False,
    )
    print('Architecture:\n', policy)
    print('Named parameters:')
    for name, param in policy.named_parameters():
        print(name, param.shape, param.is_cuda)
    print('^^^^')
    for name, module in policy.named_children():
        print(name, module)
    state_tensor = torch.rand(batch, state_dim)
    print('input', state_tensor.shape)
    output = policy(state_tensor, log_prob=True)[0]
    print('output', output.shape)
    print('one_output', output[:, 0].shape)

    # optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    #
    # des_act = torch.ones_like(output[:, 1]) * -1.0
    #
    # for ii in range(1000):
    #     output = policy(state_tensor, log_prob=False, deterministic=True)
    #     error = torch.mean((des_act - output[:, 1]) ** 2)
    #
    #     optimizer.zero_grad()
    #     error.backward()
    #     # for name, param in policy.named_parameters():
    #     #     if param.grad is None:
    #     #         print(name, None)
    #     #     else:
    #     #         print(name, param.grad.max())
    #     # input('fasdfsdf')
    #     optimizer.step()
    #     print(error.data)
