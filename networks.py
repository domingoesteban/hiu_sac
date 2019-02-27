import torch
import math
from functools import reduce

LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
LOG_SIG_MIN = -6.907755  # SIGMA 0.001
EPS = 1e-8


class NNModule(torch.nn.Module):
    """
    NN Module for modeling Q-values and activation weights
    """
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 non_linear,
                 final_non_linear,
                 batch_norm=False,
                 ):
        super(NNModule, self).__init__()
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
        init_gain_name = self.non_linear_name
        if init_gain_name == 'elu':
            init_gain_name = 'relu'
        init_gain = torch.nn.init.calculate_gain(init_gain_name)
        for layer in self.layers:
            init_fcn(layer.weight.data, gain=init_gain)
            torch.nn.init.constant_(layer.bias.data, 0)

        # Initialize output layer
        init_gain_name = self.output_non_linear_name
        if init_gain_name == 'elu':
            init_gain_name = 'relu'
        init_gain = torch.nn.init.calculate_gain(init_gain_name)
        init_fcn(self.olayer.weight.data, gain=init_gain)
        torch.nn.init.constant_(self.olayer.bias.data, 0)

    def forward(self, x):
        for ll in range(len(self.layers)):
            x = self.non_linear(self.layers[ll](x))
            if self.batch_norm:
                x = self.layer_norms[ll](x)
        x = self.output_non_linear(self.olayer(x))
        return x


class ComposablePolicy(torch.nn.Module):
    """
    Composable Gaussian Policy Module
    """
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 non_linear,
                 final_non_linear,
                 batch_norm=False,
                 ):
        super(ComposablePolicy, self).__init__()
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

        # Mean and log_std outputs
        self.mean_layer = torch.nn.Linear(i_size, output_size)
        self.log_std_layer = torch.nn.Linear(i_size, output_size)

        # Initialize weights
        self.init_weights('uniform')

    def init_weights(self, init_fcn='uniform'):
        if init_fcn.lower() == 'uniform':
            init_fcn = torch.nn.init.xavier_uniform_
        else:
            init_fcn = torch.nn.init.xavier_normal_

        # Initialize hidden layers
        init_gain_name = self.non_linear_name
        if init_gain_name == 'elu':
            init_gain_name = 'relu'
        init_gain = torch.nn.init.calculate_gain(init_gain_name)
        for layer in self.layers:
            init_fcn(layer.weight.data, gain=init_gain)
            torch.nn.init.constant_(layer.bias.data, 0)

        # Initialize mean and log_std layer
        init_gain_name = self.output_non_linear_name
        if init_gain_name == 'elu':
            init_gain_name = 'relu'
        init_gain = torch.nn.init.calculate_gain(init_gain_name)
        init_fcn(self.mean_layer.weight.data, gain=init_gain)
        init_fcn(self.log_std_layer.weight.data, gain=init_gain)
        torch.nn.init.constant_(self.mean_layer.bias.data, 0)
        torch.nn.init.constant_(self.log_std_layer.bias.data, 0)

    def forward(self, x):
        for ll in range(len(self.layers)):
            x = self.non_linear(self.layers[ll](x))
            if self.batch_norm:
                x = self.layer_norms[ll](x)
        mean = self.output_non_linear(self.mean_layer(x))
        log_std = self.output_non_linear(self.log_std_layer(x))
        return mean, log_std


class MultiValueNet(torch.nn.Module):
    """
    Base NN for both Q-values and V-values
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
                 normalize_inputs=False,
                 ):
        super(MultiValueNet, self).__init__()

        self.input_dim = input_dim
        self.shared_non_linear_name = shared_non_linear
        self.shared_batch_norm = shared_batch_norm

        self.shared_non_linear = get_non_linear_op(self.shared_non_linear_name)

        # Shared layers
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

        # Value NN modules
        self.value_nets = list()
        for ii in range(num_intentions):
            critic_net = NNModule(
                input_size=i_size,
                hidden_sizes=intention_sizes,
                output_size=1,
                non_linear=intention_non_linear,
                final_non_linear=intention_final_non_linear,
                batch_norm=intention_batch_norm,
            )
            self.value_nets.append(critic_net)
            self.add_module('intention{}'.format(ii), critic_net)

        # (Optional) input normalization
        if normalize_inputs:
            self.add_module('input_normalization', Normalizer(self.input_dim))
        else:
            self.input_normalization = None

    def init_weights(self, init_fcn='uniform'):
        if init_fcn.lower() == 'uniform':
            init_fcn = torch.nn.init.xavier_uniform_
        else:
            init_fcn = torch.nn.init.xavier_normal_

        # Initialize shared layers
        init_gain_name = self.shared_non_linear_name
        if init_gain_name == 'elu':
            init_gain_name = 'relu'
        init_gain = torch.nn.init.calculate_gain(init_gain_name)
        for layer in self.shared_layers:
            init_fcn(layer.weight.data, gain=init_gain)
            torch.nn.init.constant_(layer.bias.data, 0)

    def forward(self, x, intention=None):
        if self.input_normalization is not None:
            x = self.input_normalization(x)

        for ll in range(len(self.shared_layers)):
            x = self.shared_non_linear(self.shared_layers[ll](x))
            if self.shared_batch_norm:
                x = self.shared_layer_norms[ll](x)

        if intention is None:
            critic_nets = self.value_nets  # All critics
        else:
            critic_nets = [self.value_nets[intention]]  # Requested critic

        values = list()
        for critic in critic_nets:
            value = critic(x).unsqueeze(dim=-2)
            values.append(value)

        values_vect = torch.cat(values, dim=-2)

        return values_vect


class MultiQNet(MultiValueNet):
    """
    State-Action Value Function
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
                 normalize_inputs=False,
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
            normalize_inputs=normalize_inputs,
        )

    def forward(self, observation, action, intention=None):
        x = torch.cat((observation, action), dim=-1)
        return super(MultiQNet, self).forward(x, intention=intention)


class MultiVNet(MultiValueNet):
    """
    State Value Function
    """
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
                 normalize_inputs=False,
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
            normalize_inputs=normalize_inputs,
        )

    def forward(self, observation, intention=None):
        return super(MultiVNet, self).forward(observation, intention=intention)


class MultiPolicyNet(torch.nn.Module):
    """
    Hierarchical Policy
    """
    def __init__(self,
                 num_intentions,
                 obs_dim,
                 action_dim,
                 shared_sizes,
                 unintention_sizes,
                 shared_non_linear='relu',
                 shared_batch_norm=False,
                 intention_non_linear='relu',
                 intention_final_non_linear='linear',
                 intention_batch_norm=False,
                 combination_method='convex',
                 normalize_inputs=False,
                 ):
        super(MultiPolicyNet, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_intentions = num_intentions
        self.shared_non_linear_name = shared_non_linear
        self.shared_batch_norm = shared_batch_norm

        self.shared_non_linear = get_non_linear_op(self.shared_non_linear_name)
        self.combination_method = combination_method

        self.register_buffer('_pols_idxs', torch.arange(self.num_intentions))

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

        # Composable Policies
        self.policy_nets = list()
        for ii in range(num_intentions):
            # policy_net = NNModule(
            #     input_size=i_size,
            #     hidden_sizes=unintention_sizes,
            #     output_size=action_dim*2,  # Output means and log_stds
            #     non_linear=intention_non_linear,
            #     final_non_linear=intention_final_non_linear,
            #     batch_norm=intention_batch_norm,
            # )
            policy_net = ComposablePolicy(
                input_size=i_size,
                hidden_sizes=unintention_sizes,
                output_size=action_dim,
                non_linear=intention_non_linear,
                final_non_linear=intention_final_non_linear,
                batch_norm=intention_batch_norm,
            )
            self.policy_nets.append(policy_net)
            self.add_module('unintention{}'.format(ii), policy_net)

        # Activation vector module
        self.weights_net = NNModule(
            input_size=i_size,
            hidden_sizes=unintention_sizes,
            output_size=num_intentions*self.action_dim,  # Activation vectors
            non_linear=intention_non_linear,
            final_non_linear=intention_final_non_linear,
            batch_norm=intention_batch_norm,
        )

        if self.combination_method == 'convex':
            # self.combination_non_linear = torch.nn.Softmax(dim=-2)
            self.combination_non_linear = get_non_linear_op('softmax', dim=-2)
        elif self.combination_method == 'product':
            # self.combination_non_linear = torch.nn.Sigmoid()
            self.combination_non_linear = get_non_linear_op('sigmoid')
        else:
            self.combination_non_linear = get_non_linear_op('linear')

        if normalize_inputs:
            self.add_module('input_normalization', Normalizer(obs_dim))
        else:
            self.input_normalization = None

        # Initialize weights
        self.init_weights('uniform')

    def init_weights(self, init_fcn='uniform'):
        if init_fcn.lower() == 'uniform':
            init_fcn = torch.nn.init.xavier_uniform_
        else:
            init_fcn = torch.nn.init.xavier_normal_

        # Initialize shared layers
        init_gain_name = self.shared_non_linear_name
        if init_gain_name == 'elu':
            init_gain_name = 'relu'
        init_gain = torch.nn.init.calculate_gain(init_gain_name)
        for layer in self.shared_layers:
            init_fcn(layer.weight.data, gain=init_gain)
            torch.nn.init.constant_(layer.bias.data, 0)

    def gaussian_composition(self, means, variances, activation_weights):
        """
        Function 'f' in paper.
        :param means: Composable policy means
        :param variances: Composable policy variances
        :param activation_weights: Activation vectors
        :return: (compound policy, compound variance vector)
        """
        # Compound policy variance
        if self.combination_method == 'convex':  # Option 1
            compound_variance = torch.sum(
                variances.detach()*(activation_weights**2),
                dim=-2,
                keepdim=False,
                )
        elif self.combination_method == 'product':  # Option 2
            sig_invs = activation_weights/variances.detach()
            compound_variance = 1./torch.sum(sig_invs, dim=-2, keepdim=False)
        elif self.combination_method == 'gmm':  # Option 3
            # Sample latent variables
            z = torch.distributions.Multinomial(
                logits=activation_weights.transpose(-2, -1)
            ).sample().transpose(-2, -1)  # Batch x nIntent
            compound_variance = torch.sum(
                variances.detach()*z,
                dim=-2,
                keepdim=False,
                )
        else:
            raise ValueError("Wrong combination method!")

        # Compound policy mean
        if self.combination_method == 'convex':  # Option 1
            compound_mean = torch.sum(
                means.detach()*activation_weights,
                dim=-2,
                keepdim=False,
            )
        elif self.combination_method == 'product':  # Option 2
            compound_mean = compound_variance*torch.sum(
                means.detach()*sig_invs,
                dim=-2,
                keepdim=False
            )
        elif self.combination_method == 'gmm':  # Option 3
            compound_mean = torch.sum(
                means.detach()*z,
                dim=-2,
                keepdim=False,
            )
        else:
            raise ValueError("Wrong combination method!")

        return compound_mean, compound_variance

    def forward(self, observation, deterministic=False, intention=None,
                log_prob=False,
                ):
        if log_prob and deterministic:
            raise ValueError("It is not possible to calculate log_probs in "
                             "deterministic policies.")
        if self.input_normalization is not None:
            observation = self.input_normalization(observation)

        x = observation
        for ll in range(len(self.shared_layers)):
            x = self.shared_non_linear(self.shared_layers[ll](x))
            if self.shared_batch_norm:
                x = self.shared_layer_norms[ll](x)

        # Composable means and std vectors
        u_means = list()
        u_log_stds = list()
        for policy in self.policy_nets:
            # pol_params = policy(x).unsqueeze(dim=-2)
            # means.append(pol_params[:, :, :self.action_dim])
            # log_stds.append(pol_params[:, :, self.action_dim:])
            pol_params = policy(x)
            u_means.append(pol_params[0].unsqueeze(dim=-2))
            u_log_stds.append(pol_params[1].unsqueeze(dim=-2))

        u_means = torch.cat(u_means, dim=-2)

        # It is not efficient to calculate log_std, stds, and variances with
        # the deterministic option but it is easier to organize everything.
        u_log_stds = torch.cat(u_log_stds, dim=-2)
        # # Method 1:
        # log_stds = torch.tanh(log_stds)
        # log_stds = \
        #     LOG_SIG_MIN + 0.5 * (LOG_SIG_MAX - LOG_SIG_MIN)*(log_stds + 1)
        # Method 2:
        u_log_stds = torch.clamp(u_log_stds, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        u_stds = u_log_stds.exp()
        u_variances = u_stds**2

        # Composable activation vectors
        activation_weights = self.weights_net(x)
        activation_weights = activation_weights.reshape(-1,
                                                        self.num_intentions,
                                                        self.action_dim)
        activation_weights = self.combination_non_linear(activation_weights)

        # If the policy has only one unintention become as intention
        if not self.num_intentions > 1:
            intention = 0

        if intention is None:
            i_mean, i_variance = self.gaussian_composition(
                u_means,
                u_variances,
                activation_weights
            )

            i_std = torch.sqrt(i_variance)
            # # Method 1:
            # std = torch.tanh(std)
            # std = math.exp(LOG_SIG_MIN) + \
            #     0.5 * (math.exp(LOG_SIG_MAX) - math.exp(LOG_SIG_MIN))*(std + 1)
            # Method 2:
            i_std = torch.clamp(i_std,
                                min=math.exp(LOG_SIG_MIN),
                                max=math.exp(LOG_SIG_MAX))
            i_log_std = torch.log(i_std)
        else:
            intention = self._pols_idxs[intention]
            i_mean = \
                torch.index_select(u_means, dim=-2, index=intention).squeeze(-2)

            i_variance = \
                torch.index_select(u_variances, dim=1, index=intention).squeeze(-2)
            i_std = \
                torch.index_select(u_stds, dim=1, index=intention).squeeze(-2)
            i_log_std = \
                torch.index_select(u_log_stds, dim=1, index=intention).squeeze(-2)

        u_log_probs = None
        i_log_prob = None
        if deterministic:
            u_actions = u_means
            i_action = i_mean
        else:
            # Sample from Gaussian distribution
            noise = torch.randn_like(i_std)
            u_actions = u_stds*noise.unsqueeze(1) + u_means
            i_action = i_std*noise + i_mean

            if log_prob:
                u_log_probs = -0.5*(((u_actions - u_means) / (u_stds + EPS))**2
                                    + 2*u_log_stds + math.log(2*math.pi))
                u_log_probs = u_log_probs.sum(dim=-1, keepdim=True)

                i_log_prob = -0.5*(((i_action - i_mean) / (i_std + EPS))**2
                                   + 2*i_log_std + math.log(2*math.pi))
                i_log_prob = i_log_prob.sum(dim=-1, keepdim=True)

        # Action between -1 and 1
        i_action = torch.tanh(i_action)
        u_actions = torch.tanh(u_actions)
        if not deterministic and log_prob:
            u_log_probs -= torch.log(
                # # Method 1
                # torch.clamp(1. - u_actions**2, 0, 1)
                # Method 2
                clip_but_pass_gradient(1. - u_actions**2, 0, 1)
                + 1.e-6
            ).sum(dim=-1, keepdim=True)
            i_log_prob -= torch.log(
                # # Method 1
                # torch.clamp(1. - i_action**2, 0, 1)
                # Method 2
                clip_but_pass_gradient(1. - i_action**2, 0, 1)
                + 1.e-6
            ).sum(dim=-1, keepdim=True)

        # print(activation_weights[0, :, :].mean(dim=1).detach().cpu().numpy())

        pol_info = dict()
        pol_info['u_actions'] = u_actions  # Batch x nIntent x dA
        pol_info['u_log_probs'] = u_log_probs  # Batch x nIntent x 1
        pol_info['i_log_prob'] = i_log_prob  # Batch x 1
        pol_info['activation_weights'] = activation_weights # Batch x nInt x dA
        pol_info['u_means'] = u_means  # Batch x nIntent x dA
        pol_info['i_mean'] = i_mean  # Batch x dA
        pol_info['u_log_stds'] = u_log_stds  # Batch x nIntent x dA
        pol_info['i_log_std'] = i_log_std  # Batch x dA
        pol_info['u_stds'] = u_stds  # Batch x nIntent x dA
        pol_info['i_std'] = i_std  # Batch x dA
        pol_info['u_variances'] = u_variances  # Batch x nIntent x dA
        pol_info['i_variance'] = i_variance  # Batch x dA

        return i_action, pol_info


def get_non_linear_op(op_name, **kwargs):
    if op_name.lower() == 'relu':
        activation = torch.nn.ReLU(**kwargs)
    elif op_name.lower() == 'elu':
        activation = torch.nn.ELU(**kwargs)
    elif op_name.lower() == 'leaky_relu':
        activation = torch.nn.LeakyReLU(**kwargs)
    elif op_name.lower() == 'selu':
        activation = torch.nn.SELU(**kwargs)
    elif op_name.lower() == 'softmax':
        activation = torch.nn.Softmax(**kwargs)
    elif op_name.lower() == 'sigmoid':
        activation = torch.nn.Sigmoid()
    elif op_name.lower() == 'tanh':
        activation = torch.nn.Tanh()
    elif op_name.lower() in ['linear', 'identity']:
        activation = torch.nn.Sequential()
    else:
        raise AttributeError("Pytorch does not have activation '%s'",
                             op_name)
    return activation


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = (x > u).to(dtype=torch.float32)
    clip_low = (x < l).to(dtype=torch.float32)
    return x + ((u - x)*clip_up + (l - x)*clip_low).detach()


class Normalizer(torch.nn.Module):
    def __init__(
            self,
            size,
            eps=1e-8,
            default_clip_range=math.inf,
            mean=0,
            std=1,
    ):
        super(Normalizer, self).__init__()
        self.size = size
        self.default_clip_range = default_clip_range

        self.register_buffer('sum', torch.zeros((self.size,)))
        self.register_buffer('sumsq', torch.zeros((self.size,)))
        self.register_buffer('count', torch.zeros((1,)))
        self.register_buffer('mean', mean + torch.zeros((self.size,)))
        self.register_buffer('std', std * torch.ones((self.size,)))
        self.register_buffer('eps', eps * torch.ones((self.size,)))

        self.synchronized = True

    def forward(self, x):
        init_shape = x.shape
        x = x.view(-1, self.size)

        if self.training:
            self.update(x)
        x = self.normalize(x)

        x = x.view(init_shape)
        return x

    def update(self, v):
        if v.dim() == 1:
            v = v.expand(0)
        assert v.dim() == 2
        assert v.shape[1] == self.size
        self.sum += v.sum(dim=0)
        self.sumsq += (v**2).sum(dim=0)
        self.count[0] += v.shape[0]
        self.synchronized = False

    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean, std = self.mean, self.std
        if v.dim() == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean, std = self.mean, self.std
        if v.dim() == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return mean + v * std

    def synchronize(self):
        self.mean.data = self.sum / self.count[0]
        self.std.data = torch.sqrt(
            torch.max(
                self.eps**2,
                self.sumsq / self.count[0] - self.mean**2
            )
        )
        self.synchronized = True


if __name__ == '__main__':
    torch.cuda.manual_seed(500)
    torch.manual_seed(500)

    obs_dim = 2
    batch = 1

    # device = 'cuda:0'
    device = 'cpu'

    normalizer = Normalizer(obs_dim)
    normalizer.to(device)

    for ii in range(500):
        obs1 = torch.FloatTensor(batch, 1).uniform_(10, 50).to(device)
        obs2 = torch.FloatTensor(batch, 1).uniform_(-3000, -1000).to(device)

        cosa = torch.cat((obs1, obs2), dim=1)

        print('prev_mean', cosa.mean(dim=0))
        print('prev_std', cosa.std(dim=0))

        cosa = normalizer(cosa)
        print('after_mean', cosa.mean(dim=0))
        print('after_norm.mean',  normalizer.mean)
        print('after_std', cosa.std(dim=0))
        print('after_norm.std',  normalizer.std)
        print('--')

    input('fasd')

    # print('&&&\n'*2)
    # print("Check Intention Network")
    # intention = Intention(
    #     input_size=6,
    #     hidden_sizes=(5, 3, 4),
    #     output_size=2,
    #     non_linear='relu',
    #     final_non_linear='linear',
    #     batch_norm=False,
    # )
    # print('Architecture:\n', intention)
    # input_tensor = torch.rand(6).unsqueeze(0)
    # print('input', input_tensor)
    # output = intention(input_tensor)
    # print('output', output)

    batch = 5
    state_dim = 4
    action_dim = 2
    num_intentions = 1

    # print('&&&\n'*2)
    # print("Check Critic Network")
    # critic = MultiQNet(
    #     num_intentions=num_intentions,
    #     obs_dim=state_dim,
    #     action_dim=action_dim,
    #     shared_sizes=(2, 4),
    #     intention_sizes=(10, 15, 9),
    #     shared_non_linear='relu',
    #     shared_batch_norm=False,
    #     intention_non_linear='relu',
    #     intention_final_non_linear='linear',
    #     intention_batch_norm=False,
    # )
    # print('Architecture:\n', critic)
    # print('Named parameters:')
    # for name, param in critic.named_parameters():
    #     print(name, param.shape, param.is_cuda)
    # print('^^^^')
    # for name, module in critic.named_children():
    #     print(name, module)
    # state_tensor = torch.rand(batch, state_dim)
    # action_tensor = torch.rand(batch, action_dim)
    # print('input', state_tensor.shape, action_tensor.shape)
    # output = critic(state_tensor, action_tensor)
    # print('output', output, output.shape)
    # print('one_output', output[:, 0].shape)

    print('&&&\n'*2)
    print("Check Policy Network")
    policy = MultiPolicyNet(
        num_intentions=num_intentions,
        obs_dim=state_dim,
        action_dim=action_dim,
        shared_sizes=(5, 4),
        unintention_sizes=(10, 15, 3),
        shared_non_linear='relu',
        shared_batch_norm=False,
        intention_non_linear='relu',
        intention_final_non_linear='linear',
        intention_batch_norm=False,
        normalize_inputs=True,
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
    output, output_info = policy(state_tensor, log_prob=True)
    print('output', output.shape)
    print('one_output', output[:, 0].shape)

    des_means = torch.randn(num_intentions, action_dim)
    des_mean = torch.randn(action_dim)
    min_std = 0.001
    max_std = 5.2
    des_std = (min_std - max_std) * torch.rand(action_dim) + max_std
    des_stds = (min_std - max_std) * torch.rand(num_intentions, action_dim) + max_std

    loss_fcn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    errors = list()
    for i in range(5000):
        output, output_info = policy(state_tensor, log_prob=True)
        error_mean = loss_fcn(output_info['mean'], des_mean.expand_as(output_info['mean']))
        error_means = loss_fcn(output_info['means'], des_means.expand_as(output_info['means']))
        error_std = loss_fcn(output_info['log_std'].exp(), des_std.expand_as(output_info['log_std']))
        error_stds = loss_fcn(output_info['log_stds'].exp(), des_stds.expand_as(output_info['log_stds']))

        total_error = error_mean + error_means + error_std + error_stds

        optimizer.zero_grad()
        total_error.backward()
        optimizer.step()
        print(total_error.item(), '', error_mean.item(), error_means.item(),
              '', error_std.item(), error_stds.item())
        errors.append(total_error.item())

    import matplotlib.pyplot as plt
    plt.plot(errors)
    plt.show()

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
