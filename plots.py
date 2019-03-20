import sys
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import traceback
import plots


def get_csv_data(csv_file, labels, space_separated=False):
    data, all_labels = get_csv_data_and_labels(csv_file,
                                               space_separated=space_separated)
    # Uncommont to print the labels
    print(csv_file)
    for label in all_labels:
        print(label)
    print('***\n'*3)
    n_data = data.shape[0]

    new_data = np.zeros((len(labels), n_data))

    # # Uncomment for debugging
    # print(all_labels)

    for ll, name in enumerate(labels):
        if name in all_labels:
            idx = all_labels.index(name)
            try:
                new_data[ll, :] = data[:, idx]
            except Exception:
                print(traceback.format_exc())
                print("Error with data in %s" % csv_file)
                sys.exit(1)
        else:
            raise ValueError("Label '%s' not available in file '%s'"
                             % (name, csv_file))

    return new_data


def get_csv_data_and_labels(csv_file, space_separated=False):
    # Read from CSV file
    try:
        if space_separated:
            series = pd.read_csv(csv_file, delim_whitespace=True)
        else:
            series = pd.read_csv(csv_file)
    except Exception:
        print(traceback.format_exc())
        print("Error reading %s" % csv_file)
        sys.exit(1)

    data = series.as_matrix()
    labels = list(series)

    return data, labels


def set_latex_plot():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # rc('font', **{'family': 'serif','serif':['Times']})
    matplotlib.rcParams['font.family'] = ['serif']
    matplotlib.rcParams['font.serif'] = ['Times New Roman']


def subplots(*args, **kwargs):
    fig, axs = plt.subplots(*args, **kwargs)

    if isinstance(axs, np.ndarray):
        for aa in axs:
            axis_format(aa)
    else:
        axis_format(axs)

    return fig, axs


def fig_format(fig):
    fig.subplots_adjust(hspace=0)
    fig.set_facecolor((1, 1, 1))


def axis_format(axis):
    # axis.tick_params(axis='x', labelsize=25)
    # axis.tick_params(axis='y', labelsize=25)
    axis.tick_params(axis='x', labelsize=15)
    axis.tick_params(axis='y', labelsize=15)

    # Background
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.xaxis.grid(color='white', linewidth=2)
    axis.set_facecolor((0.917, 0.917, 0.949))


def plot_intentions_eval_returns(csv_file, num_intentions=None, block=False):
    labels_to_plot = ['Test Returns Mean']

    if num_intentions is None:
        num_intentions = 0
    # else:
    #     num_intentions += 1

    # Add Intentional-Unintentional Label
    new_labels = list()
    for label in labels_to_plot:
        for uu in range(num_intentions):
            new_string = label + (' [%02d]' % uu)
            new_labels.append(new_string)

        # Assuming the Main does not have a prefix
        new_string = label
        new_labels.append(new_string)

    n_subplots = len(labels_to_plot) * (num_intentions + 1)

    data = get_csv_data(csv_file, new_labels)

    fig, axs = subplots(n_subplots)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Average Return', fontweight='bold')

    for aa, ax in enumerate(axs):
        ax.plot(data[aa])
        ax.set_ylabel(new_labels[aa])
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)

    print('total_iters:', len(data[-1]))
    plt.show(block=block)


def plot_intentions_info(csv_file, num_intentions=None, block=False):
    infos_to_plot = [
        'Alpha',
        'Entropy',
    ]

    adim=3
    for ii in range(adim):
        infos_to_plot.append('Std Action %02d' % ii)
    # for ii in range(adim):
    #     infos_to_plot.append('Mean Action %02d' % ii)

    if num_intentions is None:
        num_intentions = 0
    # else:
    #     num_intentions += 1

    for label in infos_to_plot:
        # Add Intentional-Unintentional Label
        new_labels = list()
        for uu in range(num_intentions):
            new_string = label + (' [U-%02d]' % uu)
            new_labels.append(new_string)

        # Assuming the Main does not have a prefix
        new_string = label
        new_labels.append(new_string)

        n_subplots = (num_intentions + 1)

        data = get_csv_data(csv_file, new_labels)

        fig, axs = subplots(n_subplots)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        fig.subplots_adjust(hspace=0)
        fig.suptitle(label, fontweight='bold')

        for aa, ax in enumerate(axs):
            ax.plot(data[aa])
            ax.set_ylabel(new_labels[aa])
            plt.setp(ax.get_xticklabels(), visible=False)

        axs[-1].set_xlabel('Episodes')
        plt.setp(axs[-1].get_xticklabels(), visible=True)

    print('total_iters:', len(data[-1]))
    plt.show(block=block)


def plot_contours(ax, x_tensor, y_tensor, values):
    contours = ax.contour(x_tensor, y_tensor, values, 20,
                          colors='dimgray')
    ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
    ax.imshow(values, extent=(x_tensor.min(), x_tensor.max(),
                              y_tensor.min(), y_tensor.max()),
              origin='lower', alpha=0.5)


def plot_q_values(qf, action_lower, action_higher, obs, policy=None,
                  obs_dims=(0, 1), action_dims=(0, 1), delta=0.01,
                  device='cpu'):
    # Values Plots
    num_intentions = 2

    action_dim_x = action_dims[0]
    action_dim_y = action_dims[1]
    obs_dim_x = obs_dims[0]
    obs_dim_y = obs_dims[1]

    x_min = action_lower[action_dim_x]
    y_min = action_lower[action_dim_y]
    x_max = action_higher[action_dim_x]
    y_max = action_higher[action_dim_y]

    action_dim = len(action_lower)
    all_x = torch.arange(float(x_min), float(x_max), float(delta))
    all_y = torch.arange(float(y_min), float(y_max), float(delta))
    x_mesh, y_mesh = torch.meshgrid(all_x, all_y)

    x_mesh = x_mesh.t()
    y_mesh = y_mesh.t()
    all_acts = torch.stack((x_mesh, y_mesh), dim=-1)

    fig, all_axs = \
        subplots(1, num_intentions + 1,
                 gridspec_kw={'wspace': 0, 'hspace': 0},
                 )
    # fig.suptitle('Q-val Observation: ' + str(ob))
    fig.tight_layout()
    fig.canvas.set_window_title('q_vals_%1d_%1d' %
                                (obs[obs_dim_x], obs[obs_dim_y]))

    all_axs = np.atleast_1d(all_axs)


    all_obs = torch.tensor(obs, dtype=torch.float32, device=device)
    all_obs = all_obs.expand_as(all_acts)

    q_vals = qf(all_obs, all_acts).squeeze(-1).detach()

    if policy is not None:
        n_samples = 100
        torch_obs = torch.tensor(obs, dtype=torch.float32, device=device)
        torch_obs = torch_obs.expand(n_samples, -1)
        actions, pol_info = policy(torch_obs,
                                   deterministic=False,
                                   intention=None)
        actions = actions.detach().numpy()
        intention_actions = pol_info['u_actions'].detach().numpy()

    for intention in range(num_intentions + 1):
        if intention < num_intentions:
            ax = all_axs[intention+1]
        else:
            ax = all_axs[0]
        plot_contours(ax, x_mesh, y_mesh, q_vals[:, :, intention])

        ax.set_xlabel('Action %2d' % action_dims[0], #fontweight='bold',
                      fontsize=24)
        ax.set_ylabel('Action %2d' % action_dims[1], #fontweight='bold',
                      fontsize=24)
        ax.axis('equal')
        ax.set_aspect('equal', 'box')
        ax.grid(False)

        ax.set_xlim(x_min)
        ax.set_ylim(y_min)

        if policy is not None:
            if intention < num_intentions:
                x = intention_actions[:, intention, action_dim_x]
                y = intention_actions[:, intention, action_dim_y]
            else:
                x = actions[:, action_dim_x]
                y = actions[:, action_dim_y]
            ax.scatter(x, y, c='b', marker='*', zorder=5)

        if intention < num_intentions:
            ax.set_title('Composable Task %01d' % (intention+1),
                         fontdict={'fontsize': 32,
                                   'fontweight': 'medium'}
                         )
        else:
            ax.set_title('Compound Task',
                         fontdict={'fontsize': 32,
                                   'fontweight': 'medium'})
        if ax != all_axs[0]:
        # if intention > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
            # ax.yaxis.set_visible(False)
            ax.yaxis.label.set_visible(False)

        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.show()


def plot_multiple_intentions_eval_returns(
        csv_file_dict,
        max_iters=None,
        block=False,
        save_fig_name=None,
):
    label_to_plot = 'Test Returns Mean'
    label_x_axis = 'Episodes'
    label_y_axis = 'Average Return'

    # x_major_loc = 10
    # x_minor_loc = 5
    x_major_loc = 50
    x_minor_loc = 25
    # x_major_loc = 20
    # x_minor_loc = 10

    # # One per figure
    # title_fontsize = 50
    # axis_fontsize = 40
    # legend_fontsize = 40
    # xtick_fontsize = 25
    # ytick_fontsize = 15
    # linewidth = 5
    # std_alpha = .20

    # Three per figure
    title_fontsize = 16
    axis_fontsize = 14
    legend_fontsize = 14
    xtick_fontsize = 9
    ytick_fontsize = 10
    # linewidth = 3
    linewidth = 1
    std_alpha = .20
    
    z_confi = 1.282  # 80%
    # z_confi = 1.440  # 85%
    # z_confi = 1.645  # 90%
    # z_confi = 1.960  # 95%
    # z_confi = 2.576  # 99%

    color_list = [
        (0.55294118,  0.62745098,  0.79607843),  # Blue
        (0.98823529,  0.55294118,  0.38431373),  # Orange
        (0.40000000,  0.76078431,  0.64705882),  # Green
        'red',
    ]

    fig, axs = subplots(1, len(csv_file_dict))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)
    # fig.canvas.set_window_title('cucu'.replace(" ", "_"))

    for ax, (fig_name, fig_dict) in zip(axs, csv_file_dict.items()):
    # for fig_name, fig_dict in csv_file_dict.items():
        # fig, axs = subplots(1)
        # if not isinstance(axs, np.ndarray):
        #     axs = np.array([axs])
        # fig.subplots_adjust(hspace=0)
        # fig.canvas.set_window_title(fig_name.replace(" ", "_"))
        # ax = axs[-1]

        if ax == axs[0]:
            lines = list()
            labels = list()

        max_x = 0
        min_x = 0
        min_xdiff = np.inf
        for cc, (item_name, item_dict) in enumerate(fig_dict.items()):
            subtask = item_dict['subtask']

            color = item_dict['color']
            if color is None:
                if item_dict['algo'].lower() in 'sac':
                    color = color_list[0]
                elif item_dict['algo'].lower() in 'hiusac':
                    color = color_list[1]
                elif item_dict['algo'].lower() in 'hiusac-p':
                    color = color_list[2]
                else:
                    color = color_list[3]

            if subtask > -1:
                new_label_to_plot = label_to_plot + (' [%02d]' % subtask)
            else:
                new_label_to_plot = label_to_plot
            data_list = list()
            max_available_iters = max_iters
            for csv_file in item_dict['progress_files']:
                data = get_csv_data(csv_file, [new_label_to_plot]).squeeze()
                data_list.append(data)
                print('&&&')
                print('&&&', csv_file, '--->', len(data))
                print('&&&')
                max_available_iters = min(max_available_iters, len(data))
            data_list = [data[:min(max_available_iters, max_iters)]
                         for data in data_list]
            seeded_data = np.array(data_list)
            n_seeds = seeded_data.shape[0]

            data_mean = seeded_data.mean(axis=0)
            data_std = seeded_data.std(axis=0)
            x_data = np.arange(0, len(data_mean))

            # interactions_label = 'Accumulated Training Steps'
            # ninteraction_list = list()
            # for csv_file in item_dict['progress_files']:
            #     data = get_csv_data(csv_file, [interactions_label]).squeeze()
            #     ninteraction_list.append(ninteraction_data)
            # ninteraction_data = np.array(ninteraction_list)
            # ninteract = ninteraction_data.mean(axis=0)

            if max_iters is not None:
                data_mean = data_mean[:max_iters]
                data_std = data_std[:max_iters]
                x_data = x_data[:max_iters]

            ax.fill_between(
                x_data,
                (data_mean - z_confi * data_std/np.sqrt(n_seeds)),
                (data_mean + z_confi * data_std/np.sqrt(n_seeds)),
                alpha=std_alpha,
                color=color,
            )
            mean_plot = ax.plot(
                x_data,
                data_mean,
                color=color,
                zorder=10,
                linestyle='-',
                linewidth=linewidth,
            )[0]
            if ax == axs[0]:
                lines.append(mean_plot)

                # if item_dict['algo'].lower() in ['hiusac', 'hiusac-p', 'hiusac-m']:
                #     if subtask == -1:
                #         i_suffix = ' [I]'
                #     else:
                #         i_suffix = ' [U-%02d]' % (subtask + 1)
                # else:
                #     i_suffix = ''
                i_suffix = ''
                labels.append(item_name + i_suffix)

            max_x = max(max_x, x_data[-1])
            min_x = min(min_x, x_data[0])
            min_xdiff = min(min_xdiff, x_data[1] - x_data[0])

        # xdiff = x_data[1] - x_data[0]
        # xdiff = max_x - min_x
        ax.set_xlim(min_x-min_xdiff*1.5, max_x + min_xdiff*2.5)
        ax.set_ylabel(label_y_axis, fontsize=axis_fontsize)
        ax.xaxis.set_major_locator(plt.MultipleLocator(x_major_loc))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(x_minor_loc))
        # plt.setp(ax.get_xticklabels(), visible=True)

        ax.set_xlabel(label_x_axis, fontsize=axis_fontsize)
        # plt.setp(axs[-1].get_xticklabels(), visible=True)

        ax.xaxis.set_tick_params(labelsize=xtick_fontsize)
        ax.yaxis.set_tick_params(labelsize=ytick_fontsize)

        ax.set_title(fig_name, fontsize=title_fontsize)
        #
        # # legend = fig.legend(lines, labels, loc='lower right', ncol=1,
        # #                     # legend = fig.legend(lines, labels, loc=(-1, 0), ncol=1,
        # #                     labelspacing=0., prop={'size': legend_fontsize})
        # # legend.draggable(True)
        #
        # fig.set_size_inches(19*3, 11)  # 1920 x 1080

        # fig.tight_layout()

        # ax.get_yaxis().set_visible(False)
        # plt.setp(ax.get_ylabels(), visible=True)
        plt.setp(ax.get_yticklines(), visible=True)
        ax.tick_params(axis="y", direction="in", pad=-22)
        ax.tick_params(labelleft=False)

        ax.yaxis.label.set_visible(False)

    legend = fig.legend(lines, labels, loc='lower center', ncol=3,
                        labelspacing=0., prop={'size': legend_fontsize})
    # legend.draggable(True)

    axs[0].get_yaxis().set_visible(True)
    axs[0].yaxis.label.set_visible(True)
    fig.set_size_inches(12, 3)  # 1920 x 1080
    # fig.set_size_inches(20, 7)  # 1920 x 1080
    fig.subplots_adjust(wspace=0)
    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.31)
    plt.subplots_adjust(wspace=0.03, hspace=0)

    if save_fig_name is not None:
        fig.savefig(save_fig_name)

    # plt.show(block=block)


def plot_navitation2d():
    from envs import get_normalized_env
    env, env_params = get_normalized_env(
        'navigation2d',
        None,
        610,
        False
    )
    env.render()

    colors = np.array([
        'red',
        'green',
        'blue',
        'black',
        'purple',
    ])

    obs = [
        (-2., -2.),
        (-2., 4.),
        (4., -2.),
        (4., 4.),
        (-6., -6.),
    ]

    for ob, color in zip(obs, colors):
        env._wrapped_env._robot_marker(
            env._wrapped_env._main_ax,
            ob[0],
            ob[1],
            color=color,
            zoom=0.03
        )
