import os
import os.path as osp
from plots import plot_multiple_intentions_eval_returns, set_latex_plot


def list_files_startswith(directory, prefix):
    return list(f for f in os.listdir(directory) if f.startswith(prefix))


# env_name = 'navigation2d'
# env_name = 'reacher'  # Do not use 810
env_name = 'pusher'
# env_name = 'centauro'
base_log_dir = './logs'
progress_file = 'progress.csv'
# max_iters = None
max_iters = 250
max_iters = 300
# seeds = [610, 810, 1010, 710, 910]
# seeds = [610, 1010, 910, 710]
# seeds = [610, 810, 1010]
# seeds = [710, 1010, 610]
seeds = [1010, 610, 810]  # Pusher
# seeds = [1010, 610, 710]  # Centauro
# seeds = [510, 610, 710, 810, 910, 1010]
# seeds = [610, 710, 1010, 510]  # Reacher

# seeds = [510]
# seeds = [610]
# seeds = [710]
# seeds = [810]
# seeds = [910]
# seeds = [1010]

last_idx = -1
# last_idx = -2
# last_idx = -3
#
save_fig_name = osp.join('paper_plots', 'learning_' + env_name + 'X.pdf')


log_dict = dict()
log_dict['Compound Task'] = dict()
log_dict['Compound Task']['SAC'] = dict(
    algo='sac',
    subtask=-1,
    seeds=seeds,
    last_idx=-1,
    color=None,
)
log_dict['Compound Task']['HIUSAC-1'] = dict(
    algo='hiusac',
    subtask=-1,
    seeds=seeds,
    last_idx=-1,
    color=None,
)
log_dict['Compound Task']['HIUSAC-2'] = dict(
    algo='hiusac-p',
    subtask=-1,
    seeds=seeds,
    last_idx=last_idx,
    color=None,
)
# log_dict['Compound Task']['HIUSAC-2-2'] = dict(
#     algo='hiusac-p',
#     subtask=-1,
#     seeds=seeds,
#     last_idx=-1,
#     color='red',
# )

log_dict['Composable Task 1'] = dict()
log_dict['Composable Task 1']['SAC'] = dict(
    algo='sac',
    subtask=0,
    seeds=seeds,
    last_idx=-1,
    color=None,
)
log_dict['Composable Task 1']['HIUSAC-1'] = dict(
    algo='hiusac',
    subtask=0,
    seeds=seeds,
    last_idx=-1,
    color=None,
)
log_dict['Composable Task 1']['HIUSAC-2'] = dict(
    algo='hiusac-p',
    subtask=0,
    seeds=seeds,
    last_idx=last_idx,
    color=None,
)
# log_dict['Composable Task 1']['HIUSAC-2-2'] = dict(
#     algo='hiusac-p',
#     subtask=0,
#     seeds=seeds,
#     last_idx=-1,
#     color='red',
# )

log_dict['Composable Task 2'] = dict()
log_dict['Composable Task 2']['SAC'] = dict(
    algo='sac',
    subtask=1,
    seeds=seeds,
    last_idx=-1,
    color=None,
)
log_dict['Composable Task 2']['HIUSAC-1'] = dict(
    algo='hiusac',
    subtask=1,
    seeds=seeds,
    last_idx=-1,
    color=None,
)
log_dict['Composable Task 2']['HIUSAC-2'] = dict(
    algo='hiusac-p',
    subtask=1,
    seeds=seeds,
    last_idx=last_idx,
    color=None,
)
# log_dict['Composable Task 2']['HIUSAC-2-2'] = dict(
#     algo='hiusac-p',
#     subtask=1,
#     seeds=seeds,
#     last_idx=-1,
#     color='red',
# )

for pd in log_dict.values():
    for ld in pd.values():
        algo = ld['algo']
        stask_idx = ld['subtask']
        if algo == 'sac':
            stask_idx = ld['subtask']
        else:
            stask_idx = -1
        full_path = osp.join(
            base_log_dir,
            env_name,
            'sub'+str(stask_idx),
            env_name+'-'+algo
        )
        plot_progress_files = list()
        for seed in ld['seeds']:
            log_dir_name = env_name+'-'+algo+'--s-'+str(seed)+'--'
            all_matching_log_dirs = list_files_startswith(full_path, log_dir_name)
            all_matching_log_dirs.sort()
            if not len(all_matching_log_dirs) > 0:
                raise FileNotFoundError("No file found starting with % s" %
                                        (osp.join(full_path, log_dir_name)+'*'))
            if abs(ld['last_idx']) > len(all_matching_log_dirs):
                ld['last_idx'] = -1
            recent_dir = all_matching_log_dirs[ld['last_idx']]
            file = osp.join(full_path, recent_dir, progress_file)
            plot_progress_files.append(file)

        ld['progress_files'] = plot_progress_files

        if algo == 'sac':
            ld['subtask'] = -1

set_latex_plot()

plot_multiple_intentions_eval_returns(
    log_dict,
    max_iters=max_iters,
    block=False,
    save_fig_name=save_fig_name,
)

# input('Press a key to close the script...')


print("The script has finished. Bye!")


