import os
import os.path as osp
from plots import plot_multiple_intentions_eval_returns


def list_files_startswith(directory, prefix):
    return list(f for f in os.listdir(directory) if f.startswith(prefix))


env_name = 'navigation2d'
base_log_dir = './logs'
progress_file = 'progress.csv'


log_dict = dict()
log_dict['Main Task'] = dict()
log_dict['Main Task']['SAC'] = dict(
    algo='sac',
    subtask=-1,
    seeds=[610, 810, 1010],
)
log_dict['Main Task']['HIUSAC'] = dict(
    algo='hiusac',
    subtask=-1,
    seeds=[610, 810, 1010],
)

log_dict['Sub Task 1'] = dict()
log_dict['Sub Task 1']['SAC'] = dict(
    algo='sac',
    subtask=0,
    seeds=[610, 810, 1010],
)
log_dict['Sub Task 1']['HIUSAC'] = dict(
    algo='hiusac',
    subtask=0,
    seeds=[610, 810, 1010],
)

log_dict['Sub Task 2'] = dict()
log_dict['Sub Task 2']['SAC'] = dict(
    algo='sac',
    subtask=1,
    seeds=[610, 810, 1010],
)
log_dict['Sub Task 2']['HIUSAC'] = dict(
    algo='hiusac',
    subtask=1,
    seeds=[610, 810, 1010],
)

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
            recent_dir = all_matching_log_dirs[-1]
            file = osp.join(full_path, recent_dir, progress_file)
            plot_progress_files.append(file)

        ld['progress_files'] = plot_progress_files

        if algo == 'sac':
            ld['subtask'] = -1

plot_multiple_intentions_eval_returns(log_dict)

input('fasdf')


print("The script has finished. Bye!")


