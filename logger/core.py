import os
import json
import datetime
import dateutil.tz
from .logger import log, dict_to_safe_json, log_variant, add_tabular_output
from .logger import set_snapshot_dir, set_snapshot_mode, set_snapshot_gap
from .logger import set_log_stdout

LOCAL_LOG_DIR = './training_logs'


def setup_logger(
        exp_prefix='',
        seed=0,
        variant=None,
        log_dir=None,
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_stdout=True,
):
    """

    Args:
        exp_prefix (str): Experiment prefix. All experiments with this prefix
            will have log directories be under this directory.
        seed (int): Seed value.
        variant (dict or None): Experiment variant dictionary.
        log_dir (str or None): Logging directory.
        variant_log_file (str): Experiment variant filename.
        tabular_log_file (str): Tablular filename.
        snapshot_mode (str): Available options: 'last', 'all', 'gap', 'gap_and_last'
        snapshot_gap (int): Snapshot gap if 'gap' or 'gap_and_last' mode is selected.
        log_stdout (bool): Show logging in stdout.

    Returns:
        str: Full path name of the logging directory.

    """
    set_log_stdout(log_stdout)
    set_snapshot_mode(snapshot_mode)
    set_snapshot_gap(snapshot_gap)

    # Set log directory
    if log_dir is None:
        log_dir = LOCAL_LOG_DIR
    log_dir = create_log_dir(exp_prefix, seed=seed, base_log_dir=log_dir)
    if variant is not None:
        log("Variant:")
        log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = os.path.join(log_dir, variant_log_file)
        log_variant(variant_log_path, variant)

    tabular_log_path = os.path.join(log_dir, tabular_log_file)
    add_tabular_output(tabular_log_path)

    set_snapshot_dir(log_dir)

    return log_dir


def create_log_dir(exp_prefix='', seed=0, base_log_dir=None):
    """Creates and returns a unique log directory.

    Args:
        exp_prefix (str): Experiment prefix. All experiments with this prefix
            will have log directories be under this directory.
        seed (int): Seed number of the experiment.
        base_log_dir (str): Full path where the log directory will be created.

    Returns:
        str: Name of the logging directory.

    """
    # exp_name = create_exp_name(exp_prefix.replace("_", "-"), seed=seed)
    exp_name = create_exp_name(exp_prefix='', seed=seed)
    if base_log_dir is None:
        base_log_dir = LOCAL_LOG_DIR
    log_dir = os.path.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    # log_dir = osp.join(base_log_dir, exp_name)
    if os.path.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_exp_name(exp_prefix='', seed=0):
    """Create a semi-unique experiment name that has a timestamp.

    Args:
        exp_prefix (str): Experiment prefix. All experiments with this prefix
            will have log directories be under this directory.
        seed (int): Seed value.

    Returns:
        str: semi-unique name.

    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    if exp_prefix is None or exp_prefix == '':
        return "s-%d---%s" % (seed, timestamp)
    else:
        return "%s--s-%d---%s" % (exp_prefix, seed, timestamp)
