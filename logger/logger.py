"""
This file is basically rlkit's logger, with little modifications.

https://github.com/vitchyr/rlkit
https://github.com/rll/rllab
"""
from enum import Enum
from contextlib import contextmanager
import numpy as np
import os
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import joblib
import json
import errno

from .tabulate import tabulate


LOCAL_LOG_DIR = './logs'


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


_prefixes = []
_prefix_str = ''

_tabular_prefixes = []
_tabular_prefix_str = ''

_tabular = []

_text_outputs = []
_tabular_outputs = []

_text_fds = {}
_tabular_fds = {}
_tabular_header_written = set()

_snapshot_dir = None
_snapshot_mode = 'all'
_snapshot_gap = 1

_log_tabular_only = False
_header_printed = False

_log_stdout = True


def _add_output(file_name, arr, fds, mode='a'):
    if file_name not in arr:
        mkdir_p(os.path.dirname(file_name))
        arr.append(file_name)
        fds[file_name] = open(file_name, mode)


def _remove_output(file_name, arr, fds):
    if file_name in arr:
        fds[file_name].close()
        del fds[file_name]
        arr.remove(file_name)


def push_prefix(prefix):
    _prefixes.append(prefix)
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def add_text_output(file_name):
    _add_output(file_name, _text_outputs, _text_fds, mode='a')


def remove_text_output(file_name):
    _remove_output(file_name, _text_outputs, _text_fds)


def add_tabular_output(file_name):
    _add_output(file_name, _tabular_outputs, _tabular_fds, mode='w')


def remove_tabular_output(file_name):
    if _tabular_fds[file_name] in _tabular_header_written:
        _tabular_header_written.remove(_tabular_fds[file_name])
    _remove_output(file_name, _tabular_outputs, _tabular_fds)


def set_snapshot_dir(dir_name):
    global _snapshot_dir
    _snapshot_dir = dir_name


def get_snapshot_dir():
    return _snapshot_dir


def get_snapshot_mode():
    return _snapshot_mode


def set_snapshot_mode(mode):
    global _snapshot_mode
    _snapshot_mode = mode


def get_snapshot_gap():
    return _snapshot_gap


def set_snapshot_gap(gap):
    global _snapshot_gap
    _snapshot_gap = gap


def set_log_tabular_only(log_tabular_only):
    global _log_tabular_only
    _log_tabular_only = log_tabular_only


def get_log_tabular_only():
    return _log_tabular_only


def set_log_stdout(log_stdout):
    global _log_stdout
    _log_stdout = log_stdout


def get_log_stdout():
    return _log_stdout


def log(s, with_prefix=True, with_timestamp=True):
    out = s
    if with_prefix:
        out = _prefix_str + out
    if with_timestamp:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
        out = "%s | %s" % (timestamp, out)
    if not _log_tabular_only:
        if _log_stdout:
            # Also log to stdout
            print(out)
        for fd in list(_text_fds.values()):
            fd.write(out + '\n')
            fd.flush()
        sys.stdout.flush()


def record_tabular(key, val):
    _tabular.append((_tabular_prefix_str + str(key), str(val)))


def push_tabular_prefix(key):
    _tabular_prefixes.append(key)
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


def pop_tabular_prefix():
    del _tabular_prefixes[-1]
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


def save_extra_data(data):
    """
    Data saved here will always override the last entry

    :param data: Something pickle'able.
    """
    if _snapshot_dir:
        file_name = osp.join(_snapshot_dir, 'extra_data.pkl')
        joblib.dump(data, file_name, compress=3)


def get_table_dict():
    return dict(_tabular)


def get_table_key_set():
    return set(key for key, value in _tabular)


@contextmanager
def prefix(key):
    push_prefix(key)
    try:
        yield
    finally:
        pop_prefix()


@contextmanager
def tabular_prefix(key):
    push_tabular_prefix(key)
    yield
    pop_tabular_prefix()


class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


table_printer = TerminalTablePrinter()


def dump_tabular(*args, **kwargs):
    wh = kwargs.pop("write_header", None)
    if len(_tabular) > 0:
        if _log_tabular_only:
            table_printer.print_tabular(_tabular)
        else:
            for line in tabulate(_tabular).split('\n'):
                log(line, *args, **kwargs)
        tabular_dict = dict(_tabular)
        # Also write to the csv files
        # This assumes that the keys in each iteration won't change!
        for tabular_fd in list(_tabular_fds.values()):
            writer = csv.DictWriter(tabular_fd,
                                    fieldnames=list(tabular_dict.keys()))
            if wh or (wh is None and tabular_fd not in _tabular_header_written):
                writer.writeheader()
                _tabular_header_written.add(tabular_fd)
            writer.writerow(tabular_dict)
            tabular_fd.flush()
        del _tabular[:]


def pop_prefix():
    del _prefixes[-1]
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def save_itr_params(itr, params):
    if _snapshot_dir:
        if _snapshot_mode == 'all':
            file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % itr)
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == 'last':
            # override previous params
            file_name = osp.join(_snapshot_dir, 'params.pkl')
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == "gap":
            if itr % _snapshot_gap == 0:
                file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % itr)
                joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == "gap_and_last":
            if itr % _snapshot_gap == 0:
                file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % itr)
                joblib.dump(params, file_name, compress=3)
            file_name = osp.join(_snapshot_dir, 'params.pkl')
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == 'none':
            pass
        else:
            raise NotImplementedError


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name}
        return json.JSONEncoder.default(self, o)


def log_variant(log_file, variant_data):
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)


def record_tabular_misc_stat(key, values, placement='back'):
    if placement == 'front':
        prefix = ""
        suffix = key
    else:
        prefix = key
        suffix = ""
    if len(values) > 0:
        record_tabular(prefix + "Average" + suffix, np.average(values))
        record_tabular(prefix + "Std" + suffix, np.std(values))
        record_tabular(prefix + "Median" + suffix, np.median(values))
        record_tabular(prefix + "Min" + suffix, np.min(values))
        record_tabular(prefix + "Max" + suffix, np.max(values))
    else:
        record_tabular(prefix + "Average" + suffix, np.nan)
        record_tabular(prefix + "Std" + suffix, np.nan)
        record_tabular(prefix + "Median" + suffix, np.nan)
        record_tabular(prefix + "Min" + suffix, np.nan)
        record_tabular(prefix + "Max" + suffix, np.nan)


def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)


def create_log_dir(exp_prefix, exp_id=0, seed=0, base_log_dir=None):
    """
    Creates and returns a unique log directory.

    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: Different exp_ids will be in different directories.
    :return:
    """
    exp_name = create_exp_name(exp_prefix, exp_id=exp_id,
                               seed=seed)
    if base_log_dir is None:
        base_log_dir = LOCAL_LOG_DIR
    log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_logger(
        exp_prefix="default",
        exp_id=0,
        seed=0,
        log_dir=None,
        tabular_log_file="progress.csv",
        snapshot_mode="last",
):
    if log_dir is None:
        log_dir = LOCAL_LOG_DIR

    log_dir = create_log_dir(exp_prefix, exp_id=exp_id, seed=seed,
                             base_log_dir=log_dir)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    add_tabular_output(tabular_log_path)

    set_snapshot_dir(log_dir)
    set_snapshot_mode(snapshot_mode)

    return log_dir
