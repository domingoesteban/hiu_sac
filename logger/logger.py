"""
This file is basically rllab/rlkit logger, with little changes.

https://github.com/rll/rllab
https://github.com/vitchyr/rlkit
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
import torch

from .tabulate import tabulate


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

# _snapshot_dir = None
_snapshot_dir = os.path.join('.', 'training_logs')
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
    """

    Args:
        s (str): String to log in file descriptors and optionally stdout.
        with_prefix (bool): Add a prefix to the log.
        with_timestamp (bool): Add a timestamp before the string.

    Returns:

    """

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
            sys.stdout.write(out + '\n')
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

    Returns:
        None

    """
    if _snapshot_dir:
        file_name = osp.join(_snapshot_dir, 'extra_data.pkl')
        joblib.dump(data, file_name, compress=3)


def get_table_dict():
    """Returns a dictionary from the current table.

    Returns:
        dict:

    """
    return dict(_tabular)


def get_table_key_set():
    """Returns a set from the current table.

    Returns:
        set:

    """
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


def save_torch_models(num_iter, models_dict, replaceable_models_dict=None):
    """Save things with pytorch.save

    Returns:

    """
    save_full_path = os.path.join(
        _snapshot_dir,
        'models'
    )

    if _snapshot_mode == 'all':
        models_dirs = list((
            os.path.join(
                save_full_path,
                str('itr_%03d' % num_iter)
            ),
        ))
    elif _snapshot_mode == 'last':
        models_dirs = list((
            os.path.join(
                save_full_path,
                str('last_itr')
            ),
        ))
    elif _snapshot_mode == 'gap':
        if num_iter % _snapshot_gap == 0:
            models_dirs = list((
                os.path.join(
                    save_full_path,
                    str('itr_%03d' % num_iter)
                ),
            ))
        else:
            return
    elif _snapshot_mode == 'gap_and_last':
        models_dirs = list((
            os.path.join(
                save_full_path,
                str('last_itr')
            ),
        ))
        if num_iter % _snapshot_gap == 0:
            models_dirs.append(
                os.path.join(
                    save_full_path,
                    str('itr_%03d' % num_iter)
                ),
            )
    elif _snapshot_mode == 'none':
        return
    else:
        raise NotImplementedError

    for save_path in models_dirs:
        # logger.log('Saving models to %s' % save_full_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for key, value in models_dict.items():
            torch.save(value, os.path.join(save_path, key + '.pt'))

    if replaceable_models_dict:
        if num_iter % _snapshot_gap == 0:  # TODO: See how to do at the end of episode
            if not os.path.exists(save_full_path):
                os.makedirs(save_full_path)
            for key, value in models_dict.items():
                torch.save(value, os.path.join(save_full_path, key + '.pt'))


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


def dict_to_safe_json(d):
    """Convert each value in the dictionary into a JSON'able primitive.

    Args:
        d (dict): requested dictionary

    Returns:
        dict: New dictionary.

    """
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(
            isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False
