import sys
import time
import numpy as np
from collections import MutableMapping
import json
import resource
import platform
import math
from os.path import join


def memoize(f):
    cache = {}

    def decorated(*args):
        if args not in cache:
            cache[args] = f(*args)
        else:
            print 'loading cached values for {}'.format(args)
        return cache[args]

    return decorated


def verboserate(iterable, time_wait=5, report=None):
    """
    Iterate verbosely.
    """
    try:
        total = len(iterable)
    except TypeError:
        total = '?'

    def default_report(steps, elapsed):
        print '{} of {} processed ({} s)'.format(steps, total, elapsed)
        sys.stdout.flush()

    if report is None:
        report = default_report

    start = time.time()
    prev = start
    for steps, val in enumerate(iterable):
        current = time.time()
        since_prev = current - prev
        elapsed = current - start
        if since_prev > time_wait:
            report(steps, elapsed)
            prev = current
        yield val


def sample_if_large(arr, max_size, replace=True):
    if len(arr) > max_size:
        idx = np.random.choice(len(arr), size=max_size, replace=replace)
        return [arr[i] for i in idx]

    return list(arr)

class NestedDict(MutableMapping):
    def __init__(self):
        self.d = {}

    def __iter__(self):
        return self.d.__iter__()

    def __delitem__(self, key):
        return self.d.__delitem__(key)

    def __getitem__(self, key):
        try:
            return self.d.__getitem__(key)
        except KeyError:
            val = NestedDict()
            self.d[key] = val
            return val

    def __len__(self):
        return self.d.__len__()

    def __setitem__(self, key, value):
        return self.d.__setitem__(key, value)

    def get_nested(self, keys):
        d = self
        for k in keys:
            d = d[k]
        return d

    def set_nested(self, keys, val):
        d = self.get_nested(keys[:-1])
        return d.__setitem__(keys[-1], val)

    def __repr__(self):
        return self.d.__repr__()

    def as_dict(self):
        items = []
        for key, sub in self.iteritems():
            if isinstance(sub, NestedDict):
                val = sub.as_dict()
            else:
                val = sub
            items.append((key, val))
        return dict(items)

meta = NestedDict()
def metadata(keys, val, path=''):
    """
    Sets entries in a nested dictionary called meta.
    After each call, meta is updated and saved to meta.json in the current directory

    keys = either a string or a tuple of strings
    a tuple of strings will be interpreted as nested keys in a dictionary, i.e. dictionary[key1][key2][...]
    """
    # This is only designed to be used with CodaLab
    if isinstance(keys, tuple):
        meta.set_nested(keys, val)
    else:
        # if there is actually just one key
        meta[keys] = val

    # sync with file
    with open(join(path, 'meta.json'), 'w') as f:
        d = meta.as_dict()  # json only handles dicts
        json.dump(d, f)

def gb_used():
    used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != 'Darwin':
        # on Linux, used is in terms of kilobytes
        power = 2
    else:
        # on Mac, used is in terms of bytes
        power = 3
    return float(used) / math.pow(1024, power)