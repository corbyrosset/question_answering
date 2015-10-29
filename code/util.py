import sys
import time


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