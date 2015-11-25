import abc
from collections import defaultdict
import cPickle as pickle
import random
from datetime import datetime
import time
import numpy as np
import matplotlib.pylab as plt
import util


class Experiment(object):
    def __init__(self, model, train, dev, observers=None, controllers=None):
        ## Basic Inputs
        self.model = model
        self.train = train
        self.dev = dev

        self.halt = False

        ## Tracking
        if observers is None:
            observers = []
        self.observers = observers
        self.history = defaultdict(lambda: (list(), list()))

        ## Controls
        if controllers is None:
            controllers = [BasicController()]
        self.controllers = controllers

    def run_experiment(self):
        print 'Stochastic Gradient Descent: Examples %d ' % len(self.train)

        self.steps = 0
        self.epochs = 0
        while True:
            # reshuffle training data
            train_copy = list(self.train)
            random.shuffle(train_copy)

            # TODO: Figure out minibatches/ accumulate gradients
            # TODO: Specific to current dataset format!
            for ex in train_copy:
                self.model.backprop(np.concatenate(ex.sentences), ex.question, ex.answer)

                for controller in self.controllers:
                    controller.control(self)

                self.track()
                self.steps += 1

            self.epochs += 1
            if self.halt:
                return

    def track(self):
        report = []
        for observer in self.observers:
            metrics = observer.observe(self)
            if metrics is None:
                continue
            for name, val in metrics.iteritems():
                timestamps, values = self.history[name]
                timestamps.append(self.steps / float(len(self.train)))
                values.append(val)

                util.metadata(name, val)
                report.append((name, val))

        if len(report) > 0:
            print ', '.join(['{}: {:.3f}'.format('.'.join(name), val) for name, val in report])
            with open('history.cpkl', 'w') as f:
                pickle.dump(dict(self.history), f)


class Controller(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def control(self, experiment):
        return


class Observer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def observe(self, experiment):
        return


class BasicController(Controller):

    def __init__(self, report_wait=30, save_wait=30, max_epochs=50):
        self.report_wait = report_wait
        self.save_wait = save_wait
        self.max_epochs = max_epochs

    def control(self, experiment):
        if experiment.epochs >= self.max_epochs:
            print 'Halted after reaching max epochs.'
            experiment.halt = True

        if experiment.steps % self.report_wait == 0:
            print 'steps: {}, epochs: {:.2f}'.format(experiment.steps, experiment.epochs)
            util.metadata('steps', experiment.steps)
            util.metadata('epochs', experiment.epochs)

            # report last seen
            time_rep = datetime.now().strftime('%H:%M:%S %m/%d')
            util.metadata('last_seen', time_rep)

            # report memory used
            util.metadata('gb_used', util.gb_used())

        if experiment.steps % self.save_wait == 0 and experiment.steps != 0:
            print 'saving params...'
            experiment.model.save_params('params.cpkl')


class SpeedObserver(Observer):
    def __init__(self, report_wait=30):
        self.report_wait = report_wait
        self.prev_steps = 0
        self.prev_time = time.time()

    def observe(self, experiment):
        if experiment.steps % self.report_wait != 0:
            return None
        seconds = time.time() - self.prev_time
        steps = experiment.steps - self.prev_steps
        self.prev_time = time.time()
        self.prev_steps = experiment.steps
        return {('speed', 'speed'): steps / seconds}


class ObjectiveObserver(Observer):
    def __init__(self, dset_samples, report_wait):
        self.dset_samples = dset_samples
        self.report_wait = report_wait

    def observe(self, experiment):
        if experiment.steps % self.report_wait == 0:
            def objective_mean(dset):
                sample = util.sample_if_large(dset, self.dset_samples)
                vals = [experiment.model.objective(np.concatenate(ex.sentences),
                        ex.question, ex.answer) for ex in util.verboserate(sample)]
                return np.mean(vals)

            # Note that we never report exact on train
            return {('objective', 'train'): objective_mean(experiment.train),
                    ('objective', 'dev'): objective_mean(experiment.dev)}
        return None


class AccuracyObserver(Observer):
    def __init__(self, dset_samples, report_wait):
        self.dset_samples = dset_samples
        self.report_wait = report_wait

    def observe(self, experiment):
        if experiment.steps % self.report_wait == 0:
            def accuracy_mean(dset):
                sample = util.sample_if_large(dset, self.dset_samples)

                vals = [ex.answer == experiment.model.predict(np.concatenate(ex.sentences), ex.question)
                        for ex in util.verboserate(sample)]
                return np.mean(vals)

            # Note that we never report exact on train
            return {('accuracy', 'train'): accuracy_mean(experiment.train),
                    ('accuracy', 'dev'): accuracy_mean(experiment.dev)}
        return None


def report(path='history.cpkl'):
    with open(path) as f:
        logged_data = pickle.load(f)

    history = util.NestedDict()
    for name, val in logged_data.iteritems():
        history.set_nested(name, val)

    num_subplots = len(history)
    cols = 4  # 4 columns total
    rows = num_subplots / cols + 1

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.1, hspace=0.0)  # no margin between subplots

    # Here we assume that history is only two levels deep
    for k, (subplot_name, trend_lines) in enumerate(history.iteritems()):
        plt.subplot(rows, cols, k + 1)
        plt.title(subplot_name)
        for name, (timestamps, values) in trend_lines.iteritems():
            plt.plot(timestamps, values, label=name)
            # plt.xlabel('Num Epochs')
        plt.legend()

    plt.show()
