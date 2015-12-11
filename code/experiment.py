import sys
import abc
from collections import defaultdict
import cPickle as pickle
import random
from datetime import datetime
import time
import numpy as np
import matplotlib.pylab as plt
import util
from os.path import join


class Experiment(object):
    def __init__(self, model, train, dev, observers=None, controllers=None, path=''):
        ## Basic Inputs
        self.model = model
        self.train = train
        self.dev = dev
        self.path = path

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
            for ex in util.verboserate(train_copy):
                self.model.backprop(ex.sentences, ex.mask, ex.question,
                                    ex.answer[0], ex.hints)

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

                util.metadata(name, val, self.path)
                report.append((name, val))

        if len(report) > 0:
            print ', '.join(['{}: {:.3f}'.format('.'.join(name), val)
                            for name, val in report])
            sys.stdout.flush()
            with open(join(self.path, 'history.cpkl'), 'w') as f:
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

    def __init__(self, report_wait=30, save_wait=30, max_epochs=50, path=''):
        self.report_wait = report_wait
        self.save_wait = save_wait
        self.max_epochs = max_epochs
        self.path = path

    def control(self, experiment):
        if experiment.epochs >= self.max_epochs:
            print 'Halted after reaching max epochs.'
            experiment.halt = True

        if experiment.steps % self.report_wait == 0:
            print 'steps: {}, epochs: {:.2f}'.format(experiment.steps,
                                                     experiment.epochs)
            util.metadata('steps', experiment.steps, self.path)
            util.metadata('epochs', experiment.epochs, self.path)

            # report last seen
            time_rep = datetime.now().strftime('%H:%M:%S %m/%d')
            util.metadata('last_seen', time_rep, self.path)

            # report memory used
            util.metadata('gb_used', util.gb_used(), self.path)

        if experiment.steps % self.save_wait == 0 and experiment.steps != 0:
            print 'saving params...'
            experiment.model.save_model(self.path)


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
                vals = []
                for ex in util.verboserate(sample):
                    vals.append(experiment.model.objective(ex.sentences,
                                ex.mask, ex.question, ex.answer[0], ex.hints))
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

                vals = []
                for ex in util.verboserate(sample):
                    correct = ex.answer == experiment.model.predict(ex.sentences, ex.mask, ex.question)
                    vals.append(correct)
                return np.mean(vals)

            # Note that we never report exact on train
            return {('accuracy', 'train'): accuracy_mean(experiment.train),
                    ('accuracy', 'dev'): accuracy_mean(experiment.dev)}
        return None


class TestObserver(Observer):
    def __init__(self, test_dset, report_wait):
        self.test_dset = test_dset
        self.report_wait = report_wait

    def observe(self, experiment):
        if experiment.steps % self.report_wait == 0:
            def accuracy_mean(dset):
                vals = []
                for ex in util.verboserate(dset):
                    correct = ex.answer == experiment.model.predict(ex.sentences, ex.mask, ex.question)
                    vals.append(correct)
                return np.mean(vals)
            return {('TEST_ACCURACY', 'TEST_ACCURACY'): accuracy_mean(self.test_dset)}
        return None


def report(path='history.cpkl', tn=0, sns=True):
    if sns:
        import seaborn as sns
        sns.set_style('whitegrid')
        sns.set_style('whitegrid', {'fontsize': 50})
        sns.set_context('poster')
    with open(path) as f:
        logged_data = pickle.load(f)

    history = util.NestedDict()
    for name, val in logged_data.iteritems():
        history.set_nested(name, val)

    num_subplots = len(history)
    cols = 2  # 2 panels for Objective and Accuracy
    rows = 1

    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.2)  # room for labels [Objective, Accuracy]
    colors = [sns.xkcd_rgb['blue'], sns.xkcd_rgb['red']]

    # Here we assume that history is only two levels deep
    for k, (subplot_name, trend_lines) in enumerate(history.iteritems()):
        plt.subplot(rows, cols, k + 1)
        plt.ylabel(subplot_name.capitalize())
        plt.xlabel('Epoch')
        for i, (name, (timestamps, values)) in enumerate(trend_lines.iteritems()):
            plt.plot(timestamps, values, label=name, color=colors[i])
        plt.suptitle('Task number %d' % tn)
        plt.legend(loc='best')

    plt.show()
