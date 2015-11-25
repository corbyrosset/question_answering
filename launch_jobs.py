import os
import argparse
import copy
import configs


parser = argparse.ArgumentParser()

# task arguments
parser.add_argument('config')
parser.add_argument('--task', default='code/notebooks/qa_experiment.py')
args = parser.parse_args()

config = getattr(configs, args.config)

options = [""]
for var, val in config.iteritems():
    if isinstance(val, list):
        new_options = []
        for v in val:
            next = copy.deepcopy(options)
            for idx in xrange(len(options)):
                next[idx] += '--' + var + ' ' + str(v) + ' '

            new_options += next
        options = new_options
    else:
        for idx in xrange(len(options)):
            options[idx] += '--' + var + ' ' + str(val) + ' '

for opt in options:
    command = 'python %s %s' % (args.task, opt)
    print command
    os.system(command)
