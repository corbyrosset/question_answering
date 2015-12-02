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
                next[idx] += '-' + var + ' ' + str(v) + ' '

            new_options += next
        options = new_options
    else:
        for idx in xrange(len(options)):
            options[idx] += '-' + var + ' ' + str(val) + ' '


def get_logging_dir(opt):
    return 'experiments/' + opt.replace(' ', '').replace('--', '-')[1:]


def submit_qsubscript(command, log_dir):
    qsubscript = '''
#!/bin/bash
#$ -cwd
#$ -N %s
#$ -o %s
#$ -S /bin/bash
#$ -j y

   %s

''' % (log_dir, command)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    qsubfile = open("%s/run.submit" % log_dir, 'w')
    qsubfile.write(qsubscript)
    qsubfile.close()
    os.system('qsub %s/run.submit' % log_dir)


## COMMAND LINE ##
# for opt in options:
#     log_dir = get_logging_dir(opt)
#     if not os.path.exists(log_dir):
#         os.mkdir(log_dir)
#     opt += '--log ' + log_dir
#     command = 'python %s %s' % (args.task, opt)
#     print command
#     os.system(command)

## BARLEY ##
for opt in options:
    log_dir = get_logging_dir(opt)
    opt += '--log ' + log_dir
    command = 'python %s %s' % (args.task, opt)
    submit_qsubscript(command, log_dir)

## TODO: NLP CLUSTER/CODALAB ##
