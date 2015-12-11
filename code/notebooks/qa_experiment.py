
# coding: utf-8

# In[ ]:

import scriptinit
import random
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from models import *
from data_import import *
from util import *
import layers
import optimizers
from experiment import *
from os.path import join
import argparse


# In[ ]:

# Turn on for debugging
DEBUG = False
if DEBUG:
    theano.config.optimizer='fast_compile'
    theano.config.exception_verbosity='high'


# In[ ]:

# variables that don't change between experiments/trials
constants = {
    'datadir': 'data/',
    'glovedir': 'data/glove',
    'report_wait': 500,
    'save_wait': 1000,
    'wv_dimensions': 50,  # speed up learning by using the smallest GloVe dimension
}


# In[ ]:

# step up argument parsing
parser = argparse.ArgumentParser()

parser.add_argument('-tn', '--task_number', type=int, required=True)
parser.add_argument('-lr', '--base_lr', type=float, required=True)
parser.add_argument('-mt', '--model_type', type=str, required=True)
parser.add_argument('-hd', '--hidden_dim', type=int, required=True)
parser.add_argument('-l2', '--l2_reg', type=float, required=True)

parser.add_argument('-lhd', '--lstm_hidden_dim', type=int, required=True)
parser.add_argument('-mp', '--mean_pool', type=int, required=True)
parser.add_argument('-log', '--logging_path', type=str, required=True)

parser.add_argument('-me', '--max_epochs', type=int, required=True)


# In[ ]:

# variables that change between runs
if util.in_ipython():
    constants = {
        'datadir': '../../data/',
        'glovedir': '../../data/glove',
        'report_wait': 100,
        'save_wait': 1000,
        'max_epochs': 100,
        'wv_dimensions': 50,  # speed up learning by using the smallest GloVe dimension
    }

    hyperparams = {
        # data and logging
        'task_number': 1,  # {1, 3, 5, 17, 19}

        # HYPERPARAMETERS
        'base_lr': 1e-2,

        # all of the models
        'model_type': 'sentenceEmbedding',  # one of |sentenceEmbedding| or |averaging|
        'hidden_dim': 128,
        'l2_reg': 1e-7,


        # specific to sentence embedding model
        'lstm_hidden_dim': 128,
        'mean_pool': False,
        
        'logging_path': 'logging_dir', 
    }
else:
    args = parser.parse_args()
    hyperparams = vars(args)


# In[ ]:

# load into namespace and log to metadata
for var, val in hyperparams.iteritems():
    exec("{0} = hyperparams['{0}']".format(var))
    util.metadata(var, val)

for var, val in constants.iteritems():
    exec("{0} = constants['{0}']".format(var))
    util.metadata(var, val)


# In[ ]:

# Load Data
train_ex, _ = get_data(datadir, task_number, test=False)
word_vectors = wordVectors(train_ex)
train_ex = examples_to_example_ind(word_vectors, train_ex)  # get examples in numerical format

# shuffle data
random.shuffle(train_ex)

# split the train into 90% train, 10% dev
train = train_ex[:int(.9 * len(train_ex))]
dev = train_ex[int(.9 * len(train_ex)):]


# In[ ]:

# get word_vectors (Using glove for now)
wv_matrix = word_vectors.get_wv_matrix(wv_dimensions, glovedir)


# In[ ]:

# initial and setup the attention layer
r = 0.001
attention_embedding = np.random.rand(hidden_dim, wv_matrix.shape[1]) * 2 * r - r

attention_model = attentionModel(embeddings=attention_embedding, lstm_hidden_dim=lstm_hidden_dim)

# set up the question + facts -> answer model
if model_type == 'averaging':
    qa_model = averagingModel(wv_matrix, hidden_dim=hidden_dim, num_classes=wv_matrix.shape[1])
elif model_type == 'sentenceEmbedding':
    qa_model = embeddingModel(wv_matrix, lstm_hidden_dim=lstm_hidden_dim, nn_hidden_dim=hidden_dim,
                              num_classes=wv_matrix.shape[1], mean_pool=mean_pool)


# In[ ]:

# set up saving and loading 
def save_model(path):
    attention_model.save_params(join(path, 'attention_params.cpkl'))
    qa_model.save_params(join(path, 'qa_params.cpkl'))

def load_model(path):
    attention_model.load_params(join(path, 'attention_params.cpkl'))
    qa_model.load_params(join(path, 'qa_params.cpkl'))

qa_model.save_model = save_model
qa_model.load_model = load_model


# In[ ]:

# generate answer probabilities and predictions
support = T.imatrix()
mask = T.imatrix()
question_idxs = T.ivector()
hints = T.ivector()

# estimate relevance of each sentence
relevance_probs = attention_model.get_relevance_probs(support, mask, question_idxs)

def get_word_idxs(relevant_sentence_idxs, support_, mask_):
    rel_support = support_[relevant_sentence_idxs, :]
    rel_mask = mask_[relevant_sentence_idxs, :]
    return rel_support[rel_mask.nonzero()].ravel()

# By default, the attention model retrieves any sentence with prob > 0.5 under the model
# If no sentence exists, it returns the top two sentences in chronological order
max_idxs = T.sort(T.argsort(relevance_probs[:, 1])[-2:])  
prob_idxs = T.arange(relevance_probs.shape[0])[T.nonzero(relevance_probs[:, 1] > 0.5)]
est_idxs = ifelse(T.lt(T.sum(relevance_probs[:, 1] > 0.5), 1), max_idxs, prob_idxs)


# joint training of question model + attention model
est_rel_facts = get_word_idxs(est_idxs, support, mask)
answer_probs = qa_model.get_answer_probs(est_rel_facts, question_idxs)

# train the qa-model using the hints
# true_rel_facts = get_word_idxs(hints.nonzero(), support, mask)
# answer_probs = qa_model.get_answer_probs(true_rel_facts, question_idxs)

# predict without using the hint
answer_pred = T.argmax(qa_model.get_answer_probs(est_rel_facts, question_idxs))
#answer_pred = T.argmax(qa_model.get_answer_probs(true_rel_facts, question_idxs))


# In[ ]:

# define the loss and cost function
answer = T.iscalar()

qa_nll = -T.log(answer_probs)[0, answer]
attention_nll = -T.sum(T.log(relevance_probs)[T.arange(hints.shape[0]), hints])
loss = qa_nll + attention_nll

cost = loss + l2_reg * layers.l2_penalty(qa_model.params + attention_model.params)

param_norms = layers.l2_penalty(qa_model.params + attention_model.params)


# In[ ]:

# optimization
updates = optimizers.Adagrad(cost, qa_model.params + attention_model.params, base_lr=base_lr)


# In[ ]:

# compile functions to train and evaluate the model
print 'Compiling predict function'
qa_model.predict = theano.function(
                   inputs = [support, mask, question_idxs],
                   outputs = answer_pred)

print 'Compiling objective function'
qa_model.objective = theano.function(
                    inputs = [support, mask, question_idxs, answer, hints],
                    outputs = loss)

print 'Compiling backprop function'
qa_model.backprop = theano.function(
                    inputs=[support, mask, question_idxs, answer, hints],
                    outputs=[loss, answer_probs, qa_nll, attention_nll],
                    updates=updates)

# Function needed for error analysis
print 'Compiling diagnostic function'
qa_model.diagnostic = theano.function(inputs=[support, mask, question_idxs], outputs=[answer_pred, est_idxs])


# In[ ]:

# Set up the experiment object
controllers = [BasicController(report_wait=report_wait, save_wait=save_wait, max_epochs=max_epochs, path=logging_path)]

dset_samples = len(dev)
observers = [ObjectiveObserver(dset_samples=dset_samples, report_wait=report_wait),
             AccuracyObserver(dset_samples=dset_samples, report_wait=report_wait)]

experiment = Experiment(qa_model, train, dev, controllers=controllers, observers=observers, path=logging_path)


# In[ ]:

# launch the experiment
experiment.run_experiment()


# In[ ]:

## Plot learning curves
#report(join(logging_path, 'history.cpkl'))


# In[ ]:

## Error Analysis
model_path = 'logging_dir'
qa_model.load_model(model_path)


# In[ ]:

from diagnostics import *
error_analysis(dev, qa_model, word_vectors)

