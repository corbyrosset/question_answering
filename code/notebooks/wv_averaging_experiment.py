
# coding: utf-8

# In[ ]:

import sys
sys.path.append('../')
import random
import numpy as np
import theano
import theano.tensor as T
from models import *
from data_import import *
from util import *
import layers
import optimizers


# In[ ]:

# CONSTANTS
datadir = '../../data/'
glovedir = '../../data/glove'
SEED = 999
# np.random.seed(SEED)

task_number = 1 # simple task for now

dimensions = 50  # speed up learning

l2_reg = 0.0
num_epochs = 100


# In[ ]:

# Load Data
train_ex, _ = get_data(datadir, task_number, test=False)
word_vectors = wordVectors(train_ex)
train_ex = examples_to_example_ind(word_vectors, train_ex)  # get examples in numerical format

# shuffle data
random.shuffle(train_ex)

# split the train into 70% train, 30% dev
train = train_ex[:int(.9 * len(train_ex))]
dev = train_ex[int(.1 * len(train_ex)):]


# In[ ]:

# get word_vectors (note using glove for now)
wv_matrix = word_vectors.get_wv_matrix(dimensions, glovedir)


# In[ ]:

# set up the basic model
averaging_model = averagingModel(wv_matrix, hidden_dim=128, num_classes=wv_matrix.shape[1])

support_idxs = T.ivector()
question_idxs = T.ivector()

answer_probs = averaging_model.get_answer_probs(support_idxs, question_idxs)
answer_pred = T.argmax(answer_probs)


# In[ ]:

# define the loss and cost function
answer = T.ivector()
loss = -T.mean(T.log(answer_probs)[T.arange(answer.shape[0]), answer])
cost = loss + l2_reg * averaging_model.l2


# In[ ]:

# optimization
updates = optimizers.Adagrad(cost, averaging_model.params)


# In[ ]:

# compile functions to train and evaluate the model
predict = theano.function(
           inputs = [support_idxs, question_idxs],
           outputs = answer_pred
    )

backprop = theano.function(
            inputs=[support_idxs, question_idxs, answer],
            outputs=cost,
            updates=updates)


# In[ ]:

### Training!!!
epoch = 0
train_acc_hist = []
dev_acc_hist = []
while epoch < num_epochs:
    print 'Epoch %d ' % epoch
    for example in verboserate(train):
        backprop(np.concatenate(example.sentences), example.question, example.answer)
    
    print 'Computing Train/Val Accuracy: '
    def compute_acc(dset):
        correct = 0.
        for example in verboserate(dset):
            if example.answer == predict(np.concatenate(example.sentences), example.question):
                correct += 1
        return correct / float(len(dset))
    
    train_acc, dev_acc  = compute_acc(train), compute_acc(dev)
    train_acc_hist.append(train_acc)
    dev_acc_hist.append(dev_acc)
    print 'Train Accuracy: %f , Validation Accuracy %f' % (train_acc, dev_acc)
    epoch += 1


# In[ ]:



