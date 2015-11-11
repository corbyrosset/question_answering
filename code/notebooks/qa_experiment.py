
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

# Constants
datadir = '../../data/'
glovedir = '../../data/glove'
SEED = 999
# np.random.seed(SEED)

task_number = 1 # simple task for now


# In[ ]:

# Turn on for debugging
DEBUG = False
if DEBUG:
    theano.config.optimizer='fast_compile'
    theano.config.exception_verbosity='high'


# In[ ]:

# HYPERPARAMETERS
wv_dimensions = 50  # speed up learning by using the smallest GloVe dimension

# training
num_epochs = 100
base_lr = 1e-2

## all of the models
model_type = 'sentenceEmbedding'  # one of |sentenceEmbedding| or |averaging|
hidden_dim = 128
l2_reg = 0.0


# specific to sentence embedding model
lstm_hidden_dim = 128
mean_pool = False


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

# get word_vectors (Using glove for now)
wv_matrix = word_vectors.get_wv_matrix(wv_dimensions, glovedir)


# In[ ]:

# set up the basic model
if model_type == 'averaging':
    model = averagingModel(wv_matrix, hidden_dim=hidden_dim, num_classes=wv_matrix.shape[1])
elif model_type == 'sentenceEmbedding':
    model = embeddingModel(wv_matrix, lstm_hidden_dim=lstm_hidden_dim, nn_hidden_dim=hidden_dim,
                            num_classes=wv_matrix.shape[1], mean_pool=mean_pool)


# In[ ]:

# generate answer probabilities and predictions
support_idxs = T.ivector()
question_idxs = T.ivector()

answer_probs = model.get_answer_probs(support_idxs, question_idxs)
answer_pred = T.argmax(answer_probs)


# In[ ]:

# define the loss and cost function
answer = T.ivector()
loss = -T.mean(T.log(answer_probs)[T.arange(answer.shape[0]), answer])
cost = loss + l2_reg * model.l2


# In[ ]:

# optimization
updates = optimizers.Adagrad(cost, model.params, base_lr=base_lr)


# In[ ]:

# compile functions to train and evaluate the model
print 'Compiling predict function'
predict = theano.function(
           inputs = [support_idxs, question_idxs],
           outputs = answer_pred
        )

print 'Compiling backprop function'
backprop = theano.function(
            inputs=[support_idxs, question_idxs, answer],
            outputs=[loss, answer_probs],
            updates=updates)


# In[ ]:

## Training!!!
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

## Plot learning curves
import matplotlib.pylab as plt

plt.plot(train_acc_hist)
plt.plot(dev_acc_hist)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend(['Train', 'Dev'])
plt.show()

