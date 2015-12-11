
# coding: utf-8

# In[18]:

import sys
sys.path.append('../')


# In[19]:

from data_import import *
from sklearn.linear_model import LogisticRegression
import numpy as np


# In[89]:

task_number = 18
datadir = '../../data/'
train_ex, test_ex = get_data(datadir, task_number, test=True)


# In[90]:

word_vectors = wordVectors(train_ex) #  a canonical word <-> idx mapping self.words_to_idx, self.idx_to_word
words_to_idx, idx_to_words = word_vectors.words_to_idx, word_vectors.idx_to_word
n = len(words_to_idx.keys())


# In[91]:

def get_design_matrix(dset):
    X = np.zeros((len(dset), 2 * n))
    for i, ex in enumerate(dset):
        for sentence in ex.sentences:
            for word in tokenize(sentence):
                X[i, words_to_idx[word]] += 1
        for word in tokenize(ex.question):
            X[i, n + words_to_idx[word]] += 1
    y = np.array([words_to_idx[tokenize(ex.answer)[0]] for ex in dset])
    return X, y


# In[92]:

X_train, y_train = get_design_matrix(train_ex)
X_test, y_test = get_design_matrix(test_ex)


# In[93]:

clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(X_train, y_train)
print 'Training Accuracy: ', clf.score(X_train, y_train)


# In[94]:

print 'Testing Accuracy: ', clf.score(X_test, y_test)


# In[ ]:




# In[ ]:



