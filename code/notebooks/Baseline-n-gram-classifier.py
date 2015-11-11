
# coding: utf-8

# In[ ]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[ ]:

import sys
sys.path.append('../')


# In[ ]:

import data_import
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import numpy as np


# In[ ]:

# Input: examples, an array of examples, each a dict containing:
# example['s']: an array of sentences corresponding to statements 
# potentially helpful in solving the problem
# example['q']: the question asked
# example['a']: the answer of the question
#
# Output: arrays of strings corresponding to inputs for a bag of 
# words feature extractor. X are statements, Y is answers, and Q are questions.
def make_strings(examples):
    X = []
    Y = []
    Q = []
    for example in examples:
        X.append(" ".join(example.sentences))
        Q.append(example.question)
        Y.append(example.answer)
    print "Examples loaded: "
    print "\t X[1] = %s\n\t Y[1] = %s\n\t Q[1] = %s" % (X[1], Y[1], Q[1]) 
    return (X,Y,Q)

# Runs a given bag of words feature extractor feat_ex on inputs X, Y and Q
# from make_strings. Returns (T, Q), where T is a stack of the supporting
# statement (X) and question (Q) vectors, and Y is a matrix of answer vectors.
def get_features(feat_ex, X, Y, Q):
    X_t = feat_ex.transform(X)
    Q_t = feat_ex.transform(Q)
    Y_t = np.argmax(feat_ex.transform(Y).todense(), axis=1)
    return (hstack([X_t, Q_t]), Y_t)


# In[ ]:

datadir = "../../data/"
# tasknums: 
# 1: single supporting fact
# 5: 3 arg relations
# 7: counting
# 17: positional reasoning
# 19: path finding
tasknum = 19
(train_examples, test_examples) = data_import.get_data(datadir, tasknum)

# Create ngram vectorizer and string inputs
vectorizer = CountVectorizer(ngram_range=(1, 4),min_df=1)
(X_tr, Y_tr, Q_tr) = make_strings(train_examples)
(X_te, Y_te, Q_te) = make_strings(test_examples)

# Want the feature space to include the words in the test examples too
feature_extractor = vectorizer.fit(X_tr+X_te+Y_tr+Y_te+Q_tr+Q_te)


# In[ ]:

# Obtain featurized vector stacks 
(t_train, y_train) = get_features(feature_extractor, X_tr, Y_tr, Q_tr)
print t_train.shape


# In[ ]:

clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(t_train, y_train)
print "Train score: %f" % clf.score(t_train, y_train)

(t_test, y_test) = get_features(feature_extractor, X_te, Y_te, Q_te)
print "Test score: %f" % clf.score(t_test, y_test)

