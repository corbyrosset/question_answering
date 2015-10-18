
# coding: utf-8

# In[ ]:

import DataImport
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import numpy as np


# In[ ]:

train_examples = DataImport.getdata("qa1_single-supporting-fact_train.txt")
test_examples = DataImport.getdata("qa1_single-supporting-fact_test.txt")


# In[ ]:

vectorizer = CountVectorizer(ngram_range=(1, 3),min_df=1)


# In[ ]:

def vectorize(examples):
    X = []
    Y = []
    Q = []
    for example in examples:
        X.append(" ".join(example['s']))
        Q.append(example['q'])
        Y.append(example['a'])
    return (X,Y,Q)


# In[ ]:

(X_tr, Y_tr, Q_tr) = vectorize(train_examples)
(X_te, Y_te, Q_te) = vectorize(test_examples)
feature_extractor = vectorizer.fit(X_tr+X_te+Y_tr+Y_te+Q_tr+Q_te)


# In[ ]:

def get_features(feat_ex, X, Y, Q):
    X_t = feat_ex.transform(X)
    Q_t = feat_ex.transform(Q)
    Y_t = np.argmax(feat_ex.transform(Y).todense(), axis=1)
    return (hstack([X_t, Q_t]), Y_t)


# In[ ]:

clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
(t_train, y_train) = get_features(feature_extractor, X_tr, Y_tr, Q_tr)
print t_train.shape


# In[ ]:

clf.fit(t_train, y_train)
print "Train score: %f" % clf.score(t_train, y_train)

(t_test, y_test) = get_features(feature_extractor, X_te, Y_te, Q_te)
print "Test score: %f" % clf.score(t_test, y_test)

