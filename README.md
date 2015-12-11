# Question Answering on the Facebook bAbI dataset

We trained various neural network models to answer tasks in the [Facebook bAbI dataset](https://research.facebook.com/researchers/1543934539189348). We use [Theano](http://deeplearning.net/software/theano/) to build our models, and [GloVe](http://nlp.stanford.edu/projects/glove/) pre-trained word vectors.

# Running our model

The most basic way to run our model is to run the ipython notebook at `/code/notebooks/qa_experiment.ipynb`, or equivalently, the script at `/code/notebooks/qa_experiment.py`.

The selection of model, and the hyperparameters, can be adjusted by changing the `hyperparams` data structure at the start of the file.

The file can also be directly called:

`python code/notebook/qa_experiment.py -task_num 1 -lr 1e-2 -model sentenceEmbedding -hd 128 -l2 0 -lstm_hd 128 -mp 0 -log logging_dir`

We run many tests at once by changing `configs.py` (in the root directory) to specify a range of hyperparameters to grid search over, and then running `launch_jobs.py`, which will create a shell script for each set of hyperparameters, and launch them on a Stanford farmshare machine. 

# Files
## /code/
### models.py
Uses the layers in `layers.py` to defines the models that we built and tested, including an averaging, embedding, and attention.

### layers.py 
Describes theano models for neural network layers. We implemented a fully connected, RNN, embedding, and LSTM layer.

### data_import.py
Defines the `example` class that holds sentences, questions, answers, and hints from the training set, as well as utilities to load the files by task number, and train/test

### experiment.py
Contains generalized utilities for training models, loading and saving trained models, and 

### diagonstics.py
Used to display examples and predictions for error analysis

### optimizers.py
We implement several training algorithms: SGD, RMSProp, Adagrad, Adam

## /code/notebooks
### Baseline-n-gram-classifier.py
Implements a simple n-gram model using logistic regression as a baseline

### qa_experiment.py
The script we use to run training/testing experiments.

## /data/

Data includes the directories for the bAbI dataset, as well as GloVe word vectors (data not in repository).

The data directory itself should contain the following files:

```
qa1_single-supporting-fact_test.txt
qa1_single-supporting-fact_train.txt
qa2_two-supporting-facts_test.txt
qa2_two-supporting-facts_train.txt
qa3_three-supporting-facts_test.txt
qa3_three-supporting-facts_train.txt
qa4_two-arg-relations_test.txt
qa4_two-arg-relations_train.txt
qa5_three-arg-relations_test.txt
qa5_three-arg-relations_train.txt
qa6_yes-no-questions_test.txt
qa6_yes-no-questions_train.txt
qa7_counting_test.txt
qa7_counting_train.txt
qa8_lists-sets_test.txt
qa8_lists-sets_train.txt
qa9_simple-negation_test.txt
qa9_simple-negation_train.txt
qa10_indefinite-knowledge_test.txt
qa10_indefinite-knowledge_train.txt
qa11_basic-coreference_test.txt
qa11_basic-coreference_train.txt
qa12_conjunction_test.txt
qa12_conjunction_train.txt
qa13_compound-coreference_test.txt
qa13_compound-coreference_train.txt
qa14_time-reasoning_test.txt
qa14_time-reasoning_train.txt
qa15_basic-deduction_test.txt
qa15_basic-deduction_train.txt
qa16_basic-induction_test.txt
qa16_basic-induction_train.txt
qa17_positional-reasoning_test.txt
qa17_positional-reasoning_train.txt
qa18_size-reasoning_test.txt
qa18_size-reasoning_train.txt
qa19_path-finding_test.txt
qa19_path-finding_train.txt
qa20_agents-motivations_test.txt
qa20_agents-motivations_train.txt
```

## /data/glove/

Contains glove pre-trained word vectors

```
glove.6B.50d.txt
glove.6B.100d.txt
glove.6B.200d.txt
glove.6B.300d.txt
```