import os
import util
import sys
import re
import numpy as np


class example_ind(object):
    def __init__(self, sentences, mask, question, answer, hints):
        '''
            Object which contains relevant information for inputting into the
            model, but whose elements are integer indicies into a word vector
            matrix.
        '''
        self.sentences = sentences
        self.mask = mask
        self.question = question
        self.answer = answer
        self.hints = hints

    def __repr__(self):
        return ("Training example: \n\t Info: %s \n\t Question: %s \n\t Answer: %s \n\t Hint: %s \n"
                % (self.sentences, self.question, self.answer, self.hints))


class example(object):
    def __init__(self, sentences, question, answer, hints):
        '''
            Object which contains relevant information for inputting into the
            model.
        '''
        self.sentences = sentences
        self.question = question
        self.answer = answer
        self.hints = hints

    def __repr__(self):
        return ("Training example: \n\t Info: %s \n\t Question: %s \n\t Answer: %s \n\t Hint: %s \n"
                % (self.sentences, self.question, self.answer, self.hints))


class wordVectors(object):
    def __init__(self, dataset):
        self.words_to_idx, self.idx_to_word = self._map_words_to_idx(dataset)

    def _map_words_to_idx(self, dataset):
        tokens = []
        for example in dataset:
            # add all supporting sentence words
            for sentence in example.sentences:
                tokens += tokenize(sentence)

            tokens += tokenize(example.question)
            tokens += tokenize(example.answer)

        tokens = set(tokens)

        # loop over the tokens and establish a canonical word <-> idx mapping
        words_to_idx = {}
        idx_to_words = {}
        counter = 0
        for token in tokens:
            token = token.lower()
            if token not in words_to_idx:
                words_to_idx[token] = counter
                idx_to_words[counter] = token
                counter += 1

        return words_to_idx, idx_to_words

    def get_wv_matrix(self, dimension, glove_dir=None):
        r = 0.001
        self.wv_matrix = np.random.rand(dimension, len(self.words_to_idx)) * 2 * r - r  # TODO: pick initialization carefully
        if glove_dir is not None:
            pretrained = load_glove_vectors(glove_dir, dimension)

            for word in self.words_to_idx:
                if word in pretrained:
                    self.wv_matrix[:, self.words_to_idx[word]] = pretrained[word].ravel()

        return self.wv_matrix


# wordVectors is an instance of the output of wordVectors
def examples_to_example_ind(wordVectors, examples):
    outputs = []
    for example in examples:
        new_sents = []
        for sentence in example.sentences:
            new_sents.append(np.array([wordVectors.words_to_idx[word] for word in tokenize(sentence)], dtype='int32'))

        sentences = np.zeros((len(new_sents), max(len(s) for s in new_sents)), dtype='int32')
        mask = np.zeros_like(sentences,  dtype='int32')
        for i, sent in enumerate(new_sents):
            sentences[i, :len(sent)] = sent
            mask[i, :len(sent)] = 1

        new_quest = np.array([wordVectors.words_to_idx[word] for word in tokenize(example.question)], dtype='int32')
        new_ans = np.array([wordVectors.words_to_idx[word] for word in tokenize(example.answer)], dtype='int32')

        new_hints = np.zeros((sentences.shape[0], ), dtype='int32')
        new_hints[example.hints] = 1
        outputs.append(example_ind(sentences, mask, new_quest, new_ans, new_hints))

    return outputs


# Some of the answers aren't words eg: (n,s):
# This replaces it with "north south"
def fix_directions(examples):
    directions = {'n': 'north', 'e': 'east', 's': 'south', 'w': 'west'}
    for example in examples:
        dirs = example.answer.split(',')
        newdirs = [directions[d] for d in dirs]
        example.answer = " ".join(newdirs)


# Each Set consists of several lines (eg:)
# 1 Mary is in the park.
# 2 Julie travelled to the office.
# 3 Is Julie in the kitchen? 	no	2
# 4 Julie went back to the school.
# 5 Mary went to the office.
# 6 Is Mary in the office? 	yes	5
# 7 Fred is in the cinema.
# 8 Julie is either in the kitchen or the bedroom.
# 9 Is Julie in the bedroom? 	maybe	8
@util.memoize
def file_to_examples(file):
    f = open(file, "r")
    lines = f.readlines()
    information = []
    questans = []

    # Want tuples (information, information ..., information, answer)
    for line in lines:
        split = line.strip().split('\t')
        linesplit = split[0].split(' ')
        linenum = int(linesplit[0])
        sentence = " ".join(linesplit[1:]).strip()

        # Signals start of new set
        if linenum == 1:
            information = []
            hint_to_arr_idx = {}
            diff = 1

        # For each question, add as the information all of the previous
        # sentences that could have been relevent.
        if sentence[-1] == "?":
            question = sentence
            answer = split[1]
            hints = map(int, split[2].split(' '))

            hint_idxs = [hint_to_arr_idx[i] for i in hints]

            questans.append(example(sentences=list(information),
                                    answer=answer,
                                    question=question,
                                    hints=hint_idxs))
            diff += 1
        else:
            information.append(sentence)
            hint_to_arr_idx[linenum] = linenum - diff

    return questans

@util.memoize
def file_to_relevant_examples(file):
    f = open(file, "r")
    lines = f.readlines()
    information = []
    questans = []
    all_info = []

    # Want tuples (information, information ..., information, answer)
    for line in lines:
        split = line.strip().split('\t')
        linesplit = split[0].split(' ')
        linenum = int(linesplit[0])
        sentence = " ".join(linesplit[1:]).strip()

        # Signals start of new set
        if linenum == 1:
            information = []
            all_info = []

        all_info.append(sentence)
        # For each question, add as the information all of the previous
        # sentences that could have been relevent.
        if sentence[-1] == "?":
            question = sentence
            answer = split[1]
            hint = split[2]

            relevant = [all_info[i-1] for i in map(int, hint.split(' '))]
            questans.append(example(sentences=list(relevant),
                                    answer=answer,
                                    question=question,
                                    hints=hint))
        else:
            information.append(sentence)

    return questans

def tokenize(sentence):
    return [token.lower() for token in re.findall(r"[\w']+|[.,!?;]", sentence)]

def get_file_path(datadir, tasknum, test=False):
    fnames = {
        1:"single-supporting-fact",
        2:"two-supporting-facts",
        3:"three-supporting-facts",
        4:"two-arg-relations",
        5:"three-arg-relations",
        6:"yes-no-questions",
        7:"counting",
        8:"lists-sets",
        9:"simple-negation",
        10:"indefinite-knowledge",
        11:"basic-coreference",
        12:"conjunction",
        13:"compound-coreference",
        14:"time-reasoning",
        15:"basic-deduction",
        16:"basic-induction",
        17:"positional-reasoning",
        18:"size-reasoning",
        19:"path-finding",
        20:"agents-motivations"
    }
    if(tasknum < 1 or tasknum > 20):
        raise NotImplementedError("Task %d is not valid" % tasknum)

    traintest = "test" if test else "train"
    fname = ("qa%d_%s_%s.txt") % (tasknum, fnames[tasknum], traintest)
    return datadir + fname

def get_relevant_data(datadir, tasknum, test=False):
    train_examples = file_to_relevant_examples(get_file_path(datadir, tasknum, False))
    test_examples = file_to_relevant_examples(get_file_path(datadir, tasknum, True))
    
    if tasknum == 19:
        # hack to replace directions with their actual words
        fix_directions(train_examples)
        fix_directions(test_examples)
    if test:
        print 'WARNING: Loading TEST SET'
        return train_examples, test_examples
    else:
        return train_examples, None

def get_data(datadir, tasknum, test=False):
    train_examples = file_to_examples(get_file_path(datadir, tasknum, False))
    test_examples = file_to_examples(get_file_path(datadir, tasknum, True))
    
    if tasknum == 19:
        # hack to replace directions with their actual words
        fix_directions(train_examples)
        fix_directions(test_examples)
    if test:
        print 'WARNING: Loading TEST SET'
        return train_examples, test_examples
    else:
        return train_examples, None



@util.memoize
def load_glove_vectors(datadir, dimension):
    if dimension not in [50, 100, 200, 300]:
        raise NotImplementedError('No Glove Vectors with dimension %d' % dimension)
    file_name = 'glove.6B.%dd.txt' % dimension
    file_path = os.path.join(datadir, file_name)
    wvecs = {}
    print 'loading glove vectors'
    sys.stdout.flush()
    with open(file_path) as f_glove:
        for i, line in enumerate(f_glove):
            elems = line.split()
            word = elems[0]
            vec = np.array([float(x) for x in elems[1:]]).reshape(-1, 1)
            wvecs[word] = vec
            if i % 20000 == 0:
                print i
    print 'done'
    return wvecs
