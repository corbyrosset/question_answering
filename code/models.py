import theano
import layers
import cPickle as pickle
import theano.tensor as T


class Model(object):
    def get_answer_probs(self, supporting_indices, question_indices):
        raise NotImplementedError()

    def backprop(support_idxs, question_idxs, answer):
        raise NotImplementedError()

    def predict(support_idxs, question_idxs):
        raise NotImplementedError()

    def objective(support_idxs, question_idxs, answer):
        raise NotImplementedError()

    def load_params(self, path):
        raise NotImplementedError()

    def save_params(self, path):
        raise NotImplementedError()


class averagingModel(Model):
    '''
        Simple 1-hidden layer neural network
        Input: should be symbolic variables
        wv_matrix: should be an initialized numpy matrix
    '''
    def __init__(self, wv_matrix, hidden_dim, num_classes):
        print 'Initializing averaging model...'
        # just concatenate vector averages
        input_dim = 2 * wv_matrix.shape[0]

        # initialize layers
        self.embeddingLayer = layers.wordVectorLayer(wv_matrix)

        self.fc1 = layers.FullyConnected(input_dim=input_dim,
                                         output_dim=hidden_dim,
                                         activation='relu')

        self.fc2 = layers.FullyConnected(input_dim=hidden_dim,
                                         output_dim=hidden_dim,
                                         activation='relu')

        self.linear_layer = layers.FullyConnected(input_dim=hidden_dim,
                                                  output_dim=num_classes,
                                                  activation=None)

        self.layers = {'embeddingLayer':  self.embeddingLayer, 'fc1' : self.fc1, 
                        'fc2': self.fc2, 'linear': self.linear_layer}
        self.params = self.embeddingLayer.params + self.fc1.params + self.fc2.params + self.linear_layer.params


    def get_answer_probs(self, supporting_indices, question_indices):
        # simple averaging of the representations
        support = T.mean(self.embeddingLayer(supporting_indices), axis=1)
        question = T.mean(self.embeddingLayer(question_indices), axis=1)

        hidden_1 = self.fc1(T.concatenate([support, question]))
        hidden_2 = self.fc2(hidden_1)
        outputs = self.linear_layer(hidden_2)
        probs = layers.SoftMax(outputs)

        return probs

    def save_params(self, path):
        assert path is not None
        print 'Saving params to ', path
        params = {}
        for name, layer in self.layers.iteritems():
            params[name] = layer.get_params()
        pickle.dump(params, file(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self, path):
        assert path is not None
        print 'Restoring params from ', path
        params = pickle.load(file(path, 'r'))
        for name, layer in self.layers.iteritems():
            layer.set_params(params[name])


class attentionModel(object):
    '''
        TODO: write README
    '''
    def __init__(self, embeddings, lstm_hidden_dim, reverse=True):
        print 'Initializing attention model...'
        self.reverse = reverse
        input_dim = 2 * embeddings.shape[0]
        self.embeddingLayer = layers.embeddingLayer(embeddings)
        self.LSTMLayer = layers.LSTMLayer(input_dim, lstm_hidden_dim)
        self.linear_layer = layers.FullyConnected(input_dim=lstm_hidden_dim,
                                                  output_dim=2,
                                                  activation=None)
        self.layers = {'lstm': self.LSTMLayer, 'linear': self.linear_layer,
                       'embeddings': self.embeddingLayer}

        self.params = self.LSTMLayer.params + self.linear_layer.params + self.embeddingLayer.params

    def get_relevance_probs(self, support, mask, question_idxs):
        '''
            Support: matrix of supporting word indices
            Mask: Mask of in-sentence vs. zero-padding
            Question: vector of word indices
            Return sentences indices the model deems relevant
        '''
        question_vec = T.mean(self.embeddingLayer(question_idxs), axis=1)

        def step(s, m, h, prev_cell):
            sentence = T.mean(self.embeddingLayer(s) * m, axis=1)
            in_sentence = T.concatenate([question_vec, sentence])

            hidden_layer, next_cell = self.LSTMLayer(in_sentence, h, prev_cell)

            outputs = self.linear_layer(hidden_layer)
            prob = layers.SoftMax(outputs)
            return hidden_layer, next_cell, prob

        if self.reverse:
            support = support[::-1]  # iterate in reverse
            mask = mask[::-1]  # iterate in reverse

        [hidden, cells, probs], _ = theano.scan(
            fn=step,
            sequences=[support, mask],
            outputs_info=[self.LSTMLayer.h0, self.LSTMLayer.cell_0, None],
        )

        if self.reverse:
            return probs[:, 0, :][::-1]
        else:
            return probs[:, 0, :]

    def save_params(self, path):
        assert path is not None
        print 'Saving params to ', path
        params = {}
        for name, layer in self.layers.iteritems():
            params[name] = layer.get_params()
        pickle.dump(params, file(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self, path):
        assert path is not None
        print 'Restoring params from ', path
        params = pickle.load(file(path, 'r'))
        for name, layer in self.layers.iteritems():
            layer.set_params(params[name])


class embeddingModel(Model):
    '''
    '''
    def __init__(self, wv_matrix, lstm_hidden_dim, nn_hidden_dim, num_classes,
                 mean_pool=False):
        print 'Initializing embedding model...'
        self.mean_pool = mean_pool
        input_dim = wv_matrix.shape[0]
        self.embeddingLayer = layers.embeddingLayer(wv_matrix)
        self.LSTMLayer = layers.LSTMLayer(input_dim, lstm_hidden_dim)
        nn_input_dim = 2 * lstm_hidden_dim

        self.fc1 = layers.FullyConnected(input_dim=nn_input_dim,
                                         output_dim=nn_hidden_dim,
                                         activation='relu')

        self.fc2 = layers.FullyConnected(input_dim=nn_hidden_dim,
                                         output_dim=nn_hidden_dim,
                                         activation='relu')

        self.linear_layer = layers.FullyConnected(input_dim=nn_hidden_dim,
                                                  output_dim=num_classes,
                                                  activation=None)

        self.layers = {'embeddingLayer':  self.embeddingLayer, 'lstm': self.LSTMLayer,
                       'fc1': self.fc1, 'fc2': self.fc2, 'linear': self.linear_layer}

        self.params = self.embeddingLayer.params + self.LSTMLayer.params + \
            self.fc1.params + self.fc2.params + self.linear_layer.params

    def embed_support(self, supporting_indices):
        # matrix with appropriate columns (words)
        word_vec = self.embeddingLayer(supporting_indices)
        [hidden, cells], _ = theano.scan(
            fn=lambda word, h, prev_cell: self.LSTMLayer(word, h, prev_cell),
            sequences=word_vec.T,  # by default scan iterates over rows
            outputs_info=[self.LSTMLayer.h0, self.LSTMLayer.cell_0]
        )
        if self.mean_pool:
            return T.mean(hidden, axis=0)
        else:
            return hidden[-1]

    def embed_question(self, question_indices):
        # uses same LSTM layer as embed_support, but maybe try a separate one
        word_vec = self.embeddingLayer(question_indices)
        [hidden, cells], _ = theano.scan(
            fn=lambda word, h, prev_cell: self.LSTMLayer(word, h, prev_cell),
            sequences=word_vec.T,  # by default scan iterates over rows
            outputs_info=[self.LSTMLayer.h0, self.LSTMLayer.cell_0]
        )
        if self.mean_pool:
            return T.mean(hidden, axis=0)
        else:
            return hidden[-1]

    def get_answer_probs(self, support_sentences, mask, question_idxs):
        support_idxs = support_sentences[mask > 0].reshape((1, -1))
        support = self.embed_support(support_idxs)
        question = self.embed_question(question_idxs)

        hidden_1 = self.fc1(T.concatenate([support, question], axis=1))
        hidden_2 = self.fc2(hidden_1)
        outputs = self.linear_layer(hidden_2)
        probs = layers.SoftMax(outputs)

        return probs

    def save_params(self, path):
        assert path is not None
        print 'Saving params to ', path
        params = {}
        for name, layer in self.layers.iteritems():
            params[name] = layer.get_params()
        pickle.dump(params, file(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self, path):
        assert path is not None
        print 'Restoring params from ', path
        params = pickle.load(file(path, 'r'))
        for name, layer in self.layers.iteritems():
            layer.set_params(params[name])
