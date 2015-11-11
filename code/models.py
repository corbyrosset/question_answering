import theano
import layers
import theano.tensor as T

class Model(object):
    def get_answer_probs:
        pass

class averagingModel(object):
    '''
        Simple 1-hidden layer neural network
        Input: should be symbolic variables
        wv_matrix: should be an initialized numpy matrix
    '''
    def __init__(self, wv_matrix, hidden_dim, num_classes):
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

        self.params = self.embeddingLayer.params + self.fc1.params + self.fc2.params + self.linear_layer.params

        self.l2 = 0
        for param in self.params:
            self.l2 += (param ** 2).sum()

    def get_answer_probs(self, supporting_indices, question_indices):
        # simple averaging of the representations
        support = T.mean(self.embeddingLayer(supporting_indices), axis=1)
        question = T.mean(self.embeddingLayer(question_indices), axis=1)

        hidden_1 = self.fc1(T.concatenate([support, question]))
        hidden_2 = self.fc2(hidden_1)
        outputs = self.linear_layer(hidden_2)
        probs = layers.SoftMax(outputs)

        return probs

class embeddingModel(object):
    '''
    '''
    def __init__(self, wv_matrix, lstm_hidden_dim, nn_hidden_dim, num_classes):
        input_dim = wv_matrix.shape[0]
        self.embeddingLayer = layers.wordVectorLayer(wv_matrix)
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

        self.params = self.embeddingLayer.params + self.LSTMLayer.params + \
            self.fc1.params + self.fc2.params + self.linear_layer.params

        self.l2 = 0
        for param in self.params:
            self.l2 += (param ** 2).sum()


    def embed_support(self, supporting_indices):
        # matrix with appropriate columns (words)
        word_vec = self.embeddingLayer(supporting_indices)
        [hidden, cells] = theano.scan(
            fn=lambda word, h, prev_cell : self.LSTMLayer(word, h, prev_cell),
            sequence=word_vec.T, # by default scan iterates over rows
            outputs_info=self.LSTMLayer.h0, self.LSTMLayer.cell_0
        )
        # maybe try averaging as well as the last hidden layer
        return hidden[-1]


    def embed_question(self, question_indices):
        # uses same LSTM layer as embed_support, but maybe try a separate one
        word_vec = self.embeddingLayer(question_indices)
        [hidden, cells] = theano.scan(
            fn=lambda word, h, prev_cell : self.LSTMLayer(word, h, prev_cell),
            sequence=word_vec.T, # by default scan iterates over rows
            outputs_info=self.LSTMLayer.h0, self.LSTMLayer.cell_0
        )
        return hidden[-1]


    def get_answer_probs(self, supporting_indices, question_indices):
        support = embed_support(supporting_indices)
        question = embed_question(question_indices)

        hidden_1 = self.fc1(T.concatenate([support, question]))
        hidden_2 = self.fc2(hidden_1)
        outputs = self.linear_layer(hidden_2)
        probs = layers.SoftMax(outputs)

        return probs
