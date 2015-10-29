import theano
import layers
import theano.tensor as T


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
