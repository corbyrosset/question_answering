import numpy as np


def error_analysis(dset, model, word_vectors):
    def idx_to_words(matrix, mask):
        rows, cols = matrix.shape
        for row in xrange(rows):
            s = [word_vectors.idx_to_word[matrix[row, col]] for col in xrange(cols) if mask[row, col] == 1]
            print ' '.join(s)

    for ex in dset:
        answer_pred, est_rel_idxs = model.diagnostic(ex.sentences, ex.mask, ex.question)

        print 'Supporting Facts: '
        idx_to_words(ex.sentences, ex.mask)
        print
        print 'Question: ',
        idx_to_words(ex.question.reshape(1, -1), np.ones((1, len(ex.question))))
        print 'True Answer: ', word_vectors.idx_to_word[ex.answer[0]]
        print 'Predicted Answer: ', word_vectors.idx_to_word[int(answer_pred)]
        print 'Predicted Relevant Facts: ',
        idx_to_words(ex.sentences[est_rel_idxs, :], ex.mask[est_rel_idxs, :])
        print
