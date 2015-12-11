basic_experiment = {
    # data and logging ## every task except 8, 19
    'tn': [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20],

    # HYPERPARAMETERS
    'lr': [1e-2],

    # all of the models
    'mt': 'sentenceEmbedding',  # one of |sentenceEmbedding| or |averaging|
    'hd': 128,
    'l2': [1e-7],

    'attention': [0, 1],  # use attention model or not

    # specific to sentence embedding model
    'lhd': 128,
    'mp': [0, 1],

    # experiment specific
    'me': 70,
}
