basic_experiment = {
    # data and logging
    'tn': [1, 3],

    # HYPERPARAMETERS
    'lr': [1e-2, 1e-4],

    # all of the models
    'm': 'sentenceEmbedding',  # one of |sentenceEmbedding| or |averaging|
    'hd': 128,
    'l2': 0.0,


    # specific to sentence embedding model
    'lhd': 128,
    'mp': 0,
}
