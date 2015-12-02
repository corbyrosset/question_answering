basic_experiment = {
    # data and logging
    'tn': [1, 3],

    # HYPERPARAMETERS
    'lr': 1e-2,

    # all of the models
    'mt': 'sentenceEmbedding',  # one of |sentenceEmbedding| or |averaging|
    'hd': 128,
    'l2': 0.0,


    # specific to sentence embedding model
    'lhd': 128,
    'mp': 0,
}
