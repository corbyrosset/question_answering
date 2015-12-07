basic_experiment = {
    # data and logging
    'tn': [1],

    # HYPERPARAMETERS
    'lr': [1e-2, 3e-2],

    # all of the models
    'mt': 'sentenceEmbedding',  # one of |sentenceEmbedding| or |averaging|
    'hd': 128,
    'l2': [1e-5, 1e-4, 5e-3],


    # specific to sentence embedding model
    'lhd': 128,
    'mp': [0, 1],

    # experiment specific
    'me': 70, 
}
