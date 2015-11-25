basic_experiment = {
    # data and logging
    'task_number': 1,  # {1, 3, 5, 17, 19}

    # HYPERPARAMETERS
    'base_lr': 1e-2,

    # all of the models
    'model_type': 'sentenceEmbedding',  # one of |sentenceEmbedding| or |averaging|
    'hidden_dim': 128,
    'l2_reg': 0.0,


    # specific to sentence embedding model
    'lstm_hidden_dim': 128,
    'mean_pool': 0,

    'logging_path': 'logging_dir',
}
