from omegaconf import OmegaConf
import torch.nn as nn

_config_dnn = {
    'general': {
        'mode': 'train',  # 'train', 'inference'
        'experiment_name': 'tabular_mlp_v1',
        'seed': 42,
        'device': 'cpu',  # 'cpu' or 'cuda',
        'verbose': False
    },
    'paths': {
        'train': 'data/train.csv',
        'test': 'data/test.csv',
        'targets': 'data/real_test_submission.csv',
        'checkpoints_dir': './checkpoints/dnn',
        'submission_dir': './submissions',
        'best_weights': './checkpoints/dnn/best_model.pt',
        'log_name': 'dnn_train.log',
    },
    'data': {
        'id_column': 'PassengerId',
        'target_column': 'Survived',
        'cat_features': ['Sex', 'Embarked', 'Initial'],
        'num_features': ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
        'n_splits': 5,
        'val_size': 0.2,
        'batch_size': 32,
        'train_batch_size': '${data.batch_size}',
        'val_batch_size': 64,
        'test_batch_size': 64,
        'num_workers': 0,
    },
    'model': {
        'name': 'TabularMLP',
        'params': {
            'use_embeddings': True,
            'hidden_dims': 64,
            'dropout': 0.0,
            'use_batchnorm': False,
            'output_dim': 1,
            'num_blocks': 2,
            'activation_cls': 'LeakyReLU'
        }
    },
    'training': {
        'num_epochs': 50,
        'lr': 1e-3,
        'weight_decay': 0.0,
        'early_stopping_patience': 5,
        'gradient_clipping': True,
        'clip_value': 1.0,
    },
    'optimizer': {
        'name': 'Adam',
        'params': {
            'lr': '${training.lr}',
            'weight_decay': '${training.weight_decay}',
        }
    },
    'scheduler': {
        'name': 'ReduceLROnPlateau',
        'interval': 'epoch',
        'params': {
            'mode': 'min',
            'factor': 0.1,
            'patience': 5
        }
    },
    'earlystopping':{
        'mode': 'min',
        'patience': 5,
        'threshold': 1e-4,
        'threshold_mode':'rel'
    },
    'loss': {
        'name': 'BCEWithLogitsLoss',
    },
    'metric': {
        'name': 'accuracy',
        'threshold': 0.5,
    },
    'logging': {
        'use_wandb': False,
        'wandb_project': 'titanic_pipeline',
    }
}

config_DNN = OmegaConf.create(_config_dnn)