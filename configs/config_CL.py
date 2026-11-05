from omegaconf import OmegaConf

_config_cl = {
    'general': {
        'seed': 42,
        'mode': 'train',  # 'train', 'inference', 'evaluate_all'
        'metric': 'accuracy',
    },
    'paths': {
        'train': 'data/train.csv',
        'test': 'data/test.csv',
        'targets': 'data/real_test_submission.csv', 
        'output_dir': './outputs/classic_ml',
        'submission_dir': './submissions',
        'weights_dir': './checkpoints/classic_ml',
        'log_name': 'classic_ml.log',
    },
    'data': {
        'id_column': 'PassengerId',
        'target_column': 'Survived',
        'drop_columns': ['Name', 'Ticket', 'Cabin', 'PassengerId'],
        'cat_features': ['Sex', 'Embarked', 'Initial'],
        'num_features': ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
    },
    'preprocessing': {
        'extract_initial': True,
        'create_family_size': True,
        'fill_age_by_initial': True,
    },
    'training': {
        'n_splits': 5,
        'val_size': 0.2,
        'use_oof_stacking': True,
    },
    'models': {
        # Requirements: (OneHotEncoding + StandardScaler) for linear models
        'linear_and_distance': {
            'LogisticRegression': {
                'C': 1.0,
                'max_iter': 500,
                'penalty': 'l1',
                'solver': 'saga',
                'random_state': '${general.seed}'
            },
            'KNeighborsClassifier': {
                'metric': 'minkowski',
                'n_neighbors': 7,
                'weights': 'uniform'
            },
        },
        # Trees and boostings
        'tree_based': {
            'DecisionTreeClassifier': {
                'criterion': 'entropy',
                'max_depth': 5,
                'min_samples_leaf': 2,
                'min_samples_split': 2,
                'random_state': '${general.seed}'
            },
            'RandomForestClassifier': {
                'criterion': 'entropy',
                'max_depth': 8,
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'max_features': 'log2',
                'n_estimators': 300,
                'random_state': '${general.seed}'
            },
            'CatBoostClassifier': {
                'depth': 5,
                'iterations': 200,
                'l2_leaf_reg': 10,
                'learning_rate': 0.1,
                'random_seed': '${general.seed}',
                'subsample': 0.9,
                'cat_features': '${data.cat_features}',
                'verbose': 0
            },
            'XGBClassifier': {
                'tree_method': 'hist',
                'enable_categorical': True,
                'verbosity': 0,
                'colsample_bytree': 0.7,
                'gamma': 0.01,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 0.5,
                'n_estimators': 100,
                'random_state': '${general.seed}',
                'reg_lambda': 1,
                'subsample': 0.9
            },
            'LGBMClassifier': {
                'verbose': -1,
                'colsample_bytree': 0.9,
                'learning_rate': 0.1,
                'max_depth': 5,
                'n_estimators': 100,
                'num_leaves': 15,
                'random_state': '${general.seed}',
                'reg_lambda': 1,
                'subsample': 0.9
            }
        }
    },
    'to_train': [
        'LogisticRegression',
        'KNeighborsClassifier'
        'DecisionTreeClassifier'
        'RandomForestClassifier',
        'CatBoostClassifier',
        'XGBClassifier',
        'LGBMClassifier'
    ],
    'to_inference': [
        'catboost_oof',
        'lgbm_oof',
        'xgboost_oof'
    ]
}

config_CL = OmegaConf.create(_config_cl)