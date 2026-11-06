import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from configs.config_DNN import config_DNN
from src.data_prep.preprocessing import clean

class TitanicDataset(Dataset):
    '''
    titanic dataset for pytorch

    X: np.ndarray
    y: np.ndarray
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_kfold_datasets(
    path_to_train: str = config_DNN.paths.train,
    path_to_test: str = config_DNN.paths.test,
    path_to_test_target: str = config_DNN.paths.targets,
    n_splits: int = config_DNN.data.n_splits,
    seed: int = config_DNN.general.seed,
    is_one_hot_encoding: bool = not config_DNN.model.params.use_embeddings
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Makes StratifiedKFold, normalize them in every fold
    and returns list of datasets.
    """
    train = pd.read_csv(path_to_train)
    test = pd.read_csv(path_to_test)
    y_test_df = pd.read_csv(path_to_test_target)

    train_proc = clean(train, is_one_hot_encoding=is_one_hot_encoding)
    test_proc = clean(test, is_one_hot_encoding=is_one_hot_encoding)

    target_col = config_DNN.data.target_column
    y = train_proc[target_col].values
    X = train_proc.drop(target_col, axis=1)
    y_test = y_test_df[target_col].values

    # Aligining columns
    X, test_proc = X.align(test_proc, join='left', axis=1, fill_value=0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    num_cols = list(config_DNN.data.num_features)

    folds_data = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_idx].copy()
        y_train = y[train_idx]
        X_val = X.iloc[val_idx].copy()
        y_val = y[val_idx]
        X_test_fold = test_proc.copy()

        #Scaler
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        X_test_fold[num_cols] = scaler.transform(X_test_fold[num_cols])

        folds_data.append({
            'fold': fold_idx,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'train_dataset': TitanicDataset(X_train.values, y_train),
            'val_dataset': TitanicDataset(X_val.values, y_val),
            'test_dataset': TitanicDataset(X_test_fold.values, y_test)
        })

    return folds_data, y, y_test

