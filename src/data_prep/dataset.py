import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from configs.config_DNN import config_DNN
from src.utils import clean

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


def get_data(path_to_train =config_DNN.paths.train, path_to_test = config_DNN.paths.test, path_to_test_target=config_DNN.paths.target,
is_one_hot_encoding = True):
    '''
    Returns train, val, test datasets for NN
    '''

    train = pd.read_csv(path_to_train)
    test = pd.read_csv(path_to_test)
    y_test = pd.read_csv(path_to_test_target)

    train_proc = clean(train, is_one_hot_encoding = is_one_hot_encoding)
    test_proc = clean(test, is_one_hot_encoding = is_one_hot_encoding)

    y = train_proc[config_DNN.data.target_column].values
    X = train_proc.drop(config_DNN.data.target_column, axis=1)
    y_test = y_test[config_DNN.data.target_column].values

    #allign
    X, test_proc = X.align(test_proc, join = 'left', axis = 1, fill_value = 0)

    #train/val
    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
    
    #Raise SettingWithCopyWarning
    X_train = X_train.copy()
    X_val = X_val.copy()
    test_proc = test_proc.copy()  
    
    #Normalizing
    num_cols = config_DNN.data.num_features
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    test_proc[num_cols] = scaler.transform(test_proc[num_cols])
    
    train_dataset = TitanicDataset(X_train.values, y_train)
    val_dataset = TitanicDataset(X_val.values, y_val)
    test_dataset = TitanicDataset(test_proc.values, y_test)

    return train_dataset, val_dataset, test_dataset

