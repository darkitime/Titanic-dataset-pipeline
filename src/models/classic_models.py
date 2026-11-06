from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

MODEL_REGISTRY = {
    'LogisticRegression': LogisticRegression,
    'KNeighborsClassifier': KNeighborsClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'CatBoostClassifier': CatBoostClassifier,
    'XGBClassifier': XGBClassifier,
    'LGBMClassifier': LGBMClassifier,
}

def get_classic_model(model_name: str, cfg: DictConfig) -> Tuple[Any, str]:
    """
    Creates classic model sample by its name and parameters from config_CL.
    
    Return:
        (model_instance, model_group)
        where model_group — 'linear_and_distance' or 'tree_based'
        (it is known for the model whether or not to choose  OneHot + StandardScaler).
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not found. Available: {list(MODEL_REGISTRY.keys())}")

    # Looking parameters in cfg
    model_params = None
    model_group = None

    for group_name in ['linear_and_distance', 'tree_based']:
        group_cfg = cfg.models.get(group_name, {})
        if model_name in group_cfg:
            # OmegaConf.to_container(..., resolve=True) transforms DictConfig/ListConfig 
            # to  dict/list and puts all links
            model_params = OmegaConf.to_container(group_cfg[model_name], resolve=True)
            model_group = group_name
            break

    if model_params is None:
        raise KeyError(f"Hyperparameters for '{model_name}' are not found in cfg.models")

    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(**model_params), model_group


def train_and_eval_oof(
    model_name: str,
    cfg: DictConfig,
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_test_raw: pd.DataFrame
) -> Dict[str, Any]:
    """
    Trains on StratifiedKFold, computes OOF Accuracy
    and averages predictions for a test on all folds.
    Applies StandardScaler.
    """
    base_model, model_group = get_classic_model(model_name, cfg)
    is_linear = (model_group == 'linear_and_distance')
    
    skf = StratifiedKFold(
        n_splits=cfg.training.n_splits,
        shuffle=True,
        random_state=cfg.general.seed
    )

    oof_preds_proba = np.zeros(len(X_train_raw))
    test_preds_proba = np.zeros(len(X_test_raw))
    fold_models = []
    
    num_cols = list(cfg.data.num_features)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
        X_tr = X_train_raw.iloc[train_idx].copy()
        y_tr = y_train[train_idx]
        X_va = X_train_raw.iloc[val_idx].copy()
        y_va = y_train[val_idx]
        X_te = X_test_raw.copy()

        if is_linear:
            scaler = StandardScaler()
            X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
            X_va[num_cols] = scaler.transform(X_va[num_cols])
            X_te[num_cols] = scaler.transform(X_te[num_cols])

        model, _ = get_classic_model(model_name, cfg)
        model.fit(X_tr, y_tr)

        val_proba = model.predict_proba(X_va)[:, 1]
        oof_preds_proba[val_idx] = val_proba
        test_preds_proba += model.predict_proba(X_te)[:, 1] / cfg.training.n_splits
        
        fold_models.append(model)

    oof_preds_binary = (oof_preds_proba >= 0.5).astype(int)
    test_preds_binary = (test_preds_proba >= 0.5).astype(int)
    oof_acc = accuracy_score(y_train, oof_preds_binary)

    return {
        'model_name': model_name,
        'oof_accuracy': oof_acc,
        'oof_proba': oof_preds_proba,
        'test_proba': test_preds_proba,
        'test_preds': test_preds_binary,
        'models': fold_models
    }