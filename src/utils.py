import random
import os
import torch
import joblib
import numpy as np
import pandas as pd
from configs.config_CL import config_CL
from configs.config_DNN import config_DNN
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score

def set_seed(seed: int = config_CL.general.seed):
    '''
    Setting seed for determination
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_submission(
    passenger_ids: np.ndarray,
    preds: np.ndarray,
    filename: str,
    output_dir: str = config_CL.paths.submission_dir,
) -> str:
    '''
    Saving predictions to CSV
    '''
    os.makedirs(output_dir, exist_ok=True)
    sub = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': preds.astype(int)
    })

    filepath = os.path.join(output_dir,filename)
    sub.to_csv(filepath,index=False)
    return filepath

def build_leaderboard(
    all_results: Dict[str, Dict[str, Any]],
    y_test_true: np.ndarray = None,
    output_dir: str = config_CL.paths.submission_dir
) -> pd.DataFrame:
    """
    Making table (OOF Accuracy + Test Accuracy)
    for all classic and DNN, sorting and saving to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for model_name, res in all_results.items():
        row = {
            'Model': model_name,
            'OOF_Accuracy': round(res['oof_accuracy'], 4),
        }
        if y_test_true is not None and 'test_preds' in res:
            test_acc = accuracy_score(y_test_true, res['test_preds'])
            row['Test_Accuracy'] = round(test_acc, 4)
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    sort_col = 'Test_Accuracy' if 'Test_Accuracy' in leaderboard.columns else 'OOF_Accuracy'
    leaderboard = leaderboard.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    leaderboard_path = os.path.join(output_dir, 'leaderboard.csv')
    leaderboard.to_csv(leaderboard_path, index=False)
    return leaderboard


def save_classic_checkpoints(results: Dict[str, Dict[str, Any]], weights_dir: str = config_CL.paths.weights_dir) -> None:
    """Saving asambles for all folds for classic models."""
    os.makedirs(weights_dir, exist_ok=True)
    for model_name, res in results.items():
        filepath = os.path.join(weights_dir, f"{model_name}_folds.joblib")
        joblib.dump(res['models'], filepath)


def save_dnn_checkpoints(models: List[torch.nn.Module], checkpoints_dir: str = config_DNN.paths.weights_dir) -> None:
    """Saving state_dict for every DNN fold."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    for fold_idx, model in enumerate(models):
        filepath = os.path.join(checkpoints_dir, f"dnn_fold_{fold_idx}.pt")
        torch.save(model.state_dict(), filepath)