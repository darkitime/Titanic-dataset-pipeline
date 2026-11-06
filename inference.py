import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from configs.config_CL import config_CL
from configs.config_DNN import config_DNN
from src.data_prep.preprocessing import clean
from src.data_prep.dataset import get_kfold_datasets
from src.models.tabular_nn import TitanicModel
from src.engine.dnn_runner import predict_proba_dataloader
from src.utils import save_submission


def inference_classic_model(model_name: str, X_test: pd.DataFrame) -> np.ndarray:
    """Загружает сохраненные фолды классической модели и усредняет предсказания."""
    filepath = os.path.join(config_CL.paths.weights_dir, f"{model_name}_folds.joblib")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weights not found: {filepath}. Run main.py first.")

    fold_models = joblib.load(filepath)
    preds_proba = np.zeros(len(X_test))
    for model in fold_models:
        preds_proba += model.predict_proba(X_test)[:, 1] / len(fold_models)
    return preds_proba


def inference_dnn() -> np.ndarray:
    """Загружает сохраненные веса фолдов нейросети и делает предсказание на тесте."""
    folds_data, _, _ = get_kfold_datasets()
    device = config_DNN.general.device

    if config_DNN.model.params.use_embeddings:
        emb_dims = [(2, 2), (3, 2), (5, 3)]
        num_cont = 5
    else:
        emb_dims = None
        num_cont = folds_data[0]['train_dataset'].X.shape[1]

    test_proba = np.zeros(len(folds_data[0]['test_dataset']))

    for fold_info in folds_data:
        fold_idx = fold_info['fold']
        weight_path = os.path.join(config_DNN.paths.checkpoints_dir, f"dnn_fold_{fold_idx}.pt")
        
        model = TitanicModel(num_continuous_features=num_cont, emb_dims=emb_dims)
        model.load_state_dict(torch.load(weight_path, map_location=device))

        test_loader = DataLoader(
            fold_info['test_dataset'],
            batch_size=config_DNN.data.test_batch_size,
            shuffle=False
        )
        test_proba += predict_proba_dataloader(model, test_loader, device) / len(folds_data)

    return test_proba


def run_inference():
    test_df = pd.read_csv(config_CL.paths.test)
    passenger_ids = test_df[config_CL.data.id_column].values
    y_test_true = pd.read_csv(config_CL.paths.targets)[config_CL.data.target_column].values

    # Данные для древовидных моделей (CatBoost, LGBM, XGB)
    X_test_tree = clean(test_df, is_one_hot_encoding=False)

    # 1. Инференс бустингов
    cb_proba = inference_classic_model('CatBoostClassifier', X_test_tree)
    lgbm_proba = inference_classic_model('LGBMClassifier', X_test_tree)
    xgb_proba = inference_classic_model('XGBClassifier', X_test_tree)

    # 2. Инференс DNN
    dnn_proba = inference_dnn()

    # 3. Ансамбль (Blending) всех 4 подходов
    blend_proba = (cb_proba + lgbm_proba + xgb_proba + dnn_proba) / 4.0
    blend_preds = (blend_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test_true, blend_preds)
    sub_path = save_submission(passenger_ids, blend_preds, config_CL.paths.submission_dir, 'ensemble_submission.csv')
    print(f"Ensemble (CatBoost + LGBM + XGB + DNN) Test Accuracy: {acc:.4f}")
    print(f"Saved ensemble submission to: {sub_path}")


if __name__ == '__main__':
    run_inference()