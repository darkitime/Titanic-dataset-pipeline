import pandas as pd

from configs.config_CL import config_CL

from src.data_prep.preprocessing import clean
from src.models.classic_models import get_classic_model, train_and_eval_oof

def run_cl():
    train_df = pd.read_csv(config_CL.paths.train)
    test_df = pd.read_csv(config_CL.paths.test)

    #Creates two versions of df: for OHE and (trees and boostings)
    train_ohe = clean(train_df, is_one_hot_encoding = True)
    test_ohe = clean(test_df, is_one_hot_encoding = True)
    y_train = train_ohe[config_CL.data.target_column].values
    X_train_ohe, X_test_ohe = train_ohe.drop(config_CL.data.target_column, axis=1).align(test_ohe, join='left', axis=1, fill_value=0)   #for same amount of columns


    train_tree = clean(train_df, is_one_hot_encoding=False)
    X_train_tree = train_tree.drop(config_CL.data.target_column, axis=1)
    X_test_tree = clean(test_df, is_one_hot_encoding=False)

    results = {}
    for model_name in config_CL.to_train:
        _, group = get_classic_model(model_name, config_CL)
        X_tr = X_train_ohe if group == 'linear_and_distance' else X_train_tree
        X_te = X_test_ohe if group == 'linear_and_distance' else X_test_tree
        
        res = train_and_eval_oof(model_name, config_CL, X_tr, y_train, X_te)
        results[model_name] = res
        print(f"{model_name}: OOF Accuracy = {res['oof_accuracy']:.4f}")

    return results