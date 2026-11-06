import pandas as pd
from configs.config_CL import config_CL
from configs.config_DNN import config_DNN

from src.utils import (
    set_seed,
    save_submission,
    build_leaderboard,
    save_classic_checkpoints,
    save_dnn_checkpoints
)

from src.engine.classic_runner import run_cl
from src.engine.dnn_runner import run_dnn

def main():
    print("=== 1. Fixing Random Seeds ===")
    set_seed()

    test_df = pd.read_csv(config_CL.paths.test)
    passenger_ids = test_df[config_CL.data.id_column].values
    y_test_true = pd.read_csv(config_CL.paths.targets)[config_CL.data.target_column].values

    print(f"\n=== 2. Running Classic ML Pipeline ({config_CL.training.n_spits}-Fold OFF) ===")
    all_results = run_cl()
    save_classic_checkpoints(all_results)

    print(f"\n=== 3. Running Deep Learning Pipeline ({config_DNN.data.n_spits}-Fold OFF) ===")
    dnn_res = run_dnn()
    all_results[config_DNN.model.name] = dnn_res
    save_dnn_checkpoints(dnn_res['models'])

    print("\n=== 4. Final Leaderboard ===")
    leaderboard = build_leaderboard(
        all_results = all_results,
        y_test_true=y_test_true        
    )
    print(leaderboard.to_string(index=False))

    print("\n=== 5. Saving predictions ===")

    for i, model in enumerate(leaderboard['Model']):
        model_preds = all_results[model]['test_preds']
        sub_path = save_submission(
            passenger_ids = passenger_ids,
            preds = model_preds,
            filename = f"submit_{i}_{model}.csv"
        )
    print(f"\n Models are saved to {sub_path}")


if __name__ == '__main__':
    main
