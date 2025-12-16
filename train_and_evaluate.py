import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.sparse
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, ndcg_score
from typing import Dict, Tuple, List, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate search relevance models.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save models and results")
    parser.add_argument("--n_estimators", type=int, default=200, help="Number of trees for Random Forest")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    return parser.parse_args()

def load_data(data_dir: str) -> Dict[str, Any]:
    print(f"Loading data from {data_dir}...")
    data = {}
    splits = ['train', 'valid', 'test']

    required_files = []
    for split in splits:
        required_files.extend([
            f"X_{split}_text.npz",
            f"y_{split}.npy",
            f"questions_{split}.csv",
            f"relevance_score_{split}.npy"
        ])

    missing = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        print(f"Error: Missing files in data_dir: {missing}")
        sys.exit(1)

    for split in splits:
        data[f"X_{split}"] = scipy.sparse.load_npz(os.path.join(data_dir, f"X_{split}_text.npz"))
        data[f"y_{split}"] = np.load(os.path.join(data_dir, f"y_{split}.npy"))
        data[f"scores_{split}"] = np.load(os.path.join(data_dir, f"relevance_score_{split}.npy"))

        q_df = pd.read_csv(os.path.join(data_dir, f"questions_{split}.csv"))
        data[f"questions_{split}"] = q_df['question'].values

    return data

def train_models(
    X_train, y_train,
    n_estimators: int,
    random_state: int
) -> Tuple[RandomForestRegressor, LinearRegression]:
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print("Training Linear Regression...")
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train, y_train)

    return rf, lr

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {"rmse": rmse, "r2": r2}

def compute_mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0

    sorted_indices = np.argsort(y_score)[::-1]
    max_label = np.max(y_true)

    for rank, idx in enumerate(sorted_indices, start=1):
        if y_true[idx] == max_label:
            return 1.0 / rank

    return 0.0

def evaluate_ranking(
    questions: np.ndarray,
    y_true: np.ndarray,
    y_scores: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    print("Evaluating ranking metrics...")

    df = pd.DataFrame({'question': questions})
    grouped = df.groupby('question').indices

    metrics = {model: {'ndcg@5': [], 'mrr': []} for model in y_scores}

    for q, indices in grouped.items():
        if len(indices) < 2:
            pass

        current_y_true = y_true[indices]

        if np.unique(current_y_true).size < 2 and np.max(current_y_true) == 0:
            pass

        for model_name, scores_full in y_scores.items():
            current_y_score = scores_full[indices]

            if len(current_y_true) > 0:
                try:
                    ndcg = ndcg_score([current_y_true], [current_y_score], k=5)
                except ValueError:
                    ndcg = 0.0
            else:
                ndcg = 0.0

            mrr = compute_mrr(current_y_true, current_y_score)

            metrics[model_name]['ndcg@5'].append(ndcg)
            metrics[model_name]['mrr'].append(mrr)

    final_metrics = {}
    for model_name, values in metrics.items():
        mean_ndcg = np.mean(values['ndcg@5']) if values['ndcg@5'] else 0.0
        mean_mrr = np.mean(values['mrr']) if values['mrr'] else 0.0

        final_metrics[model_name] = {
            "mean_ndcg@5": mean_ndcg,
            "mean_mrr": mean_mrr
        }
        print(f"[{model_name}] Mean nDCG@5: {mean_ndcg:.4f}, Mean MRR: {mean_mrr:.4f}")

    return final_metrics

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.data_dir)

    rf_model, lr_model = train_models(
        data['X_train'], data['y_train'],
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )

    print(f"Saving models to {args.output_dir}...")
    joblib.dump(rf_model, os.path.join(args.output_dir, "rf_model.joblib"))
    joblib.dump(lr_model, os.path.join(args.output_dir, "linreg_model.joblib"))

    results = {
        "regression": {},
        "ranking": {}
    }

    print("\n--- Regression Evaluation ---")

    y_pred_valid_rf = rf_model.predict(data['X_valid'])
    y_pred_valid_lr = lr_model.predict(data['X_valid'])

    y_pred_test_rf = rf_model.predict(data['X_test'])
    y_pred_test_lr = lr_model.predict(data['X_test'])

    print("Validation Set:")
    results["regression"]["valid_rf"] = evaluate_regression(data['y_valid'], y_pred_valid_rf, "RF Valid")
    results["regression"]["valid_lr"] = evaluate_regression(data['y_valid'], y_pred_valid_lr, "LR Valid")

    print("Test Set:")
    results["regression"]["test_rf"] = evaluate_regression(data['y_test'], y_pred_test_rf, "RF Test")
    results["regression"]["test_lr"] = evaluate_regression(data['y_test'], y_pred_test_lr, "LR Test")

    print("\n--- Ranking Evaluation (Test Set) ---")

    ranking_scores = {
        "RandomForest": y_pred_test_rf,
        "LinearRegression": y_pred_test_lr,
        "Baseline (relevance_score)": data['scores_test']
    }

    ranking_metrics = evaluate_ranking(
        data['questions_test'],
        data['y_test'],
        ranking_scores
    )
    results["ranking"] = ranking_metrics

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    print(f"Saving evaluation results to {results_path}...")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()

