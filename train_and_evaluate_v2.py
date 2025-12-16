import argparse
import os
import sys
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import scipy.sparse
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, ndcg_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate search relevance models."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing processed data (npz, npy, csv files)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save models and evaluation results"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=200,
        help="Number of trees for Random Forest (if not tuning)"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for reproducibility"
    )

    # Random Forest hyperparameter search
    parser.add_argument(
        "--rf_tune", action="store_true",
        help="If set, use RandomizedSearchCV to tune RandomForest"
    )
    parser.add_argument(
        "--rf_n_iter", type=int, default=30,
        help="Number of parameter settings sampled in RandomizedSearchCV"
    )
    parser.add_argument(
        "--rf_cv_folds", type=int, default=3,
        help="Number of CV folds for RF RandomizedSearchCV"
    )

    # cross-validation 
    parser.add_argument(
        "--cv_folds", type=int, default=5,
        help="Number of folds for cross-validation evaluation on train set"
    )

    parser.add_argument(
        "--top_k_errors", type=int, default=20,
        help="Number of worst-predicted examples to save for error analysis"
    )

    return parser.parse_args()


def load_data(data_dir: str) -> Dict[str, Any]:
    """
    Load all processed data artifacts from the given directory.

    Expected files for each split (train/valid/test):
      - X_<split>_text.npz       : sparse feature matrix
      - y_<split>.npy            : regression targets
      - relevance_score_<split>.npy : baseline scores
      - questions_<split>.csv    : CSV with a 'question' column
    """
    print(f"Loading data from {data_dir}...")
    data: Dict[str, Any] = {}
    splits = ["train", "valid", "test"]

    required_files = []
    for split in splits:
        required_files.extend(
            [
                f"X_{split}_text.npz",
                f"y_{split}.npy",
                f"questions_{split}.csv",
                f"relevance_score_{split}.npy",
            ]
        )

    missing = [
        f for f in required_files
        if not os.path.exists(os.path.join(data_dir, f))
    ]
    if missing:
        print(f"Error: Missing files in data_dir: {missing}")
        sys.exit(1)

    for split in splits:
        data[f"X_{split}"] = scipy.sparse.load_npz(
            os.path.join(data_dir, f"X_{split}_text.npz")
        )
        data[f"y_{split}"] = np.load(
            os.path.join(data_dir, f"y_{split}.npy")
        )
        data[f"scores_{split}"] = np.load(
            os.path.join(data_dir, f"relevance_score_{split}.npy")
        )

        q_df = pd.read_csv(os.path.join(data_dir, f"questions_{split}.csv"))
        data[f"questions_{split}"] = q_df["question"].values

    return data


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int,
    random_state: int,
    rf_tune: bool,
    rf_n_iter: int,
    rf_cv_folds: int,
):
    """
    Train a Random Forest regressor.
    If rf_tune is True, perform RandomizedSearchCV to tune hyperparameters.

    Returns:
        rf_model: trained RandomForestRegressor
        best_params: dict of best hyperparameters (or None if no tuning)
    """
    if rf_tune:
        print("Tuning Random Forest with RandomizedSearchCV...")
        rf = RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
        )

        param_distributions = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.2, 0.5],
        }

        search = RandomizedSearchCV(
            rf,
            param_distributions=param_distributions,
            n_iter=rf_n_iter,
            cv=rf_cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_rf = search.best_estimator_
        best_params = search.best_params_
        print("Best RF params:", best_params)
        return best_rf, best_params
    else:
        print("Training Random Forest with fixed hyperparameters...")
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        return rf, None


def train_linear_regression(X_train, y_train) -> LinearRegression:
    """
    Train a simple Linear Regression model.
    """
    print("Training Linear Regression...")
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train, y_train)
    return lr


def evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray, name: str
) -> Dict[str, float]:
    """
    Compute RMSE and R^2 score for regression.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    print(f"[{name}] RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {"rmse": rmse, "r2": r2}


def compute_mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Reciprocal Rank for a single query list.
    Rank is based on sorting y_score descending.
    Target is the item(s) with the maximum y_true label.
    """
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
    y_scores: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute nDCG@5 and MRR grouped by question for multiple models.

    Args:
        questions: array of question identifiers/text, length = n_samples
        y_true: true relevance labels, shape (n_samples,)
        y_scores: dict mapping model_name -> predicted scores (array of length n_samples)

    Returns:
        final_metrics: dict of per-model mean nDCG@5 and mean MRR.
    """
    print("Evaluating ranking metrics...")

    df = pd.DataFrame({"question": questions})
    grouped = df.groupby("question").indices

    metrics = {model: {"ndcg@5": [], "mrr": []} for model in y_scores}

    for _, indices in grouped.items():
        current_y_true = y_true[indices]
        if len(current_y_true) == 0:
            continue

        for model_name, scores_full in y_scores.items():
            current_y_score = scores_full[indices]

            # nDCG@5
            try:
                ndcg = float(
                    ndcg_score(
                        [current_y_true],
                        [current_y_score],
                        k=5,
                    )
                )
            except ValueError:
                ndcg = 0.0

            # MRR
            mrr = float(compute_mrr(current_y_true, current_y_score))

            metrics[model_name]["ndcg@5"].append(ndcg)
            metrics[model_name]["mrr"].append(mrr)

    final_metrics: Dict[str, Dict[str, float]] = {}
    for model_name, values in metrics.items():
        mean_ndcg = float(np.mean(values["ndcg@5"])) if values["ndcg@5"] else 0.0
        mean_mrr = float(np.mean(values["mrr"])) if values["mrr"] else 0.0

        final_metrics[model_name] = {
            "mean_ndcg@5": mean_ndcg,
            "mean_mrr": mean_mrr,
        }
        print(
            f"[{model_name}] Mean nDCG@5: {mean_ndcg:.4f}, "
            f"Mean MRR: {mean_mrr:.4f}"
        )

    return final_metrics


def run_cross_validation(
    model,
    X_train,
    y_train,
    cv_folds: int,
    name: str,
) -> Dict[str, float]:
    """
    Run k-fold cross-validation for a given model on the training set.
    Returns mean RMSE and R^2 across folds.
    """
    print(f"Running {cv_folds}-fold CV for {name}...")
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv_folds,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    mean_rmse = float(-np.mean(cv_results["test_rmse"]))
    mean_r2 = float(np.mean(cv_results["test_r2"]))
    print(f"[CV {name}] RMSE: {mean_rmse:.4f}, R2: {mean_r2:.4f}")
    return {"rmse": mean_rmse, "r2": mean_r2}


def save_error_analysis(
    questions: np.ndarray,
    y_true: np.ndarray,
    y_pred_rf: np.ndarray,
    y_pred_lr: np.ndarray,
    baseline_scores: np.ndarray,
    output_dir: str,
    top_k: int = 20,
):
    """
    Save error analysis CSV files:

    - errors_all.csv: all samples with predictions and absolute errors
    - top_errors_rf.csv: top-K worst RF predictions by absolute error
    - top_errors_lr.csv: top-K worst LR predictions by absolute error
    """
    print("Saving error analysis files...")

    df = pd.DataFrame(
        {
            "question": questions,
            "y_true": y_true,
            "rf_pred": y_pred_rf,
            "lr_pred": y_pred_lr,
            "baseline_score": baseline_scores,
        }
    )
    df["rf_abs_error"] = np.abs(df["y_true"] - df["rf_pred"])
    df["lr_abs_error"] = np.abs(df["y_true"] - df["lr_pred"])

    # full error table
    all_path = os.path.join(output_dir, "errors_all.csv")
    df.to_csv(all_path, index=False)
    print(f"Saved all errors to {all_path}")

    # Top-K worst RF predictions
    top_rf = df.sort_values("rf_abs_error", ascending=False).head(top_k)
    top_rf_path = os.path.join(output_dir, "top_errors_rf.csv")
    top_rf.to_csv(top_rf_path, index=False)
    print(f"Saved top-{top_k} RF errors to {top_rf_path}")

    # Top-K worst LR predictions
    top_lr = df.sort_values("lr_abs_error", ascending=False).head(top_k)
    top_lr_path = os.path.join(output_dir, "top_errors_lr.csv")
    top_lr.to_csv(top_lr_path, index=False)
    print(f"Saved top-{top_k} LR errors to {top_lr_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.data_dir)

    # RandomForest (with optional hyperparameter tuning)
    rf_model, rf_best_params = train_random_forest(
        data["X_train"],
        data["y_train"],
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        rf_tune=args.rf_tune,
        rf_n_iter=args.rf_n_iter,
        rf_cv_folds=args.rf_cv_folds,
    )

    # Linear Regression
    lr_model = train_linear_regression(
        data["X_train"],
        data["y_train"],
    )

    print(f"Saving models to {args.output_dir}...")
    joblib.dump(rf_model, os.path.join(args.output_dir, "rf_model.joblib"))
    joblib.dump(lr_model, os.path.join(args.output_dir, "linreg_model.joblib"))

    results = {
        "regression": {},
        "ranking": {},
        "cv": {},
        "meta": {
            "rf_best_params": rf_best_params,
        },
    }

    # Cross-validation
    if rf_best_params is not None:
        rf_for_cv = RandomForestRegressor(
            **rf_best_params,
            random_state=args.random_state,
            n_jobs=-1,
        )
    else:
        rf_for_cv = RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            n_jobs=-1,
        )

    rf_cv_scores = run_cross_validation(
        rf_for_cv,
        data["X_train"],
        data["y_train"],
        cv_folds=args.cv_folds,
        name="RandomForest",
    )
    results["cv"]["RandomForest"] = rf_cv_scores

    lr_for_cv = LinearRegression(n_jobs=-1)
    lr_cv_scores = run_cross_validation(
        lr_for_cv,
        data["X_train"],
        data["y_train"],
        cv_folds=args.cv_folds,
        name="LinearRegression",
    )
    results["cv"]["LinearRegression"] = lr_cv_scores

    # Regression evaluation on Valid/Test
    print("\n--- Regression Evaluation ---")

    # Random Forest 
    y_pred_valid_rf = rf_model.predict(data["X_valid"])
    y_pred_test_rf = rf_model.predict(data["X_test"])

    results["regression"]["valid_rf"] = evaluate_regression(
        data["y_valid"], y_pred_valid_rf, "RF Valid"
    )
    results["regression"]["test_rf"] = evaluate_regression(
        data["y_test"], y_pred_test_rf, "RF Test"
    )

    # Linear Regression 
    y_pred_valid_lr = lr_model.predict(data["X_valid"])
    y_pred_test_lr = lr_model.predict(data["X_test"])

    results["regression"]["valid_lr"] = evaluate_regression(
        data["y_valid"], y_pred_valid_lr, "LR Valid"
    )
    results["regression"]["test_lr"] = evaluate_regression(
        data["y_test"], y_pred_test_lr, "LR Test"
    )

    # Ranking evaluation
    print("\n--- Ranking Evaluation (Test Set) ---")
    ranking_scores = {
        "RandomForest": y_pred_test_rf,
        "LinearRegression": y_pred_test_lr,
        "Baseline (relevance_score)": data["scores_test"],
    }

    ranking_metrics = evaluate_ranking(
        data["questions_test"],
        data["y_test"],
        ranking_scores,
    )
    results["ranking"] = ranking_metrics

    # Error analysis
    save_error_analysis(
        questions=data["questions_test"],
        y_true=data["y_test"],
        y_pred_rf=y_pred_test_rf,
        y_pred_lr=y_pred_test_lr,
        baseline_scores=data["scores_test"],
        output_dir=args.output_dir,
        top_k=args.top_k_errors,
    )

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    print(f"Saving evaluation results to {results_path}...")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
