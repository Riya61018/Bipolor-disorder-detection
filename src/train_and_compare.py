"""Train, cross-validate, compare 10 ML models, and export the best model for final use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import TaskType, load_eeg_dataset
from src.models import get_candidate_models

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _get_positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Extract positive-class probability or decision score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _plot_model_comparison(df_results: pd.DataFrame, task: str, out_path: Path) -> None:
    """Save horizontal bar chart comparing CV ROC-AUC, Test ROC-AUC, and Test F1."""
    df_sorted = df_results.sort_values("cv_roc_auc_mean", ascending=True)
    y_pos = np.arange(len(df_sorted))
    height = 0.27

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(y_pos + height, df_sorted["cv_roc_auc_mean"], height=height, label="CV ROC-AUC (5-fold)", color="#2563eb")
    ax.barh(y_pos, df_sorted["test_roc_auc"], height=height, label="Holdout Test ROC-AUC", color="#10b981")
    ax.barh(y_pos - height, df_sorted["test_f1"], height=height, label="Holdout Test F1", color="#f59e0b")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["model"], fontsize=10)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title(f"Model Comparison — EEG Bipolar Detection ({task})", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(
            min(row["cv_roc_auc_mean"] + 0.01, 0.99),
            idx + height,
            f"{row['cv_roc_auc_mean']:.3f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_roc_curves(
    fitted_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_model_name: str,
    task: str,
    out_path: Path,
) -> None:
    """Plot holdout ROC curves for all benchmarked models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, model in fitted_models.items():
        scores = _get_positive_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_val = roc_auc_score(y_test, scores)
        lw = 2.8 if name == best_model_name else 1.3
        alpha = 1.0 if name == best_model_name else 0.7
        prefix = "[BEST] " if name == best_model_name else ""
        ax.plot(fpr, tpr, linewidth=lw, alpha=alpha, label=f"{prefix}{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Holdout ROC Curves ({task})", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: tuple[str, str],
    best_model_name: str,
    out_path: Path,
) -> None:
    """Plot confusion matrix for the selected best model."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {best_model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_top_features(
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_model_name: str,
    out_path: Path,
    top_n: int = 15,
) -> list[dict[str, float | str]]:
    """Compute permutation importance on holdout set and save top EEG biomarkers chart."""
    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    importances = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    top_df = importances.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["feature"], top_df["importance_mean"], xerr=top_df["importance_std"], color="#6366f1")
    ax.set_xlabel("Permutation Importance (ROC-AUC drop)")
    ax.set_title(f"Top {top_n} EEG Features — {best_model_name}", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return importances.head(top_n).to_dict(orient="records")


def run_training_and_comparison(
    task: TaskType = "bipolar_vs_healthy",
    top_k: int = 120,
    eeg_only: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Run full CV + holdout evaluation across all candidate models and save the winner."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_eeg_dataset(task=task, eeg_only=eeg_only)
    X, y = dataset.X, dataset.y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Save unseen holdout samples for CLI & Streamlit demos
    sample_test = X_test.copy()
    sample_test["true_label"] = [dataset.class_names[val] for val in y_test]
    sample_test.to_csv(PROCESSED_DIR / "sample_test_patients.csv", index=False)

    effective_k = min(top_k, X_train.shape[1]) if top_k > 0 else 0
    candidate_models = get_candidate_models(top_k=effective_k, random_state=random_state)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
    }

    rows: list[dict] = []
    fitted_models: dict = {}

    print(f"\n=== Bipolar EEG Detection Benchmark ({task}) ===")
    print(
        f"Samples: total={len(X)}, train={len(X_train)}, test={len(X_test)} | "
        f"Features={X.shape[1]} (SelectKBest top_k={effective_k}) | "
        f"Positive class='{dataset.class_names[1]}' ({int(y.sum())}/{len(y)})\n"
    )

    for name, pipeline in candidate_models.items():
        cv_out = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_scores = _get_positive_scores(pipeline, X_test)

        row = {
            "model": name,
            "cv_roc_auc_mean": float(np.mean(cv_out["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_out["test_roc_auc"])),
            "cv_balanced_acc_mean": float(np.mean(cv_out["test_balanced_accuracy"])),
            "cv_f1_mean": float(np.mean(cv_out["test_f1"])),
            "test_roc_auc": float(roc_auc_score(y_test, y_scores)),
            "test_balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        }
        rows.append(row)
        print(
            f"{name:28s} | CV AUC: {row['cv_roc_auc_mean']:.3f} +/- {row['cv_roc_auc_std']:.3f} "
            f"| Test AUC: {row['test_roc_auc']:.3f} | Test BalAcc: {row['test_balanced_acc']:.3f} "
            f"| Test F1: {row['test_f1']:.3f}"
        )

    df_results = (
        pd.DataFrame(rows)
        .sort_values(
            by=["cv_roc_auc_mean", "cv_balanced_acc_mean", "cv_f1_mean"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    df_results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    best_row = df_results.iloc[0].to_dict()
    best_model_name = str(best_row["model"])
    best_model = fitted_models[best_model_name]

    y_best_pred = best_model.predict(X_test)
    report_str = classification_report(
        y_test,
        y_best_pred,
        target_names=list(dataset.class_names),
        digits=4,
    )
    (RESULTS_DIR / "best_model_report.txt").write_text(
        f"Best Model (selected by 5-fold CV ROC-AUC): {best_model_name}\n"
        f"Task: {task}\n\n"
        f"{report_str}",
        encoding="utf-8",
    )

    _plot_model_comparison(df_results, task, PLOTS_DIR / "model_comparison.png")
    _plot_roc_curves(fitted_models, X_test, y_test, best_model_name, task, PLOTS_DIR / "roc_curves.png")
    _plot_confusion_matrix(
        y_test,
        y_best_pred,
        dataset.class_names,
        best_model_name,
        PLOTS_DIR / "best_model_confusion_matrix.png",
    )
    top_features = _plot_top_features(
        best_model,
        X_test,
        y_test,
        best_model_name,
        PLOTS_DIR / "top_eeg_features.png",
    )

    # Save the winning pipeline + reference medians for safe inference on partial inputs
    feature_medians = X_train.median(numeric_only=True).to_dict()
    artifact_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "task": task,
        "eeg_only": eeg_only,
        "class_names": dataset.class_names,
        "feature_names": dataset.feature_names,
        "feature_medians": feature_medians,
        "metrics": best_row,
        "top_features": top_features,
    }
    joblib.dump(artifact_bundle, MODELS_DIR / "best_bipolar_model.joblib")

    metadata = {
        "best_model_name": best_model_name,
        "selection_criterion": "5-fold CV ROC-AUC (with CV Balanced Accuracy & F1 tie-breakers)",
        "task": task,
        "eeg_only": eeg_only,
        "class_names": list(dataset.class_names),
        "n_samples_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "top_k_selected": int(effective_k),
        "best_model_metrics": best_row,
        "top_features": top_features,
        "all_models": df_results.to_dict(orient="records"),
    }
    (MODELS_DIR / "best_model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\n[WINNER] Selected model: {best_model_name}")
    print(
        f"  CV ROC-AUC={best_row['cv_roc_auc_mean']:.3f} | "
        f"Holdout Test ROC-AUC={best_row['test_roc_auc']:.3f} | "
        f"Holdout Test Balanced Acc={best_row['test_balanced_acc']:.3f} | "
        f"Holdout Test F1={best_row['test_f1']:.3f}"
    )
    print(f"  Saved final model -> {MODELS_DIR / 'best_bipolar_model.joblib'}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare ML models for Bipolar EEG detection.")
    parser.add_argument(
        "--task",
        choices=["bipolar_vs_healthy", "bipolar_vs_depression", "bipolar_vs_all"],
        default="bipolar_vs_healthy",
        help="Classification task to train and benchmark.",
    )
    parser.add_argument("--top-k", type=int, default=120, help="Number of top ANOVA EEG features per fold.")
    parser.add_argument("--eeg-only", action="store_true", help="Exclude demographic columns (sex, age, education, IQ).")
    args = parser.parse_args()

    run_training_and_comparison(
        task=args.task,
        top_k=args.top_k,
        eeg_only=args.eeg_only,
    )


if __name__ == "__main__":
    main()
