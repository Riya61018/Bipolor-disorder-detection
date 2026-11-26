"""Model suite definitions with leak-free preprocessing and feature selection pipelines."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _build_pipeline(estimator, top_k: int = 120, scale: bool = True) -> Pipeline:
    """Wrap an estimator in a leak-free imputation, scaling, and feature-selection pipeline."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    if top_k > 0:
        steps.append(("selector", SelectKBest(score_func=f_classif, k=top_k)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def get_candidate_models(
    top_k: int = 120,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Return a dictionary of diverse ML pipelines for bipolar EEG classification."""
    models = {
        "Logistic Regression (L2)": _build_pipeline(
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            ),
            top_k=top_k,
        ),
        "Linear Discriminant (LDA)": _build_pipeline(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            top_k=top_k,
        ),
        "SVM (RBF Kernel)": _build_pipeline(
            CalibratedClassifierCV(
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=random_state,
                ),
                cv=3,
                ensemble=False,
            ),
            top_k=top_k,
        ),
        "K-Nearest Neighbors (KNN)": _build_pipeline(
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
            top_k=top_k,
        ),
        "Gaussian Naive Bayes": _build_pipeline(
            GaussianNB(),
            top_k=top_k,
        ),
        "Random Forest": _build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            top_k=top_k,
        ),
        "Extra Trees": _build_pipeline(
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            top_k=top_k,
        ),
        "Gradient Boosting": _build_pipeline(
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.85,
                random_state=random_state,
            ),
            top_k=top_k,
        ),
        "HistGradientBoosting": _build_pipeline(
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=random_state,
            ),
            top_k=top_k,
        ),
        "Neural Network (MLP)": _build_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                alpha=1.0,
                max_iter=600,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=random_state,
            ),
            top_k=top_k,
        ),
    }
    return models
