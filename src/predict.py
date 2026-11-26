"""Final inference module using the winning saved EEG model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import clean_feature_frame

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_bipolar_model.joblib"


class BipolarEEGPredictor:
    """Loads the winning trained pipeline and predicts Bipolar Disorder risk from EEG data."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Run 'python src/train_and_compare.py' first."
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.model_name: str = bundle["model_name"]
        self.task: str = bundle["task"]
        self.eeg_only: bool = bundle.get("eeg_only", False)
        self.class_names: tuple[str, str] = tuple(bundle["class_names"])
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.feature_medians: dict[str, float] = dict(bundle.get("feature_medians", {}))
        self.metrics: dict = dict(bundle.get("metrics", {}))
        self.top_features: list[dict] = list(bundle.get("top_features", []))

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align raw or preprocessed input dataframe to expected training feature schema."""
        cleaned = clean_feature_frame(df, eeg_only=self.eeg_only)
        aligned_data = {
            col: (
                cleaned[col]
                if col in cleaned.columns
                else np.full(len(cleaned), self.feature_medians.get(col, np.nan), dtype=float)
            )
            for col in self.feature_names
        }
        return pd.DataFrame(aligned_data, index=cleaned.index)

    @staticmethod
    def _risk_tier(prob: float) -> str:
        if prob >= 0.70:
            return "High"
        if prob >= 0.45:
            return "Moderate"
        return "Low"

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions, positive-class probabilities, and risk tiers for input rows."""
        X_ready = self.prepare_features(df)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_ready)[:, 1]
        else:
            scores = self.model.decision_function(X_ready)
            probs = 1.0 / (1.0 + np.exp(-scores))

        preds = self.model.predict(X_ready)

        results = pd.DataFrame(
            {
                "predicted_class_id": preds.astype(int),
                "predicted_label": [self.class_names[int(p)] for p in preds],
                "bipolar_probability": np.round(probs.astype(float), 4),
                "risk_tier": [self._risk_tier(float(p)) for p in probs],
                "model_used": self.model_name,
            },
            index=df.index,
        )
        if "true_label" in df.columns:
            results["true_label"] = df["true_label"].values
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bipolar EEG predictions using the best trained model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "sample_test_patients.csv",
        help="Path to input CSV containing patient EEG rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "predictions.csv",
        help="Where to write prediction results CSV.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to saved joblib model bundle.",
    )
    args = parser.parse_args()

    df_in = pd.read_csv(args.input)
    predictor = BipolarEEGPredictor(model_path=args.model)
    df_out = predictor.predict(df_in)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    print(f"Loaded winning model: {predictor.model_name} (task={predictor.task})")
    print(f"Predicted {len(df_out)} patients -> saved to {args.output}\n")
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
