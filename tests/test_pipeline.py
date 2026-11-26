"""Unit and integration tests for the Bipolar EEG ML pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_eeg_dataset
from src.predict import BipolarEEGPredictor


class TestBipolarEEGPipeline(unittest.TestCase):
    """Verify dataset loading, saved best model artifacts, and final prediction."""

    def test_load_bipolar_vs_healthy(self) -> None:
        ds = load_eeg_dataset(task="bipolar_vs_healthy")
        self.assertEqual(len(ds.X), 162)
        self.assertEqual(int(ds.y.sum()), 67)
        self.assertNotIn("Unnamed: 122", ds.X.columns)
        self.assertNotIn("main.disorder", ds.X.columns)
        self.assertNotIn("specific.disorder", ds.X.columns)

    def test_load_bipolar_vs_depression(self) -> None:
        ds = load_eeg_dataset(task="bipolar_vs_depression")
        self.assertEqual(len(ds.X), 266)
        self.assertEqual(int(ds.y.sum()), 67)

    def test_saved_best_model_predicts_cleanly(self) -> None:
        model_path = PROJECT_ROOT / "models" / "best_bipolar_model.joblib"
        sample_path = PROJECT_ROOT / "data" / "processed" / "sample_test_patients.csv"
        self.assertTrue(model_path.exists(), "best_bipolar_model.joblib must exist")
        self.assertTrue(sample_path.exists(), "sample_test_patients.csv must exist")

        predictor = BipolarEEGPredictor(model_path=model_path)
        df_sample = pd.read_csv(sample_path).head(5)
        preds = predictor.predict(df_sample)

        self.assertEqual(len(preds), 5)
        self.assertIn("predicted_label", preds.columns)
        self.assertIn("bipolar_probability", preds.columns)
        self.assertTrue(((preds["bipolar_probability"] >= 0.0) & (preds["bipolar_probability"] <= 1.0)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
