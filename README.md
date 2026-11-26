# Bipolar Disorder Detection Using EEG & Machine Learning

End-to-end machine learning project that trains, cross-validates, and compares **10 ML models** on the **BRMH EEG Psychiatric Disorders Dataset**, automatically selects the best-performing model via 5-fold Stratified Cross-Validation, and exports it for final CLI & web-app inference.

> **Medical Disclaimer:** This project is a research and educational prototype and is **not** intended for standalone clinical diagnosis.

## Project Structure

```text
bipolar-eeg-ml/
├── app.py                              # Streamlit web app for final predictions & comparison dashboard
├── requirements.txt                    # Python dependencies
├── data/
│   ├── raw/EEG.machinelearing_data_BRMH.csv
│   └── processed/sample_test_patients.csv
├── models/
│   ├── best_bipolar_model.joblib       # Winning pipeline selected by CV ROC-AUC
│   └── best_model_metadata.json
├── results/
│   ├── model_comparison.csv            # Full benchmark across all 10 models
│   ├── best_model_report.txt
│   └── plots/                          # Comparison, ROC curves, confusion matrix, top EEG features
├── src/
│   ├── data_loader.py                  # Loads & cleans BRMH EEG CSV for selected task
│   ├── models.py                       # 10 leak-free sklearn Pipelines (Imputer + Scaler + SelectKBest)
│   ├── train_and_compare.py            # 5-fold CV + Holdout evaluation + winner selection
│   └── predict.py                      # Final CLI/Python predictor using the winning model
└── tests/
    └── test_pipeline.py                # Automated verification suite
```

## Quick Start

1. **Train & compare all 10 models** (selects and saves the best model automatically):
   ```bash
   python src/train_and_compare.py --task bipolar_vs_healthy
   ```
   Optional tasks:
   - `--task bipolar_vs_healthy` *(default: 67 Bipolar vs 95 Healthy controls)*
   - `--task bipolar_vs_depression` *(67 Bipolar vs 199 Depressive disorder)*
   - `--task bipolar_vs_all` *(67 Bipolar vs 878 others)*
   - Add `--eeg-only` to exclude demographic columns (`sex`, `age`, `education`, `IQ`).

2. **Run predictions with the winning model (CLI):**
   ```bash
   python src/predict.py --input data/processed/sample_test_patients.csv --output results/predictions.csv
   ```

3. **Launch the interactive Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Run tests:**
   ```bash
   python -m unittest discover -s tests -v
   ```
