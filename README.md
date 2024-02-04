Customer Churn Prediction System

Overview
This project predicts customer churn using the Telco Customer Churn dataset. The focus is on practical machine learning challenges such as class imbalance, missing values, feature engineering, model comparison, threshold tuning, and feature importance for explainability.

Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib

Project structure
data/raw contains the raw CSV
data/processed contains the cleaned CSV
src contains scripts for feature engineering, training, evaluation, and importance
models contains trained pipelines
reports contains metrics and plots

How to run
1. Place the dataset CSV into data/raw as telco_churn.csv
2. Run data preparation
python -m src.data_prep
3. Run exploratory analysis
python -m src.eda
4. Train models
python -m src.train
5. Evaluate models
python -m src.evaluate
6. Compute feature importance
python -m src.importance

What to expect
Evaluation saves ROC and precision-recall plots for each model in reports.
metrics.json contains ROC AUC, average precision, a tuned threshold, a confusion matrix, and a full classification report.

Business framing for threshold selection
A higher threshold favors precision and reduces false positives.
A lower threshold favors recall and reduces missed churners.
The evaluation script selects a threshold using an F-beta score with beta set to 2.0 to bias toward recall.

Limitations
This is a static dataset. A real production system would monitor drift, retrain on a schedule, and log predictions for audit.

Next steps
Add cross-validation for more stable comparisons
Add calibration checks for predicted probabilities
Add a simple model card that documents data, metrics, and usage
