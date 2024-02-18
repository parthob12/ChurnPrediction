import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    classification_report,
    confusion_matrix,
)


from src.features import add_features, split_xy


def load_split():
    raw = pd.read_csv("data/processed/telco_clean.csv")
    df = add_features(raw)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def score_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s_min = float(np.min(s))
        s_max = float(np.max(s))
        denom = (s_max - s_min) if (s_max != s_min) else 1.0
        return (s - s_min) / denom
    return model.predict(X).astype(float)


def choose_threshold_by_fbeta(y_true, y_score, beta: float = 2.0) -> float:
    precision, recall, thresh = precision_recall_curve(y_true, y_score)

    b2 = beta * beta
    denom = (b2 * precision + recall)
    denom = np.where(denom == 0, 1e-9, denom)
    f = (1 + b2) * precision * recall / denom

    best_idx = int(np.argmax(f))
    if best_idx >= len(thresh):
        return 0.5
    return float(thresh[best_idx])


def plot_roc(y_true, y_score, name: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC curve {name} AUC {auc:.4f}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.tight_layout()
    plt.savefig(f"reports/roc_{name}.png")
    plt.close()

    return float(auc)


def plot_pr(y_true, y_score, name: str):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"Precision recall {name} AP {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(f"reports/pr_{name}.png")
    plt.close()

    return float(ap)


def main():
    X_test, y_test = load_split()

    model_names = ["logreg", "rf", "hgb"]
    results = {}

    for name in model_names:
        model = joblib.load(f"models/{name}.pkl")
        y_score = score_proba(model, X_test)

        auc = plot_roc(y_test, y_score, name)
        ap = plot_pr(y_test, y_score, name)

        threshold = choose_threshold_by_fbeta(y_test, y_score, beta=2.0)
        y_pred = (y_score >= threshold).astype(int)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "roc_auc": auc,
            "avg_precision": ap,
            "threshold": threshold,
            "classification_report": report,
            "confusion_matrix": cm,
        }

        print("\nModel:", name)
        print("ROC AUC:", round(auc, 4))
        print("Avg precision:", round(ap, 4))
        print("Chosen threshold:", round(threshold, 4))
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:", cm)

    with open("reports/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved reports:")
    print("reports/metrics.json")
    print("reports/roc_name.png and reports/pr_name.png for each model")


if __name__ == "__main__":
    main()
