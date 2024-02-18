import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from src.features import add_features, split_xy


def load_split():
    raw = pd.read_csv("data/processed/telco_clean.csv")
    df = add_features(raw)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def best_model_name() -> str:
    with open("reports/metrics.json", "r") as f:
        metrics = json.load(f)
    best = None
    best_auc = -1.0
    for name, m in metrics.items():
        auc = float(m["roc_auc"])
        if auc > best_auc:
            best_auc = auc
            best = name
    return best if best else "logreg"


def get_feature_names(pipeline):
    pre = pipeline.named_steps["pre"]
    names = []
    for block_name, transformer, cols in pre.transformers_:
        if block_name == "num":
            names.extend(cols)
        elif block_name == "cat":
            ohe = transformer.named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(cols).tolist()
            names.extend(cat_names)
    return names


def plot_top(features, values, title, out_path, top_k=15):
    idx = np.argsort(values)[::-1][:top_k]
    top_features = [features[i] for i in idx]
    top_values = [values[i] for i in idx]

    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], top_values[::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    X_test, y_test = load_split()
    name = best_model_name()
    model = joblib.load(f"models/{name}.pkl")

    feat_names = get_feature_names(model)

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=8, random_state=42, scoring="roc_auc"
    )
    perm_mean = perm.importances_mean

    plot_top(
        feat_names,
        perm_mean,
        f"Permutation importance {name}",
        "reports/feature_importance_permutation.png",
        top_k=15,
    )

    out = {
        "best_model": name,
        "top_features": [
            {"feature": feat_names[i], "importance": float(perm_mean[i])}
            for i in np.argsort(perm_mean)[::-1][:20]
        ],
    }

    with open("reports/feature_importance.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Best model:", name)
    print("Saved reports/feature_importance_permutation.png")
    print("Saved reports/feature_importance.json")


if __name__ == "__main__":
    main()
