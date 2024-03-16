import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from src.features import add_features, split_xy
from src.preprocess import build_preprocessor


def load_data() -> pd.DataFrame:
    return pd.read_csv("data/processed/telco_clean.csv")


def build_models():
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            n_jobs=None,
        ),
        "hgb": HistGradientBoostingClassifier(random_state=42),
    }


def main() -> None:
    raw = load_data()
    df = add_features(raw)
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = build_preprocessor(df)

    models = build_models()
    saved = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, f"models/{name}.pkl")
        saved[name] = {"path": f"models/{name}.pkl"}

    with open("models/index.json", "w") as f:
        json.dump(
            {
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "models": saved,
            },
            f,
            indent=2,
        )

    print("Saved models:", ", ".join(models.keys()))
    print("Saved models/index.json")


if __name__ == "__main__":
    main()
