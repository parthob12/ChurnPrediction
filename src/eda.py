import pandas as pd
import matplotlib.pyplot as plt


def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def churn_distribution(df: pd.DataFrame) -> None:
    counts = df["Churn"].value_counts()
    total = counts.sum()
    yes = counts.get("Yes", 0)
    rate = (yes / total) if total else 0

    print("Rows:", len(df))
    print("Columns:", df.shape[1])
    print("Churn counts:")
    print(counts)
    print("Churn rate percent:", round(rate * 100, 2))

    ax = counts.plot(kind="bar")
    ax.set_title("Churn distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("reports/churn_distribution.png")
    plt.close()


def missing_values(df: pd.DataFrame, top_n: int = 12) -> None:
    miss = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values top columns:")
    print(miss.head(top_n))


def quick_numeric_summary(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        print("\nNo numeric columns detected.")
        return

    print("\nNumeric describe:")
    print(df[numeric_cols].describe().T)


if __name__ == "__main__":
    df = load_clean("data/processed/telco_clean.csv")
    churn_distribution(df)
    missing_values(df)
    quick_numeric_summary(df)
    print("\nSaved plot: reports/churn_distribution.png")
