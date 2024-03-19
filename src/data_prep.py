import os
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Loads the raw churn dataset from CSV.

    Parameters:
        path: file path to the raw dataset

    Returns:
        Pandas DataFrame containing customer records
    """
    df = pd.read_csv(path)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs minimal cleaning needed before feature engineering.

    Steps:
    - Remove duplicate customer IDs if any
    - Convert TotalCharges to numeric (dataset has some blank values)
    - Drop rows with missing churn label (should not exist, but safe)

    Returns:
        Cleaned DataFrame
    """
    df = df.drop_duplicates()

    # Convert TotalCharges column to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows where churn is missing
    df = df.dropna(subset=["Churn"])

    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Prints basic statistics needed for resume alignment:
    - Dataset size
    - Churn class imbalance
    - Missing value counts
    """
    print("Dataset Shape:", df.shape)

    churn_counts = df["Churn"].value_counts()
    print("\nChurn Distribution:")
    print(churn_counts)

    churn_rate = churn_counts["Yes"] / churn_counts.sum()
    print("\nChurn Rate:", round(churn_rate * 100, 2), "%")

    print("\nMissing Values (Top Columns):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    raw_path = "data/raw/telco_churn.csv"

    df_raw = load_raw_data(raw_path)
    df_clean = basic_cleaning(df_raw)

    os.makedirs("data/processed", exist_ok=True)
    df_clean.to_csv("data/processed/telco_clean.csv", index=False)

    print_dataset_summary(df_clean)
