"""
preprocessing.py
Handles data cleaning, feature scaling, and train/test splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset:
    - Drops exact duplicate rows
    - Verifies no missing values (dataset is already clean, but we check defensively)
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))

    if before != after:
        print(f"Removed {before - after} duplicate rows")

    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame, target_col: str = "diagnosis"):
    """Splits DataFrame into X (features) and y (target)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Splits data into train/test sets, stratified by target class
    to preserve the benign/malignant ratio in both sets.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def scale_features(X_train, X_test):
    """
    Fits a StandardScaler on training data only (avoids data leakage),
    then transforms both train and test sets.

    Returns:
        X_train_scaled, X_test_scaled, fitted_scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    df = clean_data(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print(f"Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
    print(f"Test class balance:\n{y_test.value_counts(normalize=True)}")
    