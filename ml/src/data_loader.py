"""
data_loader.py
Loads the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn
and returns it as a clean pandas DataFrame.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data() -> pd.DataFrame:
    """
    Loads the Breast Cancer Wisconsin dataset.

    Returns:
        pd.DataFrame: Features + target column ('diagnosis').
                      diagnosis: 0 = malignant, 1 = benign (sklearn's native encoding)
    """
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()

    # sklearn names the target column "target"; rename for clarity
    df = df.rename(columns={"target": "diagnosis"})

    return df


def get_feature_names() -> list:
    """Returns the list of 30 feature names used by the model."""
    dataset = load_breast_cancer(as_frame=True)
    return list(dataset.feature_names)


def get_target_names() -> list:
    """Returns human-readable class labels in sklearn's index order."""
    dataset = load_breast_cancer(as_frame=True)
    return list(dataset.target_names)  # ['malignant', 'benign']


if __name__ == "__main__":
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nClass distribution:\n{df['diagnosis'].value_counts()}")
    print(f"\nTarget mapping: 0={get_target_names()[0]}, 1={get_target_names()[1]}")
    print(f"\nMissing values: {df.isnull().sum().sum()}")