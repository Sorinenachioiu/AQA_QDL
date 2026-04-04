import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

# Load KDD99 dataset
def load_kdd99(percent10: bool = True) -> pd.DataFrame:
    bunch = fetch_kddcup99(
        percent10=percent10,
        as_frame=True,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    df = bunch.frame.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            )
    return df

def take_subset(X: np.ndarray, y: np.ndarray, n_samples: int = 2000):
    if len(X) <= n_samples:
        return X, y
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=n_samples, replace=False)
    return X[idx], y[idx]

def preprocess_kdd99(
    percent10: bool = True,
    n_selected_features: int = 8, 
    vis_samples: int = 1000,
) -> dict:
    df = load_kdd99(percent10=percent10)
    target_col = "labels"
    df[target_col] = (df[target_col] != "normal.").astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    original_dim = X.shape[1]
    # One-hot encode categorical features
    categorical_cols = ["protocol_type", "service", "flag"]
    present_categorical = [c for c in categorical_cols if c in X.columns]
    X = pd.get_dummies(X, columns=present_categorical, drop_first=False)
    encoded_dim = X.shape[1]
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    train_df = X_train.copy()
    train_df["label"] = y_train.values
    # Under-sample majority class to balance dataset
    majority_df = train_df[train_df["label"] == train_df["label"].value_counts().idxmax()]
    minority_df = train_df[train_df["label"] == train_df["label"].value_counts().idxmin()]
    majority_down = resample(
        majority_df, replace=False, n_samples=len(minority_df), random_state=RANDOM_STATE
    )
    train_bal = pd.concat([majority_down, minority_df], axis=0).sample(frac=1.0, random_state=RANDOM_STATE)
    X_train_bal = train_bal.drop(columns=["label"])
    y_train_bal = train_bal["label"].to_numpy().astype(int)
    y_test = y_test.to_numpy().astype(int)
    print(f"Selecting top {n_selected_features} features using RFE...")
    # Use smaller subset for RFE bc the full dataset is large
    X_rfe_subset, y_rfe_subset = take_subset(X_train_bal.values, y_train_bal, n_samples=5000)
    # Fit RFE with a simple Random Forest classifier
    selector = RFE(estimator=RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE), 
                   n_features_to_select=n_selected_features, step=5)
    selector = selector.fit(X_rfe_subset, y_rfe_subset)
    
    # Apply selection
    X_train_selected = X_train_bal.iloc[:, selector.support_]
    X_test_selected = X_test.iloc[:, selector.support_]
    feature_names = X_train_selected.columns.tolist()
    print(f"Selected features: {feature_names}")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_final = scaler.fit_transform(X_train_selected).astype(np.float32)
    X_test_final = scaler.transform(X_test_selected).astype(np.float32)
    X_vis, y_vis = take_subset(X_train_final, y_train_bal, n_samples=vis_samples)

    return {
        "X_train": X_train_final,
        "X_test": X_test_final,
        "X_vis": X_vis,
        "y_vis": y_vis,
        "y_train": y_train_bal,
        "y_test": y_test,
        "scaler": scaler,
        "selector": selector,
        "original_dim": original_dim,
        "encoded_dim": encoded_dim,
        "final_dim": n_selected_features,
        "feature_names": feature_names
    }

def plot_dimensionality_summary(original_dim: int, encoded_dim: int, final_dim: int):
    stages = ["Raw", "One-hot", "After RFE"]
    dims = [original_dim, encoded_dim, final_dim]
    plt.figure(figsize=(8, 5))
    plt.bar(stages, dims, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel("Number of dimensions")
    plt.title("KDD99 Paper-based Preprocessing Pipeline")
    plt.tight_layout()
    plt.show()

def timed_preprocess(percent10: bool = True, n_features: int = 8, vis_samples: int = 1000):
    start = time.time()
    result = preprocess_kdd99(
        percent10=percent10,
        n_selected_features=n_features,
        vis_samples=vis_samples,
    )
    print(f"Preprocessing time: {time.time() - start:.2f}s")
    return result
