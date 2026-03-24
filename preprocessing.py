import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

from sklearn.datasets import fetch_kddcup99
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample


RANDOM_STATE = 42


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
    n_components: int = 4,
    vis_samples: int = 1000,
) -> dict:
    df = load_kdd99(percent10=percent10)

    target_col = "labels"
    df[target_col] = (df[target_col] != "normal.").astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    original_dim = X.shape[1]

    categorical_cols = ["protocol_type", "service", "flag"]
    present_categorical = [c for c in categorical_cols if c in X.columns]
    X = pd.get_dummies(X, columns=present_categorical, drop_first=False)

    encoded_dim = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df["label"] = y_train.values

    class_counts = train_df["label"].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    majority_df = train_df[train_df["label"] == majority_class]
    minority_df = train_df[train_df["label"] == minority_class]

    majority_down = resample(
        majority_df,
        replace=False,
        n_samples=len(minority_df),
        random_state=RANDOM_STATE,
    )

    train_bal = pd.concat([majority_down, minority_df], axis=0)
    train_bal = train_bal.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    X_train_bal = train_bal.drop(columns=["label"])
    y_train_bal = train_bal["label"].to_numpy()
    y_test = y_test.to_numpy()

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
    X_test_pca = pca.transform(X_test_scaled).astype(np.float32)

    X_vis_scaled, y_vis = take_subset(X_train_scaled, y_train_bal, n_samples=vis_samples)
    X_vis_pca = pca.transform(X_vis_scaled).astype(np.float32)

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "X_vis_scaled": X_vis_scaled,
        "X_vis_pca": X_vis_pca,
        "y_vis": y_vis,
        "y_train": y_train_bal,
        "y_test": y_test,
        "pca": pca,
        "scaler": scaler,
        "original_dim": original_dim,
        "encoded_dim": encoded_dim,
        "final_dim": n_components,
    }


def plot_dimensionality_summary(original_dim: int, encoded_dim: int, final_dim: int):
    stages = ["Raw", "One-hot", "After PCA"]
    dims = [original_dim, encoded_dim, final_dim]

    plt.figure(figsize=(8, 5))
    plt.bar(stages, dims)
    plt.ylabel("Number of dimensions")
    plt.title("KDD99 dimensionality reduction pipeline")
    plt.tight_layout()
    plt.show()


def plot_explained_variance(pca: PCA):
    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(evr) + 1), cumulative, marker="o")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(X_pca: np.ndarray, y: np.ndarray, max_points: int = 3000):
    X_plot, y_plot = take_subset(X_pca, y, n_samples=max_points)

    if X_plot.shape[1] < 2:
        raise ValueError("Need at least 2 PCA components to make a scatter plot.")

    plt.figure(figsize=(8, 6))
    for label, name in [(0, "normal"), (1, "attack")]:
        mask = y_plot == label
        plt.scatter(
            X_plot[mask, 0],
            X_plot[mask, 1],
            s=10,
            alpha=0.5,
            label=name,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KDD99 projected onto first 2 PCA components")
    plt.legend()
    plt.tight_layout()
    plt.show()


def timed_preprocess(percent10: bool = True, n_components: int = 4, vis_samples: int = 1000):
    start = time.time()
    result = preprocess_kdd99(
        percent10=percent10,
        n_components=n_components,
        vis_samples=vis_samples,
    )
    print(f"Preprocessing time: {time.time() - start:.2f}s")
    return result



# -------------------- UMAP STUFF -----------------------

# https://www.geeksforgeeks.org/data-visualization/techniques-for-visualizing-high-dimensional-data/
def compute_umap_embedding(X: np.ndarray, n_neighbors: int = 10, min_dist: float = 0.3):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        low_memory=True,
    )
    return reducer.fit_transform(X)


def plot_umap(X: np.ndarray, y: np.ndarray, title: str):
    X_embedded = compute_umap_embedding(X)

    plt.figure(figsize=(8, 6))
    for label, name in [(0, "normal"), (1, "attack")]:
        mask = y == label
        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            s=10,
            alpha=0.5,
            label=name,
        )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_umap_comparison(X_high: np.ndarray, X_pca: np.ndarray, y: np.ndarray):
    X_high_emb = compute_umap_embedding(X_high)
    X_pca_emb = compute_umap_embedding(X_pca)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for label, name in [(0, "normal"), (1, "attack")]:
        mask = y == label

        axes[0].scatter(
            X_high_emb[mask, 0],
            X_high_emb[mask, 1],
            s=10,
            alpha=0.5,
            label=name,
        )
        axes[1].scatter(
            X_pca_emb[mask, 0],
            X_pca_emb[mask, 1],
            s=10,
            alpha=0.5,
            label=name,
        )

    axes[0].set_title("UMAP on high-dimensional scaled data")
    axes[1].set_title("UMAP after PCA")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.show()

