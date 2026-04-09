import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def load_cic_iot23(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Path not found: {path}")

    if path.is_file():
        return pd.read_csv(path)

    raise ValueError(f"Smth went wrong")



def take_subset(X: np.ndarray, y: np.ndarray, n_samples: int = 2000):
    if len(X) <= n_samples:
        return X, y

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=n_samples, replace=False)
    return X[idx], y[idx]


def _balance_binary_training_set(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    train_df = X_train.copy()
    train_df["label"] = np.asarray(y_train)

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

    return X_train_bal, y_train_bal


def preprocess_tabular_dataset(
    df: pd.DataFrame,
    label_col: str,
    label_map_fn,
    n_components: int = 4,
    vis_samples: int = 1000,
    categorical_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
    balance_train: bool = True,
) -> dict:
    df = df.copy()

    if drop_cols is not None:
        existing_drop_cols = [c for c in drop_cols if c in df.columns and c != label_col]
        df = df.drop(columns=existing_drop_cols)

    df[label_col] = label_map_fn(df[label_col]).astype(int)

    X = df.drop(columns=[label_col])
    y = df[label_col]

    original_dim = X.shape[1]

    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        categorical_cols = [c for c in categorical_cols if c in X.columns]

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print("Dropping NaN columns:", all_nan_cols)
        X = X.drop(columns=all_nan_cols)

    X = X.fillna(0)

    arr = X.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(arr).all():
        bad_cols = X.columns[~np.isfinite(arr).all(axis=0)].tolist()
        raise ValueError(f"Bad columns: {bad_cols}")

    encoded_dim = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if balance_train:
        X_train_used, y_train_used = _balance_binary_training_set(X_train, y_train)
    else:
        X_train_used = X_train.copy()
        y_train_used = np.asarray(y_train)

    y_test = np.asarray(y_test)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_used).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
    X_test_pca = pca.transform(X_test_scaled).astype(np.float32)

    X_vis_scaled, y_vis = take_subset(X_train_scaled, y_train_used, n_samples=vis_samples)
    X_vis_pca = pca.transform(X_vis_scaled).astype(np.float32)

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "X_vis_scaled": X_vis_scaled,
        "X_vis_pca": X_vis_pca,
        "y_vis": y_vis,
        "y_train": y_train_used,
        "y_test": y_test,
        "pca": pca,
        "scaler": scaler,
        "original_dim": original_dim,
        "encoded_dim": encoded_dim,
        "final_dim": n_components,
        "feature_names_after_encoding": X.columns.tolist(),
    }


def preprocess_kdd99(
    percent10: bool = True,
    n_components: int = 4,
    vis_samples: int = 1000,
) -> dict:
    df = load_kdd99(percent10=percent10)

    return preprocess_tabular_dataset(
        df=df,
        label_col="labels",
        label_map_fn=lambda s: s != "normal.",
        n_components=n_components,
        vis_samples=vis_samples,
        categorical_cols=["protocol_type", "service", "flag"],
        drop_cols=None,
        balance_train=True,
    )


def preprocess_cic_iot23(
    path: str | Path,
    n_components: int = 4,
    vis_samples: int = 1000,
    label_col: str = "Label",
    benign_values: tuple[str, ...] = ("benign",),
    drop_cols: list[str] | None = None,
    balance_train: bool = True,
) -> dict:
    df = load_cic_iot23(path)

    benign_values_lower = {v.lower() for v in benign_values}

    def label_map_fn(s: pd.Series) -> pd.Series:
        return ~s.astype(str).str.strip().str.lower().isin(benign_values_lower)

    return preprocess_tabular_dataset(
        df=df,
        label_col=label_col,
        label_map_fn=label_map_fn,
        n_components=n_components,
        vis_samples=vis_samples,
        categorical_cols=None,
        drop_cols=drop_cols,
        balance_train=balance_train,
    )


def timed_preprocess(
    percent10: bool = True,
    n_components: int = 4,
    vis_samples: int = 1000,
):
    start = time.time()
    result = preprocess_kdd99(
        percent10=percent10,
        n_components=n_components,
        vis_samples=vis_samples,
    )
    print(f"KDD99 preprocessing time: {time.time() - start:.2f}s")
    return result


def timed_preprocess_cic_iot23(
    path: str | Path,
    n_components: int = 4,
    vis_samples: int = 1000,
    label_col: str = "Label",
    benign_values: tuple[str, ...] = ("benign",),
    drop_cols: list[str] | None = None,
    balance_train: bool = True,
):
    start = time.time()
    result = preprocess_cic_iot23(
        path=path,
        n_components=n_components,
        vis_samples=vis_samples,
        label_col=label_col,
        benign_values=benign_values,
        drop_cols=drop_cols,
        balance_train=balance_train,
    )
    print(f"CIC IoT 23 preprocessing time: {time.time() - start:.2f}s")
    return result


def plot_dimensionality_summary(
    original_dim: int,
    encoded_dim: int,
    final_dim: int,
    title: str = "Dimensionality reduction pipeline",
):
    stages = ["Raw", "One-hot", "After PCA"]
    dims = [original_dim, encoded_dim, final_dim]

    plt.figure(figsize=(8, 5))
    plt.bar(stages, dims)
    plt.ylabel("Number of dimensions")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_explained_variance(pca: PCA, dataset = ""):
    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(evr) + 1), cumulative, marker="o")
    plt.xlabel("Number of PCA components") 
    plt.ylabel("Cumulative explained variance")
    plt.title(f"PCA explained variance - {dataset}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(
    X_pca: np.ndarray,
    y: np.ndarray,
    max_points: int = 3000,
    title: str = "Projected onto first 2 PCA components",
):
    X_plot, y_plot = take_subset(X_pca, y, n_samples=max_points)

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
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_umap_embedding(
    X: np.ndarray,
    n_neighbors: int = 10,
    min_dist: float = 0.3,
):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        low_memory=True,
        random_state=RANDOM_STATE,
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