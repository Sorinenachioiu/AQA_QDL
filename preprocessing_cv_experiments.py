from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_kddcup99
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
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
        raise ValueError(f"Path does not exist: {path}")
    if path.is_file():
        return pd.read_csv(path, low_memory=False)
    if path.is_dir():
        files = sorted(path.glob("Merged*.csv"))
        if not files:
            raise ValueError(f"No files matching 'Merged*.csv' found in folder: {path}")
        dfs = [pd.read_csv(f, low_memory=False) for f in files]
        return pd.concat(dfs, ignore_index=True)
    raise ValueError(f"Unsupported path: {path}")


def kdd99_binary_label_map(s: pd.Series) -> pd.Series:
    return (s != "normal.").astype(int)


def benign_binary_label_map(benign_values: Sequence[str]) -> Callable[[pd.Series], pd.Series]:
    benign_values_lower = {v.lower() for v in benign_values}
    def _map(s: pd.Series) -> pd.Series:
        return (~s.astype(str).str.strip().str.lower().isin(benign_values_lower)).astype(int)
    return _map


def make_fixed_stratified_subset(
    X_df: pd.DataFrame,
    y: np.ndarray,
    total_size: int = 15000,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, np.ndarray]:
    y = np.asarray(y)
    if total_size >= len(X_df):
        return X_df.reset_index(drop=True).copy(), y.copy()

    X_sub, _, y_sub, _ = train_test_split(
        X_df,
        y,
        train_size=total_size,
        random_state=random_state,
        stratify=y,
    )
    return X_sub.reset_index(drop=True), np.asarray(y_sub)


def make_shared_kfolds(
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    shuffle: bool = True,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds.append((fold_id, train_idx, test_idx))
    return folds


def _balance_binary_training_set(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    train_df = X_train.copy()
    train_df["__label__"] = np.asarray(y_train)
    class_counts = train_df["__label__"].value_counts()
    if len(class_counts) != 2:
        raise ValueError(f"Expected binary labels, got {class_counts.to_dict()}")

    maj = class_counts.idxmax()
    mino = class_counts.idxmin()

    maj_df = train_df[train_df["__label__"] == maj]
    min_df = train_df[train_df["__label__"] == mino]

    maj_down = resample(
        maj_df,
        replace=False,
        n_samples=len(min_df),
        random_state=random_state,
    )

    train_bal = pd.concat([maj_down, min_df], axis=0)
    train_bal = train_bal.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    Xb = train_bal.drop(columns=["__label__"])
    yb = train_bal["__label__"].to_numpy()
    return Xb, yb


@dataclass
class FoldPreprocessorConfig:
    n_components: int = 4
    categorical_cols: Optional[List[str]] = None
    drop_cols: Optional[List[str]] = None
    clip_value: float = 1e12
    balance_train: bool = True
    random_state: int = RANDOM_STATE


class FoldPreprocessor:
    def __init__(self, config: FoldPreprocessorConfig):
        self.config = config
        self.train_columns_: Optional[List[str]] = None
        self.valid_columns_: Optional[List[str]] = None
        self.imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[MinMaxScaler] = None
        self.pca_: Optional[PCA] = None
        self.original_dim_: Optional[int] = None
        self.encoded_dim_: Optional[int] = None

    def _drop_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.config.drop_cols:
            cols = [c for c in self.config.drop_cols if c in X.columns]
            X = X.drop(columns=cols)
        return X

    def _choose_categoricals(self, X: pd.DataFrame) -> List[str]:
        if self.config.categorical_cols is not None:
            return [c for c in self.config.categorical_cols if c in X.columns]
        return X.select_dtypes(include=["object", "category"]).columns.tolist()

    def _one_hot(self, X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        return X

    def _clean_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._drop_cols(X)
        self.original_dim_ = X.shape[1]
        cat_cols = self._choose_categoricals(X)
        X = self._one_hot(X, cat_cols)

        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            X = X.drop(columns=all_nan_cols)

        self.train_columns_ = list(X.columns)

        self.imputer_ = SimpleImputer(strategy="constant", fill_value=0.0)
        arr = self.imputer_.fit_transform(X)
        X = pd.DataFrame(arr, columns=self.train_columns_, index=X.index)

        X = X.clip(lower=-self.config.clip_value, upper=self.config.clip_value)

        std = X.std(axis=0, ddof=0)
        self.valid_columns_ = std[std > 0].index.tolist()
        if not self.valid_columns_:
            raise ValueError("Data has only const values")
        X = X[self.valid_columns_]

        self.encoded_dim_ = X.shape[1]
        return X

    def _clean_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.train_columns_ is None or self.valid_columns_ is None or self.imputer_ is None:
            raise RuntimeError("Preprocessor must be fit before transform.")
        X = self._drop_cols(X)
        cat_cols = self._choose_categoricals(X)
        X = self._one_hot(X, cat_cols)
        X = X.reindex(columns=self.train_columns_, fill_value=np.nan)
        arr = self.imputer_.transform(X)
        X = pd.DataFrame(arr, columns=self.train_columns_, index=X.index)
        X = X.clip(lower=-self.config.clip_value, upper=self.config.clip_value)
        X = X[self.valid_columns_]
        return X

    def fit_transform(self, X_train_df: pd.DataFrame, y_train: np.ndarray) -> Dict[str, np.ndarray]:
        X_train_df = self._clean_fit(X_train_df)

        if self.config.balance_train:
            X_train_df, y_train = _balance_binary_training_set(
                X_train_df, y_train, random_state=self.config.random_state
            )

        self.scaler_ = MinMaxScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train_df).astype(np.float32)

        self.pca_ = PCA(n_components=self.config.n_components, random_state=self.config.random_state)
        X_train_pca = self.pca_.fit_transform(X_train_scaled).astype(np.float32)

        return {
            "X_train_scaled": X_train_scaled,
            "X_train_pca": X_train_pca,
            "y_train": np.asarray(y_train),
        }

    def transform(self, X_test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if self.scaler_ is None or self.pca_ is None:
            raise RuntimeError("Preprocessor must be fit before transform.")
        X_test_df = self._clean_transform(X_test_df)
        X_test_scaled = self.scaler_.transform(X_test_df).astype(np.float32)
        X_test_pca = self.pca_.transform(X_test_scaled).astype(np.float32)
        return {
            "X_test_scaled": X_test_scaled,
            "X_test_pca": X_test_pca,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
    }


def summarize_metrics(
    results_df: pd.DataFrame,
    metric_cols: Sequence[str] = ("accuracy", "precision", "recall", "f1", "roc_auc"),
) -> pd.DataFrame:
    rows = []
    for m in metric_cols:
        mean = results_df[m].mean()
        std = results_df[m].std(ddof=1)
        rows.append({"metric": m, "mean": mean, "std": std, "mean ± std": f"{mean:.4f} ± {std:.4f}"})
    return pd.DataFrame(rows)


def make_inner_validation_split(
    y_train: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_sub_idx, val_idx = next(splitter.split(np.zeros(len(y_train)), y_train))
    return train_sub_idx, val_idx


def run_linear_pca2_on_shared_folds(
    X_df: pd.DataFrame,
    y: np.ndarray,
    folds: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    preproc_config: FoldPreprocessorConfig,
) -> pd.DataFrame:
    rows = []
    for fold_id, train_idx, test_idx in folds:
        fp = FoldPreprocessor(preproc_config)
        train_pack = fp.fit_transform(X_df.iloc[train_idx], y[train_idx])
        test_pack = fp.transform(X_df.iloc[test_idx])

        X_train_pca2 = train_pack["X_train_pca"][:, :2]
        X_test_pca2 = test_pack["X_test_pca"][:, :2]
        y_train = train_pack["y_train"]
        y_test = y[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=preproc_config.random_state)
        clf.fit(X_train_pca2, y_train)
        y_pred = clf.predict(X_test_pca2)
        y_prob = clf.predict_proba(X_test_pca2)[:, 1]

        row = {"fold": fold_id + 1, "model": "PCA-2 + LogisticRegression"}
        row.update(compute_metrics(y_test, y_pred, y_prob))
        rows.append(row)
    return pd.DataFrame(rows)


def run_svm_pca4_on_shared_folds(
    X_df: pd.DataFrame,
    y: np.ndarray,
    folds: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    preproc_config: FoldPreprocessorConfig,
) -> pd.DataFrame:
    rows = []
    for fold_id, train_idx, test_idx in folds:
        fp = FoldPreprocessor(preproc_config)
        train_pack = fp.fit_transform(X_df.iloc[train_idx], y[train_idx])
        test_pack = fp.transform(X_df.iloc[test_idx])

        X_train = train_pack["X_train_pca"]
        X_test = test_pack["X_test_pca"]
        y_train = train_pack["y_train"]
        y_test = y[test_idx]

        clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=preproc_config.random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        row = {"fold": fold_id + 1, "model": "PCA-4 + SVM"}
        row.update(compute_metrics(y_test, y_pred, y_prob))
        rows.append(row)
    return pd.DataFrame(rows)
