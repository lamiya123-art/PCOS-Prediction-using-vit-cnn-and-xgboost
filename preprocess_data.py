"""
Preprocess tabular PCOS data: impute, scale, and persist transformer.

Usage:
  python preprocess_data.py --data_dir data --out_dir data/tabular_processed

Outputs:
  - data/tabular_processed/train.npy, val.npy, test.npy (features-only, float32)
  - data/tabular_processed/labels_{split}.npy (labels, int64)
  - artifacts/tabular_preprocessor.pkl (sklearn pipeline)
  - artifacts/feature_names.json
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def _detect_sep(sample: str) -> str:
    # prefer tab if present; else comma
    if "\t" in sample and sample.count("\t") >= sample.count(","):
        return "\t"
    return ","


def load_split_csv(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / "tabular" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    # Peek first non-empty line to detect delimiter
    with path.open("r", encoding="utf-8") as f:
        first_nonempty = ""
        for line in f:
            if line.strip():
                first_nonempty = line
                break
    if not first_nonempty:
        raise ValueError(f"CSV is empty: {path}")
    sep = _detect_sep(first_nonempty)
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] == 0:
        raise ValueError(f"No columns parsed from {path}. Check delimiter and file contents.")
    if len(df) == 0:
        raise ValueError(f"No rows found in {path}. It must contain at least one data row.")
    return df


def infer_numeric_columns(df: pd.DataFrame, label_col: str) -> List[str]:
    candidates = [c for c in df.columns if c != label_col]
    numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


def build_numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )


def preprocess(data_dir: str, out_dir: str, label_col: str = "label"):
    data_dir_p = Path(data_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load train first (fit on train only)
    train_df = load_split_csv(data_dir_p, "train")
    if label_col not in train_df.columns:
        raise ValueError(f"CSV must include '{label_col}' column")

    numeric_cols = infer_numeric_columns(train_df, label_col)
    if not numeric_cols:
        raise ValueError("No numeric feature columns found to preprocess.")

    transformer = ColumnTransformer(
        transformers=[
            ("num", build_numeric_pipeline(), numeric_cols),
        ],
        remainder="drop",
        n_jobs=None,
    )

    X_train = train_df[numeric_cols]
    y_train = train_df[label_col].astype("int64").to_numpy()

    X_train_proc = transformer.fit_transform(X_train).astype(np.float32)

    np.save(out_dir_p / "train.npy", X_train_proc)
    np.save(out_dir_p / "labels_train.npy", y_train)

    # Persist artifacts
    joblib.dump(transformer, artifacts_dir / "tabular_preprocessor.pkl")
    with (artifacts_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(numeric_cols, f, indent=2)

    # Process val/test using the same transformer
    for split in ["val", "test"]:
        try:
            df = load_split_csv(data_dir_p, split)
        except FileNotFoundError:
            continue
        if label_col not in df.columns:
            raise ValueError(f"{split}.csv must include '{label_col}' column")
        X = df[numeric_cols]
        y = df[label_col].astype("int64").to_numpy()
        X_proc = transformer.transform(X).astype(np.float32)
        np.save(out_dir_p / f"{split}.npy", X_proc)
        np.save(out_dir_p / f"labels_{split}.npy", y)

    print("✅ Preprocessing complete:")
    print(f"  Features saved to: {out_dir_p}")
    print(f"  Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PCOS tabular CSVs")
    parser.add_argument("--data_dir", type=str, default="data", help="Input data directory")
    parser.add_argument("--out_dir", type=str, default="data/tabular_processed", help="Output directory for processed arrays")
    parser.add_argument("--label_col", type=str, default="label", help="Name of the label column")
    args = parser.parse_args()
    preprocess(args.data_dir, args.out_dir, args.label_col)


