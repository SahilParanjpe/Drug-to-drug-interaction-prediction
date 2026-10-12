"""
Binary Drug-Drug Interaction (DDI) classification.

Expected CSV columns:
    label, smile1, smile2

Models:
    - Random Forest (RF)
    - XGBoost (XGB)

Features:
    - Morgan
    - Custom32
    - Custom32 + WL

The supplied test.csv is never sampled or used for training.
Labels 0 and 1 are used as provided by the dataset.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FEATURES = {
    "morgan": "morgan_features",
    "custom32": "custom32_features",
    "custom32_wl": "custom32_wl_features",
}

CLASSIFIERS = ("rf", "xgb")


def read_dataset(path):
    """Read and validate label, smile1, smile2."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Check the CSV path."
        )

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"label", "smile1", "smile2"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns {sorted(missing)}. "
            "Expected: label, smile1, smile2."
        )

    df = df[["label", "smile1", "smile2"]].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["smile1"] = df["smile1"].fillna("").astype(str).str.strip()
    df["smile2"] = df["smile2"].fillna("").astype(str).str.strip()

    before = len(df)
    df = df.dropna(subset=["label"])
    df = df[
        (df["smile1"] != "")
        & (df["smile2"] != "")
    ].copy()

    if len(df) != before:
        print(f"[WARN] Dropped {before - len(df)} rows with missing values.")

    if not df["label"].isin([0, 1]).all():
        invalid = sorted(df.loc[~df["label"].isin([0, 1]), "label"].unique())
        raise ValueError(
            f"Labels must be 0/1 (or 0.0/1.0). Invalid values: {invalid}"
        )

    df["label"] = df["label"].astype(int)

    if df.empty:
        raise ValueError(f"No valid rows found in {path}.")

    print(
        f"[DATA] {path}: {len(df):,} rows; "
        f"class counts={df['label'].value_counts().sort_index().to_dict()}"
    )

    return df.reset_index(drop=True)


def limit_training_rows(df, max_pairs, seed):
    """Optionally sample training rows while preserving class proportions."""
    if max_pairs is None or max_pairs <= 0 or len(df) <= max_pairs:
        print(f"[DATA] Using all {len(df):,} training rows.")
        return df

    if max_pairs < 2:
        raise ValueError("--max_pairs must be at least 2.")

    if df["label"].nunique() < 2:
        raise ValueError("Training data must contain both classes.")

    if df["label"].value_counts().min() < 2:
        raise ValueError(
            "Stratified sampling requires at least two rows per class."
        )

    sampled, _ = train_test_split(
        df,
        train_size=max_pairs,
        random_state=seed,
        stratify=df["label"],
    )

    sampled = sampled.reset_index(drop=True)

    print(
        f"[DATA] Sampled training rows: {len(sampled):,} "
        f"of {len(df):,}"
    )
    print(
        "[DATA] Sampled class counts:",
        sampled["label"].value_counts().sort_index().to_dict(),
    )

    return sampled


def make_features(df, module_name, batch_log=5000):
    """
    Extract features from each SMILES pair.

    Rows that fail feature extraction are skipped. Labels remain aligned
    with their corresponding feature vectors.
    """
    module = importlib.import_module(module_name)
    feature_fn = module.get_pair_features

    features = []
    labels = []
    valid_rows = []
    errors = []
    expected_width = None

    for i, row in enumerate(df.itertuples(index=False), start=1):
        label, smile1, smile2 = row

        try:
            feat = np.asarray(
                feature_fn(smile1, smile2),
                dtype=np.float32,
            ).reshape(-1)

            if feat.size == 0 or not np.all(np.isfinite(feat)):
                raise ValueError("Empty or non-finite feature vector")

            if expected_width is None:
                expected_width = feat.size
            elif feat.size != expected_width:
                raise ValueError(
                    f"Feature length mismatch: expected {expected_width}, "
                    f"received {feat.size}"
                )

            features.append(feat)
            labels.append(int(label))
            valid_rows.append((smile1, smile2, int(label)))

        except Exception as exc:
            if len(errors) < 10:
                errors.append((i, str(exc)))

        if i % batch_log == 0 or i == len(df):
            print(
                f"  [{module_name}] processed {i:,}/{len(df):,}; "
                f"usable={len(features):,}"
            )

    if not features:
        raise RuntimeError(
            f"No valid features produced by {module_name}. "
            "Check its get_pair_features(smile1, smile2) function."
        )

    if errors:
        print(
            f"[WARN] {module_name}: feature extraction failed for "
            f"{len(df) - len(features):,} rows. Examples: {errors[:3]}"
        )

    X = np.vstack(features).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)

    print(f"[FEATURES] {module_name}: X={X.shape}, y={y.shape}")

    return X, y, valid_rows


def make_model(name, seed):
    """Keep the existing RF and XGBoost hyperparameters."""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )

    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required. Install it using: pip install xgboost"
            ) from exc

        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported classifier: {name}")


def predict_symmetric(model, module_name, valid_test_rows, batch_size=1024):
    """
    Average predictions for (A, B) and (B, A).

    This is appropriate when the target interaction is symmetric.
    """
    module = importlib.import_module(module_name)
    positive_index = list(model.classes_).index(1)

    probabilities = []

    for start in range(0, len(valid_test_rows), batch_size):
        batch = valid_test_rows[start:start + batch_size]

        forward = []
        reverse = []

        for smile1, smile2, _ in batch:
            forward.append(
                np.asarray(
                    module.get_pair_features(smile1, smile2),
                    dtype=np.float32,
                ).reshape(-1)
            )
            reverse.append(
                np.asarray(
                    module.get_pair_features(smile2, smile1),
                    dtype=np.float32,
                ).reshape(-1)
            )

        X_forward = np.vstack(forward)
        X_reverse = np.vstack(reverse)

        p_forward = model.predict_proba(X_forward)[:, positive_index]
        p_reverse = model.predict_proba(X_reverse)[:, positive_index]

        probabilities.extend((p_forward + p_reverse) / 2.0)

    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(int)

    return predictions, probabilities


def evaluate(y_true, y_pred, probabilities):
    has_two_classes = len(np.unique(y_true)) == 2

    result = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if has_two_classes else None
        ),
        "average_precision": (
            float(average_precision_score(y_true, probabilities))
            if has_two_classes else None
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["label_0", "label_1"],
            output_dict=True,
            zero_division=0,
        ),
        "threshold": 0.5,
        "prediction_method": "average of forward and reverse pair probabilities",
    }

    return result


def run(args):
    train_df = read_dataset(args.train)
    test_df = read_dataset(args.test)
    if args.max_test_pairs is not None:
        test_df = test_df.head(args.max_test_pairs).copy()

    train_df = limit_training_rows(
        train_df, args.max_pairs, args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "task": "binary drug-drug interaction classification",
        "train_csv": str(Path(args.train).resolve()),
        "test_csv": str(Path(args.test).resolve()),
        "label_meaning": (
            "Labels 0 and 1 are used as supplied. Confirm their meaning "
            "from the dataset documentation."
        ),
        "seed": args.seed,
        "features": {},
    }

    for feature_name in args.features:
        module_name = FEATURES[feature_name]

        print(
            f"\n{'=' * 70}\n"
            f"FEATURES: {feature_name}\n"
            f"{'=' * 70}"
        )

        X_train, y_train, _ = make_features(
            train_df,
            module_name,
        )

        X_test, y_test, valid_test_rows = make_features(
            test_df,
            module_name,
        )

        if set(np.unique(y_train)) != {0, 1}:
            raise ValueError(
                f"{feature_name}: training data must contain both classes."
            )

        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"{feature_name}: train/test feature dimensions differ: "
                f"{X_train.shape[1]} vs {X_test.shape[1]}"
            )

        summary["features"][feature_name] = {}

        for classifier_name in args.classifiers:
            model_name = f"{feature_name}_{classifier_name}"

            print(f"\n[TRAIN] {model_name}")
            start = time.time()

            model = make_model(classifier_name, args.seed)
            model.fit(X_train, y_train)

            y_pred, probabilities = predict_symmetric(
                model,
                module_name,
                valid_test_rows,
            )

            metrics = evaluate(y_test, y_pred, probabilities)
            metrics.update({
                "model": model_name,
                "feature_module": module_name,
                "classifier": classifier_name,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "feature_count": int(X_train.shape[1]),
                "training_and_evaluation_seconds": round(
                    time.time() - start, 2
                ),
            })

            model_path = output_dir / f"{model_name}.joblib"
            joblib.dump({
                "model": model,
                "feature_module": module_name,
                "feature_name": feature_name,
                "classifier": classifier_name,
                "label_meaning": summary["label_meaning"],
            }, model_path)

            (output_dir / f"{model_name}_metrics.json").write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )

            summary["features"][feature_name][classifier_name] = metrics

            print(
                f"[RESULT] {model_name}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']}"
            )
            print(
                "Confusion matrix [[TN, FP], [FN, TP]]:\n",
                np.asarray(metrics["confusion_matrix_labels_0_1"]),
            )
            print(f"[SAVED] {model_path}")

    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\n[DONE] Summary saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train binary DDI models using train.csv and test.csv."
    )

    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument(
        "--output-dir", default="models/binary_ddi"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Optional maximum training rows. 0 or omitted means use all.",
    )
    parser.add_argument("--max_test_pairs", type=int, default=None,
                      help="Maximum number of test pairs to use")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=tuple(FEATURES),
        default=list(FEATURES),
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=CLASSIFIERS,
        default=list(CLASSIFIERS),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())