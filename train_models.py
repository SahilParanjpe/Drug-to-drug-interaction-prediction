"""
Train and Serialize RF and SVM Models for Drug-Drug Interaction (DDI) Prediction
Level 1: Pharmacokinetic (PK, class 0) vs. Pharmacodynamic (PD, class 1)
"""

import time
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

def main():
    print("=" * 70)
    print("DDI-HierPred: Training RF and SVM Models for Step 6 Pipeline")
    print("=" * 70)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # 1. Load SMILES
    print("\n[1/5] Loading Canonical DrugBank SMILES database...")
    smiles_df = pd.read_csv("Final_Canonical_Full_DrugBank_SMILES.csv")
    print(f"Loaded {len(smiles_df)} drug records.")

    # 2. Compute Morgan Fingerprints (radius=2, 2048 bits)
    print("\n[2/5] Computing 2048-bit Morgan Fingerprints...")
    t0 = time.time()
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=True
    )

    drug_fps = {}
    for _, row in smiles_df.iterrows():
        drug_id = str(row["DrugBank_ID"]).strip().upper()
        smiles = str(row["SMILES"]).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            drug_fps[drug_id] = morgan_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)

    print(f"Successfully generated fingerprints for {len(drug_fps)} drugs in {time.time()-t0:.2f}s.")
    joblib.dump(drug_fps, models_dir / "drug_fps.joblib")
    print(f"Saved fingerprint cache to {models_dir / 'drug_fps.joblib'}")

    # 3. Load Level 1 Training and Testing datasets
    print("\n[3/5] Loading Level 1 Train and Test drug pairs...")
    train_df = pd.read_csv("DDI-HierPred_Dataset/Level1_Train_Pairs.csv")
    test_df = pd.read_csv("DDI-HierPred_Dataset/Level1_Final_Test_Pairs.csv")

    # Sample for robust & fast training (30,000 train, 5,000 test)
    sample_train_size = min(30000, len(train_df))
    sample_test_size = min(5000, len(test_df))
    train_sample = train_df.sample(n=sample_train_size, random_state=42)
    test_sample = test_df.sample(n=sample_test_size, random_state=42)

    def extract_pair_features(pairs_df):
        X, y = [], []
        for _, row in pairs_df.iterrows():
            d1 = str(row["Pair_Drug_A"]).strip().upper()
            d2 = str(row["Pair_Drug_B"]).strip().upper()
            if d1 in drug_fps and d2 in drug_fps:
                # Concatenate 2048 + 2048 = 4096 bits
                feat = np.concatenate([drug_fps[d1], drug_fps[d2]])
                X.append(feat)
                y.append(int(row["Target"]))
        return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int8)

    print("Constructing 4096-dimensional drug-pair feature matrices...")
    X_train, y_train = extract_pair_features(train_sample)
    X_test, y_test = extract_pair_features(test_sample)
    print(f"X_train: {X_train.shape}, Class distribution: PK={np.sum(y_train==0)}, PD={np.sum(y_train==1)}")
    print(f"X_test:  {X_test.shape}, Class distribution: PK={np.sum(y_test==0)}, PD={np.sum(y_test==1)}")

    # 4. Train Support Vector Machine (LinearSVC with Platt Sigmoid Calibration)
    print("\n[4/5] Training Calibrated Support Vector Machine (SVM)...")
    t_start = time.time()
    base_svm = LinearSVC(C=1.0, dual="auto", max_iter=3000, random_state=42)
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=3)
    calibrated_svm.fit(X_train, y_train)
    svm_train_time = time.time() - t_start

    svm_preds = calibrated_svm.predict(X_test)
    svm_probs = calibrated_svm.predict_proba(X_test)[:, 1]
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_auc = roc_auc_score(y_test, svm_probs)
    svm_f1 = f1_score(y_test, svm_preds)
    print(f"SVM Training Completed in {svm_train_time:.2f}s")
    print(f"  - Accuracy: {svm_acc:.4f}")
    print(f"  - ROC-AUC:  {svm_auc:.4f}")
    print(f"  - F1 Score: {svm_f1:.4f}")

    joblib.dump(calibrated_svm, models_dir / "svm_calibrated_model.joblib")
    print(f"Saved SVM model to {models_dir / 'svm_calibrated_model.joblib'}")

    # 5. Train Random Forest Classifier
    print("\n[5/5] Training Random Forest (RF) Classifier...")
    t_start = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t_start

    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_f1 = f1_score(y_test, rf_preds)
    print(f"RF Training Completed in {rf_train_time:.2f}s")
    print(f"  - Accuracy: {rf_acc:.4f}")
    print(f"  - ROC-AUC:  {rf_auc:.4f}")
    print(f"  - F1 Score: {rf_f1:.4f}")

    joblib.dump(rf, models_dir / "rf_model.joblib")
    print(f"Saved RF model to {models_dir / 'rf_model.joblib'}")

    # Save summary metadata
    metrics = {
        "dataset": "DDI-HierPred Level 1 (PK vs PD)",
        "features": "Morgan Fingerprints (radius=2, bits=2048, chirality=True, pair_dim=4096)",
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "svm": {
            "model": "CalibratedClassifierCV(LinearSVC)",
            "accuracy": round(svm_acc, 4),
            "roc_auc": round(svm_auc, 4),
            "f1": round(svm_f1, 4),
            "train_time_sec": round(svm_train_time, 2)
        },
        "rf": {
            "model": "RandomForestClassifier(n_estimators=100, max_depth=20)",
            "accuracy": round(rf_acc, 4),
            "roc_auc": round(rf_auc, 4),
            "f1": round(rf_f1, 4),
            "train_time_sec": round(rf_train_time, 2)
        }
    }

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetadata saved to {models_dir / 'metrics.json'}")
    print("=" * 70)
    print("Training successfully completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
