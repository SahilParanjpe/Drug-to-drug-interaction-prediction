"""
Morgan-fingerprint DDI feature pipeline.

Feature extraction only: this file does NOT train a model and does not produce
predictions unless a previously trained compatible classifier is supplied.
Install dependencies: pip install numpy pandas joblib rdkit
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def mol_from_smiles(smiles: str):
    """Parse SMILES; raise ValueError for invalid/missing structures."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def get_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return a deterministic 0/1 Morgan fingerprint of length n_bits."""
    mol = mol_from_smiles(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    return generator.GetFingerprintAsNumPy(mol).astype(np.uint8)


def get_pair_features(smiles_a: str, smiles_b: str,
                      radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return shape (1, 2*n_bits), concatenating A then B fingerprints."""
    fp_a = get_morgan_fp(smiles_a, radius, n_bits)
    fp_b = get_morgan_fp(smiles_b, radius, n_bits)
    return np.concatenate([fp_a, fp_b]).astype(np.float32).reshape(1, -1)


def predict_pair(smiles_a: str, smiles_b: str, model_path: str):
    """
    Predict with a previously trained sklearn-compatible classifier.
    Class label 1 must mean 'interaction'. No random/simulated output is used.
    """
    import joblib

    model = joblib.load(model_path)
    X = get_pair_features(smiles_a, smiles_b)
    pred = model.predict(X)[0]
    result = {"prediction": int(pred), "features": X}
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            result["interaction_probability"] = float(
                model.predict_proba(X)[0][classes.index(1)]
            )
        else:
            result["interaction_probability"] = None
    return result


if __name__ == "__main__":
    # Replace these example SMILES with structures retrieved from your dataset.
    smiles_a = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
    smiles_b = "CC(=O)Nc1ccc(O)cc1"        # paracetamol
    features = get_pair_features(smiles_a, smiles_b)
    print("Morgan pair feature shape:", features.shape)  # (1, 4096)
    print("Feature values are deterministic; no model was trained here.")
