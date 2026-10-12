"""
Custom 32-feature + Weisfeiler-Lehman (WL) pair-feature pipeline.

Per drug: 32 custom features (10 BFS/DFS graph + 10 atom counts + 12 descriptors).
Per pair: 64 base features + 16 WL pair-comparison features = 80 total.

The 16 WL values are four comparisons at each of four WL rounds:
  cosine similarity of label-count histograms,
  weighted Jaccard similarity,
  normalized L1 similarity,
  fraction of unique labels shared.
Atom labels include atomic number, formal charge, degree, aromaticity, and
hydrogen count. Neighbor labels are refined iteratively.

Install dependencies: pip install numpy joblib rdkit
This file extracts features; it does not train a classifier.
"""

from collections import Counter, deque
from hashlib import sha256
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I")


def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def adjacency_list(mol):
    adj = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[a].append(b)
        adj[b].append(a)
    return adj


def bfs_component_and_distances(adj, start, seen):
    queue = deque([start])
    seen.add(start)
    distances = {start: 0}
    vertices = []
    while queue:
        u = queue.popleft()
        vertices.append(u)
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                distances[v] = distances[u] + 1
                queue.append(v)
    return vertices, distances


def bfs_graph_metrics(mol):
    n = mol.GetNumAtoms()
    if n == 0:
        return [0.0] * 4
    adj = adjacency_list(mol)
    seen, components = set(), 0
    for start in range(n):
        if start not in seen:
            bfs_component_and_distances(adj, start, seen)
            components += 1
    all_distances, diameter, max_eccentricity = [], 0, 0
    for start in range(n):
        queue = deque([start])
        dist = {start: 0}
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        ecc = max(dist.values()) if dist else 0
        diameter = max(diameter, ecc)
        max_eccentricity = max(max_eccentricity, ecc)
        all_distances.extend(d for node, d in dist.items() if node != start)
    return [
        float(components), float(diameter),
        float(np.mean(all_distances)) if all_distances else 0.0,
        float(max_eccentricity)
    ]


def dfs_graph_metrics(mol):
    n = mol.GetNumAtoms()
    if n == 0:
        return [0.0] * 6
    adj = adjacency_list(mol)
    visited, depths, max_depth = set(), [], 0
    for start in range(n):
        if start in visited:
            continue
        stack = [(start, 0)]
        visited.add(start)
        while stack:
            u, depth = stack.pop()
            depths.append(depth)
            max_depth = max(max_depth, depth)
            for v in reversed(adj[u]):
                if v not in visited:
                    visited.add(v)
                    stack.append((v, depth + 1))
    degrees = [len(x) for x in adj]
    return [
        float(max_depth), float(np.mean(depths)) if depths else 0.0,
        float(sum(d == 1 for d in degrees)),
        float(sum(d >= 3 for d in degrees)),
        float(max(degrees)) if degrees else 0.0,
        float(np.mean(degrees)) if degrees else 0.0
    ]


def extract_graph_features(mol):
    return bfs_graph_metrics(mol) + dfs_graph_metrics(mol)


def extract_atom_features(mol):
    counts = {element: 0 for element in ELEMENTS}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in counts:
            counts[atom.GetSymbol()] += 1
    halogens = counts["F"] + counts["Cl"] + counts["Br"] + counts["I"]
    return [float(counts[e]) for e in ELEMENTS] + [float(halogens)]


def extract_chem_descriptors(mol):
    return [
        float(Descriptors.MolWt(mol)), float(Descriptors.MolLogP(mol)),
        float(Descriptors.TPSA(mol)), float(rdMolDescriptors.CalcNumHBD(mol)),
        float(rdMolDescriptors.CalcNumHBA(mol)),
        float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        float(Chem.rdmolops.GetFormalCharge(mol)),
        float(rdMolDescriptors.CalcFractionCSP3(mol)),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        float(Descriptors.MolMR(mol)),
        float(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
    ]


def get_32_features(smiles):
    mol = mol_from_smiles(smiles)
    result = np.asarray(
        extract_graph_features(mol) + extract_atom_features(mol)
        + extract_chem_descriptors(mol), dtype=np.float32
    )
    if result.shape != (32,):
        raise RuntimeError(f"Expected 32 features, got {result.shape}")
    return result


def _stable_label(value: str) -> str:
    """Stable digest so WL labels are reproducible across runs."""
    return sha256(value.encode("utf-8")).hexdigest()[:24]


def wl_label_histograms(mol, iterations=3):
    """
    Return label-count histograms for rounds 0..iterations.
    Each round refines atom labels from the atom's current label and sorted
    neighboring labels. Labels are stable across molecules and Python runs.
    """
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    adj = adjacency_list(mol)
    labels = []
    for atom in mol.GetAtoms():
        initial = (
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetDegree(),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(includeNeighbors=True),
            int(atom.IsInRing()),
        )
        labels.append(_stable_label(repr(initial)))

    histograms = [Counter(labels)]
    for _ in range(iterations):
        new_labels = []
        for i, label in enumerate(labels):
            neighbor_labels = sorted(labels[j] for j in adj[i])
            signature = repr((label, tuple(neighbor_labels)))
            new_labels.append(_stable_label(signature))
        labels = new_labels
        histograms.append(Counter(labels))
    return histograms


def _compare_histograms(hist_a, hist_b):
    """Four deterministic similarity features for two WL label histograms."""
    keys = set(hist_a) | set(hist_b)
    if not keys:
        return [1.0, 1.0, 1.0, 1.0]
    a = np.asarray([hist_a.get(k, 0) for k in keys], dtype=float)
    b = np.asarray([hist_b.get(k, 0) for k in keys], dtype=float)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / denom) if denom else 0.0

    minima = float(np.minimum(a, b).sum())
    maxima = float(np.maximum(a, b).sum())
    weighted_jaccard = minima / maxima if maxima else 1.0

    l1 = float(np.abs(a - b).sum())
    l1_scale = float(a.sum() + b.sum())
    normalized_l1_similarity = 1.0 - (l1 / l1_scale) if l1_scale else 1.0

    unique_a = {k for k, v in hist_a.items() if v > 0}
    unique_b = {k for k, v in hist_b.items() if v > 0}
    unique_union = unique_a | unique_b
    shared_label_fraction = (
        len(unique_a & unique_b) / len(unique_union) if unique_union else 1.0
    )
    return [cosine, weighted_jaccard, normalized_l1_similarity,
            float(shared_label_fraction)]


def calculate_wl_kernel_features(smiles_a, smiles_b, iterations=3):
    """
    Return 16 fixed-length pair features: 4 histogram similarities x 4 WL rounds.
    Default iterations=3 gives rounds 0, 1, 2, 3 => 4*4 = 16 values.
    """
    if iterations != 3:
        raise ValueError("This function returns exactly 16 features; use iterations=3.")
    mol_a, mol_b = mol_from_smiles(smiles_a), mol_from_smiles(smiles_b)
    hist_a = wl_label_histograms(mol_a, iterations=iterations)
    hist_b = wl_label_histograms(mol_b, iterations=iterations)
    features = []
    for round_a, round_b in zip(hist_a, hist_b):
        features.extend(_compare_histograms(round_a, round_b))
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (16,):
        raise RuntimeError(f"Expected 16 WL features, got {result.shape}")
    return result


def get_pair_features(smiles_a, smiles_b):
    """Return shape (1, 80): 32 A + 32 B + 16 WL pair features."""
    base = np.concatenate([get_32_features(smiles_a), get_32_features(smiles_b)])
    wl = calculate_wl_kernel_features(smiles_a, smiles_b)
    result = np.concatenate([base, wl]).astype(np.float32).reshape(1, -1)
    if result.shape != (1, 80):
        raise RuntimeError(f"Expected pair shape (1, 80), got {result.shape}")
    return result


def predict_pair(smiles_a, smiles_b, model_path):
    """Use a trained sklearn-compatible classifier; never returns random output."""
    import joblib
    model = joblib.load(model_path)
    X = get_pair_features(smiles_a, smiles_b)
    pred = model.predict(X)[0]
    result = {"prediction": int(pred), "features": X}
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        result["interaction_probability"] = (
            float(model.predict_proba(X)[0][classes.index(1)]) if 1 in classes else None
        )
    return result


if __name__ == "__main__":
    smiles_a = "CC(=O)OC1=CC=CC=C1C(=O)O"
    smiles_b = "CC(=O)Nc1ccc(O)cc1"
    print("32 features per drug:", get_32_features(smiles_a).shape)
    print("16 WL pair features:", calculate_wl_kernel_features(smiles_a, smiles_b).shape)
    print("Full pair feature shape:", get_pair_features(smiles_a, smiles_b).shape)  # (1, 80)
