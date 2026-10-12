"""
Custom 32-feature DDI pipeline with real BFS/DFS graph measurements.

Feature layout per molecule:
  10 graph features + 10 element-count features + 12 RDKit descriptors = 32.
Pair vector: [features(A), features(B)] => 64 values.

Install dependencies: pip install numpy joblib rdkit
This file extracts features; it does not train a classifier.
"""

from collections import deque
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
    """Return an undirected atom-index adjacency list."""
    adj = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[a].append(b)
        adj[b].append(a)
    return adj


def bfs_component_and_distances(adj, start, seen):
    """BFS one connected component; returns vertices and shortest-path distances."""
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
    """
    Four BFS-derived metrics:
      component count, graph diameter (largest finite shortest path),
      mean finite shortest-path distance, maximum BFS eccentricity.
    Distances are measured in bonds. Disconnected pairs are excluded.
    """
    n = mol.GetNumAtoms()
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0]
    adj = adjacency_list(mol)
    seen = set()
    components = []
    for start in range(n):
        if start not in seen:
            vertices, _ = bfs_component_and_distances(adj, start, seen)
            components.append(vertices)

    all_distances = []
    diameter = 0
    max_eccentricity = 0
    # BFS from every atom; only within-component distances are counted.
    for start in range(n):
        queue = deque([start])
        dist = {start: 0}
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if dist:
            eccentricity = max(dist.values())
            max_eccentricity = max(max_eccentricity, eccentricity)
            diameter = max(diameter, eccentricity)
            all_distances.extend(d for node, d in dist.items() if node != start)

    mean_distance = float(np.mean(all_distances)) if all_distances else 0.0
    return [
        float(len(components)),
        float(diameter),
        mean_distance,
        float(max_eccentricity),
    ]


def dfs_graph_metrics(mol):
    """
    Six DFS/degree-derived metrics:
      maximum DFS tree depth, mean DFS tree depth, leaf atom count,
      branching atom count (degree >= 3), maximum atom degree,
      mean atom degree.
    DFS depths restart at zero for each disconnected component.
    """
    n = mol.GetNumAtoms()
    if n == 0:
        return [0.0] * 6
    adj = adjacency_list(mol)
    visited = set()
    depths = []
    max_depth = 0

    # Iterative DFS avoids Python recursion-depth limits.
    for start in range(n):
        if start in visited:
            continue
        stack = [(start, 0)]
        visited.add(start)
        while stack:
            u, depth = stack.pop()
            depths.append(depth)
            max_depth = max(max_depth, depth)
            # Reverse for stable traversal order.
            for v in reversed(adj[u]):
                if v not in visited:
                    visited.add(v)
                    stack.append((v, depth + 1))

    degrees = [len(neighbors) for neighbors in adj]
    leaf_count = sum(degree == 1 for degree in degrees)
    branching_count = sum(degree >= 3 for degree in degrees)
    return [
        float(max_depth),
        float(np.mean(depths)) if depths else 0.0,
        float(leaf_count),
        float(branching_count),
        float(max(degrees)) if degrees else 0.0,
        float(np.mean(degrees)) if degrees else 0.0,
    ]


def extract_graph_features(mol):
    """Return exactly 10 computed graph features (4 BFS + 6 DFS/degree)."""
    return bfs_graph_metrics(mol) + dfs_graph_metrics(mol)


def extract_atom_features(mol):
    """Return counts for C, N, O, S, P, F, Cl, Br, I and total halogens."""
    counts = {element: 0 for element in ELEMENTS}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in counts:
            counts[symbol] += 1
    total_halogens = counts["F"] + counts["Cl"] + counts["Br"] + counts["I"]
    return [float(counts[e]) for e in ELEMENTS] + [float(total_halogens)]


def extract_chem_descriptors(mol):
    """Return the same 12 RDKit descriptors as the original design."""
    return [
        float(Descriptors.MolWt(mol)),
        float(Descriptors.MolLogP(mol)),
        float(Descriptors.TPSA(mol)),
        float(rdMolDescriptors.CalcNumHBD(mol)),
        float(rdMolDescriptors.CalcNumHBA(mol)),
        float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        float(Chem.rdmolops.GetFormalCharge(mol)),
        float(rdMolDescriptors.CalcFractionCSP3(mol)),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        float(Descriptors.MolMR(mol)),
        float(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
    ]


def get_32_features(smiles: str) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    features = (
        extract_graph_features(mol)
        + extract_atom_features(mol)
        + extract_chem_descriptors(mol)
    )
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape != (32,):
        raise RuntimeError(f"Expected 32 features, got {arr.shape}")
    return arr


def get_pair_features(smiles_a: str, smiles_b: str) -> np.ndarray:
    """Return shape (1, 64), ordered as all A features followed by all B."""
    return np.concatenate(
        [get_32_features(smiles_a), get_32_features(smiles_b)]
    ).astype(np.float32).reshape(1, -1)


def predict_pair(smiles_a: str, smiles_b: str, model_path: str):
    """Use a trained sklearn-compatible classifier; never returns a random result."""
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
    a = get_32_features(smiles_a)
    pair = get_pair_features(smiles_a, smiles_b)
    print("Features per molecule:", a.shape)  # (32,)
    print("Pair feature shape:", pair.shape)   # (1, 64)
    print("Graph features [10]:", a[:10].tolist())
