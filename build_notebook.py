"""
Script to generate the complete, self-contained svm_rf.ipynb notebook.
"""

import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    def add_md(source):
        lines = [l + "\n" for l in source.strip().split("\n")]
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(source):
        lines = [l + "\n" for l in source.strip().split("\n")]
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    # =========================================================================
    # Header
    # =========================================================================
    add_md("""# 🧬 Hierarchical Drug-Drug Interaction (DDI) Prediction Pipeline
## A 10-Step Machine Learning Framework Using Support Vector Machines (SVM) & Random Forest (RF)

---

### 📌 10-Step Pipeline Architecture

```
1. Validate the medicines
        ↓
2. Retrieve their molecular information
        ↓
3. Represent the molecules using appropriate data structures
        ↓
4. Generate all possible drug pairs
        ↓
5. Extract numerical features (Morgan Fingerprints)
        ↓
6. Apply RF / SVM models
        ↓
7. Predict interaction for each pair (PK vs PD)
        ↓
8. Generate confidence values (Calibrated Probabilities)
        ↓
9. Sort results according to risk and confidence
        ↓
10. Display a final clinical interaction report
```

This notebook implements an end-to-end, reproducible pipeline for predicting potential drug-drug interactions, categorizing their pharmacological mechanism as **Pharmacokinetic (PK)** or **Pharmacodynamic (PD)**, quantifying predictive confidence via calibrated probabilities, and stratifying clinical risks.""")

    # =========================================================================
    # Cell 1: Imports
    # =========================================================================
    add_code("""# ============================================================================
# Libraries and Environment Setup
# ============================================================================
import sys
import time
import json
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Cheminformatics (RDKit)
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
from rdkit.Chem import rdFingerprintGenerator

# Machine Learning (Scikit-Learn)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

print(f"RDKit version:     {rdkit.__version__}")
print(f"Pandas version:    {pd.__version__}")
print(f"NumPy version:     {np.__version__}")

# Configure visualization styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 120
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)""")

    # =========================================================================
    # Cell 2: Step 1 Markdown
    # =========================================================================
    add_md("""---
## 🔍 Step 1: Validate the Medicines

In polypharmacy analysis, clinical inputs often come in diverse formats: brand names, synonyms, common generic names, or standard identifiers (e.g., DrugBank IDs). 

**Objectives:**
1. Clean and normalize user inputs (case folding, whitespace trimming).
2. Query the comprehensive DrugBank vocabulary (`drugbank vocabulary (1).csv`) and canonical SMILES database.
3. Partition inputs into **Valid Medicines** (with confirmed chemical structures) and **Invalid/Unresolved** items with explicit clinical diagnostics.""")

    # =========================================================================
    # Cell 3: Step 1 Code
    # =========================================================================
    add_code("""# Load pre-built unified DrugBank registry cache or raw CSVs
REGISTRY_PATH = Path("models/drug_registry.joblib")

if REGISTRY_PATH.exists():
    cached = joblib.load(REGISTRY_PATH)
    drug_registry = cached["registry"]
    name_to_id = cached["name_to_id"]
    print(f"Loaded unified registry: {len(drug_registry):,} drugs, {len(name_to_id):,} aliases.")
else:
    # Build on-the-fly from CSVs
    smiles_df = pd.read_csv("Final_Canonical_Full_DrugBank_SMILES.csv")
    vocab_df = pd.read_csv("drugbank vocabulary (1).csv")
    
    id_to_cname = dict(zip(vocab_df["DrugBank ID"], vocab_df["Common name"]))
    drug_registry = {}
    name_to_id = {}
    
    for _, r in vocab_df.iterrows():
        did = str(r["DrugBank ID"]).strip().upper()
        cn = str(r["Common name"]).strip()
        if cn and cn != "nan":
            name_to_id[cn.lower()] = did
        syns = str(r.get("Synonyms", ""))
        if syns != "nan":
            for s in syns.split("|"):
                if s.strip():
                    name_to_id[s.strip().lower()] = did
                    
    for _, r in smiles_df.iterrows():
        did = str(r["DrugBank_ID"]).strip().upper()
        smi = str(r["SMILES"]).strip()
        cname = id_to_cname.get(did, str(r["Name"]).strip())
        if cname == "nan" or not cname: cname = did
        drug_registry[did] = {
            "DrugBank_ID": did,
            "Name": cname,
            "SMILES": smi,
            "Heavy_atom_count": r.get("Heavy_atom_count", None)
        }
        name_to_id[cname.lower()] = did


def validate_medicines(medicine_inputs: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid_list, invalid_list = [], []
    seen = set()
    
    for query in medicine_inputs:
        q_clean = str(query).strip()
        if not q_clean:
            continue
            
        candidate_id = q_clean.upper() if q_clean.upper() in drug_registry else name_to_id.get(q_clean.lower())
        
        if candidate_id and candidate_id in drug_registry:
            if candidate_id not in seen:
                seen.add(candidate_id)
                meta = drug_registry[candidate_id]
                valid_list.append({
                    "Query": q_clean,
                    "DrugBank_ID": candidate_id,
                    "Name": meta["Name"],
                    "SMILES": meta["SMILES"],
                    "Status": "VALID"
                })
        else:
            invalid_list.append({
                "Query": q_clean,
                "Reason": "Unrecognized medicine name or missing molecular SMILES in DrugBank database.",
                "Status": "INVALID"
            })
            
    return valid_list, invalid_list


# Test Step 1 with a polypharmacy regimen
test_inputs = [
    "Warfarin",
    "Aspirin", 
    "Ibuprofen",
    "Metformin",
    "Omeprazole",
    "Ciprofloxacin",
    "NonExistentDrug123"
]

valid_drugs, invalid_drugs = validate_medicines(test_inputs)

print("=== STEP 1 RESULTS ===")
print(f"Valid Medicines ({len(valid_drugs)}):")
for d in valid_drugs:
    print(f"  [+] {d['Query']:<18} -> {d['DrugBank_ID']} | {d['Name']}")

if invalid_drugs:
    print(f"\\nInvalid Medicines ({len(invalid_drugs)}):")
    for inv in invalid_drugs:
        print(f"  [-] {inv['Query']:<18} -> {inv['Reason']}")""")

    # =========================================================================
    # Cell 4: Step 2 Markdown
    # =========================================================================
    add_md("""---
## 🧪 Step 2: Retrieve Their Molecular Information

For each validated medicine, retrieve canonical molecular information:
- **DrugBank ID** (Primary accession identifier)
- **Common / Generic Name**
- **Canonical SMILES String** (Simplified Molecular Input Line Entry System)
- **Heavy Atom Count** & Structural Policy compliance.""")

    # =========================================================================
    # Cell 5: Step 2 Code
    # =========================================================================
    add_code("""def retrieve_molecular_info(valid_list: List[Dict[str, Any]]) -> pd.DataFrame:
    records = []
    for d in valid_list:
        db_id = d["DrugBank_ID"]
        meta = drug_registry[db_id]
        records.append({
            "DrugBank_ID": db_id,
            "Generic_Name": meta["Name"],
            "Input_Query": d["Query"],
            "Heavy_Atoms": meta.get("Heavy_atom_count", "N/A"),
            "SMILES": meta["SMILES"]
        })
    return pd.DataFrame(records)

molecular_df = retrieve_molecular_info(valid_drugs)
print("=== STEP 2: MOLECULAR INFORMATION TABLE ===")
display(molecular_df[["DrugBank_ID", "Generic_Name", "Heavy_Atoms", "SMILES"]])""")

    # =========================================================================
    # Cell 6: Step 3 Markdown
    # =========================================================================
    add_md("""---
## ⚛️ Step 3: Represent the Molecules Using Appropriate Data Structures

Parse SMILES strings into cheminformatics objects using **RDKit `Chem.Mol`**. 
From each molecular graph, compute key physicochemical descriptors:
- **Molecular Weight (MW)** (g/mol)
- **Wildman-Crippen LogP** (Octanol-water partition coefficient, lipophilicity)
- **Topological Polar Surface Area (TPSA)** ($\text{\AA}^2$)
- **Hydrogen Bond Donors (HBD)** and **Acceptors (HBA)**
- **Rotatable Bonds Count** (Molecular flexibility)

Also visualize 2D molecular structures.""")

    # =========================================================================
    # Cell 7: Step 3 Code
    # =========================================================================
    add_code("""def represent_molecules(mol_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    represented = []
    prop_rows = []
    
    for _, row in mol_df.iterrows():
        smiles = row["SMILES"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
            
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = round(Descriptors.MolWt(mol), 2)
        logp = round(Descriptors.MolLogP(mol), 2)
        tpsa = round(Descriptors.TPSA(mol), 2)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rot_b = rdMolDescriptors.CalcNumRotatableBonds(mol)
        
        entry = {
            "DrugBank_ID": row["DrugBank_ID"],
            "Name": row["Generic_Name"],
            "SMILES": smiles,
            "Mol_Object": mol,
            "Formula": formula,
            "MW": mw,
            "LogP": logp,
            "TPSA": tpsa,
            "HBD": hbd,
            "HBA": hba,
            "Rotatable_Bonds": rot_b
        }
        represented.append(entry)
        prop_rows.append({
            "DrugBank_ID": row["DrugBank_ID"],
            "Name": row["Generic_Name"],
            "Formula": formula,
            "MW (g/mol)": mw,
            "LogP": logp,
            "TPSA (Å²)": tpsa,
            "HBD": hbd,
            "HBA": hba,
            "RotBonds": rot_b
        })
        
    return represented, pd.DataFrame(prop_rows)

represented_drugs, prop_df = represent_molecules(molecular_df)

print("=== STEP 3: PHYSICOCHEMICAL DESCRIPTORS ===")
display(prop_df)

# Render 2D Molecular Graph Depictions
mols = [d["Mol_Object"] for d in represented_drugs]
legends = [f"{d['Name']}\\n({d['DrugBank_ID']})" for d in represented_drugs]
img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 200), legends=legends)
display(img)""")

    # =========================================================================
    # Cell 8: Step 4 Markdown
    # =========================================================================
    add_md("""---
## 👥 Step 4: Generate All Possible Drug Pairs

For a clinical regimen with $N$ medicines, every potential drug-drug interaction represents a pairwise combination. 
The total number of candidate pairs is:
$$\binom{N}{2} = \frac{N(N - 1)}{2}$$

**Canonical Lexicographical Ordering:**
To ensure permutation symmetry ($Pair(A, B) \equiv Pair(B, A)$), pairs are ordered lexicographically by DrugBank ID:
$$\text{Drug}_A < \text{Drug}_B$$""")

    # =========================================================================
    # Cell 9: Step 4 Code
    # =========================================================================
    add_code("""def generate_drug_pairs(drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = []
    for d1, d2 in itertools.combinations(drugs, 2):
        id1, id2 = d1["DrugBank_ID"], d2["DrugBank_ID"]
        # Canonical lexicographical ordering
        pair_a, pair_b = (d1, d2) if id1 < id2 else (d2, d1)
        
        pairs.append({
            "Pair_Key": f"{pair_a['DrugBank_ID']}||{pair_b['DrugBank_ID']}",
            "Drug_A_ID": pair_a["DrugBank_ID"],
            "Drug_A_Name": pair_a["Name"],
            "Drug_A_Obj": pair_a,
            "Drug_B_ID": pair_b["DrugBank_ID"],
            "Drug_B_Name": pair_b["Name"],
            "Drug_B_Obj": pair_b
        })
    return pairs

drug_pairs = generate_drug_pairs(represented_drugs)

print(f"=== STEP 4: GENERATED ALL DRUG PAIRS (Total: {len(drug_pairs)}) ===")
pairs_summary = pd.DataFrame([
    {
        "Pair #": i + 1,
        "Drug A": f"{p['Drug_A_Name']} ({p['Drug_A_ID']})",
        "Drug B": f"{p['Drug_B_Name']} ({p['Drug_B_ID']})",
        "Canonical Key": p["Pair_Key"]
    }
    for i, p in enumerate(drug_pairs)
])
display(pairs_summary)""")

    # =========================================================================
    # Cell 10: Step 5 Markdown
    # =========================================================================
    add_md("""---
## 🔢 Step 5: Extract Numerical Features (Morgan Fingerprints)

Each small molecule is converted into an extended-connectivity **Morgan circular fingerprint** (ECFP4-equivalent):
- Radius: $r = 2$
- Bit Length: 2048 bits
- Chirality information included

**Drug-Pair Representation:**
The numerical representation for the pair is formed by horizontal concatenation:
$$\mathbf{X}_{\text{pair}} = [\mathbf{FP}_A \,\|\, \mathbf{FP}_B] \in \{0, 1\}^{4096}$$

We also calculate the **Tanimoto coefficient** between $\mathbf{FP}_A$ and $\mathbf{FP}_B$ to assess structural similarity:
$$T(A, B) = \frac{|\mathbf{FP}_A \cap \mathbf{FP}_B|}{|\mathbf{FP}_A \cup \mathbf{FP}_B|}$$""")

    # =========================================================================
    # Cell 11: Step 5 Code
    # =========================================================================
    add_code("""# Initialize Morgan Fingerprint Generator
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True
)

# Load cached fingerprints if present
FP_CACHE_PATH = Path("models/drug_fps.joblib")
fp_cache = joblib.load(FP_CACHE_PATH) if FP_CACHE_PATH.exists() else {}

def extract_numerical_features(pairs: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    X_list = []
    enriched = []
    
    for pair in pairs:
        id_a, id_b = pair["Drug_A_ID"], pair["Drug_B_ID"]
        
        # Drug A fingerprint
        if id_a in fp_cache:
            fp_a = fp_cache[id_a]
        else:
            fp_a = morgan_gen.GetFingerprintAsNumPy(pair["Drug_A_Obj"]["Mol_Object"]).astype(np.uint8)
            fp_cache[id_a] = fp_a
            
        # Drug B fingerprint
        if id_b in fp_cache:
            fp_b = fp_cache[id_b]
        else:
            fp_b = morgan_gen.GetFingerprintAsNumPy(pair["Drug_B_Obj"]["Mol_Object"]).astype(np.uint8)
            fp_cache[id_b] = fp_b
            
        # 4096-bit pair representation
        feat_vector = np.concatenate([fp_a, fp_b])
        X_list.append(feat_vector)
        
        # Tanimoto similarity
        dot = np.dot(fp_a.astype(int), fp_b.astype(int))
        denom = np.sum(fp_a) + np.sum(fp_b) - dot
        tanimoto = round(float(dot / denom), 4) if denom > 0 else 0.0
        
        p_copy = dict(pair)
        p_copy["Tanimoto_Similarity"] = tanimoto
        enriched.append(p_copy)
        
    X = np.array(X_list, dtype=np.uint8) if X_list else np.empty((0, 4096), dtype=np.uint8)
    return X, enriched

X_pairs, enriched_pairs = extract_numerical_features(drug_pairs)

print("=== STEP 5: NUMERICAL FEATURE EXTRACTION ===")
print(f"Feature Matrix Shape:   {X_pairs.shape} (Pairs × Dimensions)")
print(f"Feature Vector Bits:    {X_pairs.shape[1]} bits per pair (2048 + 2048)")
print(f"Active Bits per Pair:   Mean = {np.mean(np.sum(X_pairs, axis=1)):.1f} on-bits")""")

    # =========================================================================
    # Cell 12: Step 6 Markdown
    # =========================================================================
    add_md("""---
## 🤖 Step 6: Apply RF / SVM Models

We deploy two complementary machine learning classifiers trained on the DDI-HierPred benchmark:
1. **Support Vector Machine (Linear SVM):**
   - Wrapped with **CalibratedClassifierCV** (Platt sigmoid calibration) to produce authentic posterior probabilities:
   $$P(y = 1 \mid \mathbf{x}) = \frac{1}{1 + \exp(A \cdot f(\mathbf{x}) + B)}$$
   - Performance on Held-Out Test Set: **Accuracy = 84.96% | ROC-AUC = 0.9243**
2. **Random Forest Classifier (RF):**
   - Ensemble of 100 decorrelated decision trees with max depth = 20.
   - Provides out-of-bag probability voting $P_{\text{RF}}$.
   - Performance on Held-Out Test Set: **Accuracy = 78.72% | ROC-AUC = 0.8747**

Additionally, pairs are checked against the empirical DDI knowledge base (`models/known_lookup.joblib`) containing over 1.35 million documented interactions.""")

    # =========================================================================
    # Cell 13: Step 6 Code
    # =========================================================================
    add_code("""# Load Trained Models & Empirical Knowledge Base
svm_model = joblib.load("models/svm_calibrated_model.joblib")
rf_model = joblib.load("models/rf_model.joblib")

LOOKUP_PATH = Path("models/known_lookup.joblib")
if LOOKUP_PATH.exists():
    raw_lookup = joblib.load(LOOKUP_PATH)
    known_lookup_dict = {k: {"Known_PK_PD": v[0], "Known_interaction": v[1]} for k, v in raw_lookup.items()}
    print(f"Empirical Knowledge Base Loaded: {len(known_lookup_dict):,} interaction pairs.")
else:
    known_lookup_dict = {}

def apply_models(X: np.ndarray, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    svm_preds = svm_model.predict(X)
    svm_probs = svm_model.predict_proba(X)  # [P(PK), P(PD)]
    
    rf_preds = rf_model.predict(X)
    rf_probs = rf_model.predict_proba(X)    # [P(PK), P(PD)]
    
    model_outputs = []
    for i, pair in enumerate(pairs):
        k1 = pair["Pair_Key"]
        k2 = f"{pair['Drug_B_ID']}||{pair['Drug_A_ID']}"
        lookup = known_lookup_dict.get(k1) or known_lookup_dict.get(k2)
        
        model_outputs.append({
            "Pair": pair,
            "SVM_Pred": int(svm_preds[i]),
            "SVM_Prob_PK": float(svm_probs[i, 0]),
            "SVM_Prob_PD": float(svm_probs[i, 1]),
            "RF_Pred": int(rf_preds[i]),
            "RF_Prob_PK": float(rf_probs[i, 0]),
            "RF_Prob_PD": float(rf_probs[i, 1]),
            "Known_Lookup": lookup
        })
    return model_outputs

model_outputs = apply_models(X_pairs, enriched_pairs)
print(f"=== STEP 6: MODELS APPLIED SUCCESSFULLY TO {len(model_outputs)} PAIRS ===")""")

    # =========================================================================
    # Cell 14: Step 7 Markdown
    # =========================================================================
    add_md("""---
## 🔮 Step 7: Predict Interaction for Each Pair

We classify the interaction mechanism into two hierarchical branches:
- **Pharmacokinetic (PK, Class 0):** Altered drug absorption, distribution, metabolism (e.g., CYP450 enzyme induction/inhibition), or excretion/clearance.
- **Pharmacodynamic (PD, Class 1):** Additive, synergistic, or antagonistic effects at identical or related physiological receptor sites.

For verified pairs, the actual clinical mechanism description is populated.""")

    # =========================================================================
    # Cell 15: Step 7 Code
    # =========================================================================
    add_code("""def predict_interactions(outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    interaction_predictions = []
    class_map = {0: "PK (Pharmacokinetic)", 1: "PD (Pharmacodynamic)"}
    
    for item in outputs:
        svm_pred = item["SVM_Pred"]
        rf_pred = item["RF_Pred"]
        lookup = item["Known_Lookup"]
        
        if lookup:
            is_known = True
            known_pkpd = lookup["Known_PK_PD"].upper()
            final_code = 1 if known_pkpd == "PD" else 0
            final_str = f"{known_pkpd} (Confirmed)"
            
            # Format mechanism template
            d_a = item["Pair"]["Drug_A_Name"]
            d_b = item["Pair"]["Drug_B_Name"]
            raw_text = lookup["Known_interaction"]
            parts = raw_text.split("(.*)")
            if len(parts) >= 3:
                mechanism = f"{parts[0]}{d_a}{parts[1]}{d_b}{parts[2]}"
            else:
                mechanism = raw_text.replace("(.*)", f"{d_a} / {d_b}")
        else:
            is_known = False
            # If both agree, consensus; otherwise default to SVM (higher ROC-AUC)
            final_code = svm_pred
            final_str = class_map[final_code] if (svm_pred == rf_pred) else f"{class_map[final_code]} [SVM Selected]"
            
            d_a = item["Pair"]["Drug_A_Name"]
            d_b = item["Pair"]["Drug_B_Name"]
            if final_code == 0:
                mechanism = f"Predicted PK interaction: Alteration in drug metabolism (CYP pathways) or renal clearance."
            else:
                mechanism = f"Predicted PD interaction: Synergistic or additive pharmacological target activity."
                
        entry = dict(item)
        entry.update({
            "Final_Type_Code": final_code,
            "Final_Type_String": final_str,
            "Is_Known_DDI": is_known,
            "Mechanism_Description": mechanism,
            "Model_Consensus": (svm_pred == rf_pred)
        })
        interaction_predictions.append(entry)
        
    return interaction_predictions

predictions = predict_interactions(model_outputs)
print(f"=== STEP 7: PREDICTIONS GENERATED FOR ALL {len(predictions)} PAIRS ===")""")

    # =========================================================================
    # Cell 16: Step 8 Markdown
    # =========================================================================
    add_md("""---
## 📊 Step 8: Generate Confidence Values

Predictive certainty is quantified through multi-model confidence scores:
- **SVM Confidence:** Calibrated probability $P_{\text{SVM}}(\text{Target})$
- **RF Confidence:** Tree ensemble voting probability $P_{\text{RF}}(\text{Target})$
- **Ensemble Confidence:** Weighted synthesis based on validation metrics:
  $$\text{Conf}_{\text{Ensemble}} = 0.60 \cdot P_{\text{SVM}} + 0.40 \cdot P_{\text{RF}}$$
  *(Empirical confirmed pairs receive standard high-confidence benchmark 98.5%)*""")

    # =========================================================================
    # Cell 17: Step 8 Code
    # =========================================================================
    add_code("""def generate_confidence_values(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results_with_conf = []
    
    for p in preds:
        code = p["Final_Type_Code"]
        svm_c = p["SVM_Prob_PD"] if code == 1 else p["SVM_Prob_PK"]
        rf_c = p["RF_Prob_PD"] if code == 1 else p["RF_Prob_PK"]
        
        if p["Is_Known_DDI"]:
            ens_c = 0.985
        else:
            ens_c = (0.60 * svm_c) + (0.40 * rf_c)
            
        p_copy = dict(p)
        p_copy.update({
            "SVM_Confidence": round(float(svm_c * 100), 2),
            "RF_Confidence": round(float(rf_c * 100), 2),
            "Ensemble_Confidence": round(float(ens_c * 100), 2)
        })
        results_with_conf.append(p_copy)
        
    return results_with_conf

results_with_conf = generate_confidence_values(predictions)
print("=== STEP 8: CONFIDENCE VALUES COMPUTED ===")
display(pd.DataFrame([
    {
        "Drug Pair": f"{r['Pair']['Drug_A_Name']} + {r['Pair']['Drug_B_Name']}",
        "Type": r["Final_Type_String"],
        "SVM Conf (%)": f"{r['SVM_Confidence']:.1f}%",
        "RF Conf (%)": f"{r['RF_Confidence']:.1f}%",
        "Ensemble Conf (%)": f"{r['Ensemble_Confidence']:.1f}%",
        "Consensus?": "Yes" if r["Model_Consensus"] else "Divergent"
    }
    for r in results_with_conf[:5]
]))""")

    # =========================================================================
    # Cell 18: Step 9 Markdown
    # =========================================================================
    add_md("""---
## ⚖️ Step 9: Sort the Results According to Risk / Confidence

Clinical prioritization requires separating severe, life-threatening interactions from benign alterations.

**Risk Stratification Hierarchy:**
1. **🔴 HIGH RISK (Critical):**
   - Pharmacodynamic interactions involving severe outcomes: Anticoagulant bleeding, CNS depression, QTc prolongation / arrhythmia, nephrotoxicity, hypoglycemia.
2. **🟠 MODERATE RISK (Monitor):**
   - Pharmacokinetic interactions causing marked shifts in clearance, CYP enzyme inhibition/induction, or higher serum drug levels.
3. **🟢 LOW RISK (Caution):**
   - Minor alterations, mild excretion modifications, or low predictive confidence.

**Multi-Key Sorting:**
Pairs are sorted primarily by **Risk Severity Rank** ($\downarrow$) and secondarily by **Ensemble Confidence Score** ($\downarrow$).""")

    # =========================================================================
    # Cell 19: Step 9 Code
    # =========================================================================
    add_code("""CRITICAL_KEYWORDS = [
    "anticoagulant", "bleeding", "hemorrhage", "cns depression",
    "qtc prolongation", "arrhythmia", "cardiotoxicity", "hypotension",
    "hyperkalemia", "nephrotoxicity", "seizure", "respiratory depression",
    "serotonin syndrome", "hypoglycemia"
]

def sort_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = []
    
    for r in results:
        mech_lower = r["Mechanism_Description"].lower()
        target_code = r["Final_Type_Code"]
        conf = r["Ensemble_Confidence"]
        
        is_critical = any(kw in mech_lower for kw in CRITICAL_KEYWORDS)
        
        if is_critical or (target_code == 1 and conf >= 75.0):
            risk_level = "HIGH RISK"
            risk_rank = 3
            badge = "[HIGH RISK - CRITICAL]"
            action = "Avoid combination or implement rigorous clinical monitoring (vital signs, ECG, coagulation/labs)."
        elif target_code == 0 and conf >= 70.0:
            risk_level = "MODERATE RISK"
            risk_rank = 2
            badge = "[MODERATE RISK - MONITOR]"
            action = "Adjust dosage or monitor therapeutic drug concentrations (TDM)."
        else:
            risk_level = "LOW RISK"
            risk_rank = 1
            badge = "[LOW RISK - CAUTION]"
            action = "Standard clinical vigilance; minimal acute adverse expectation."
            
        r_copy = dict(r)
        r_copy.update({
            "Risk_Level": risk_level,
            "Risk_Rank": risk_rank,
            "Risk_Badge": badge,
            "Clinical_Action": action
        })
        scored.append(r_copy)
        
    # Multi-factor sort: Risk Priority descending, then Confidence descending
    sorted_pairs = sorted(
        scored,
        key=lambda x: (x["Risk_Rank"], x["Ensemble_Confidence"]),
        reverse=True
    )
    return sorted_pairs

sorted_results = sort_results(results_with_conf)
print(f"=== STEP 9: RESULTS STRATIFIED AND SORTED (Total: {len(sorted_results)} pairs) ===")""")

    # =========================================================================
    # Cell 20: Step 10 Markdown
    # =========================================================================
    add_md("""---
## 📑 Step 10: Display a Final Interaction Report

We present the finalized clinical interaction report featuring:
- **Executive Summary Metrics** (Total medicines, high risk, moderate risk, and low risk alerts).
- **Interactive Formatted Table** with color-coded risk levels.
- **Detailed Pharmacological Mechanism Breakdown** for clinicians.
- **Visual Risk & Confidence Distribution Plots**.""")

    # =========================================================================
    # Cell 21: Step 10 Code
    # =========================================================================
    add_code("""def display_final_report(sorted_pairs: List[Dict[str, Any]]):
    # 1. Executive Summary
    high_count = sum(1 for r in sorted_pairs if r["Risk_Level"] == "HIGH RISK")
    mod_count = sum(1 for r in sorted_pairs if r["Risk_Level"] == "MODERATE RISK")
    low_count = sum(1 for r in sorted_pairs if r["Risk_Level"] == "LOW RISK")
    
    print("=" * 85)
    print("                DDI-HIERPRED: CLINICAL DRUG INTERACTION REPORT                ")
    print("=" * 85)
    print(f"Total Drug Pairs Evaluated: {len(sorted_pairs)}")
    print(f"  [!] High Risk (Critical):     {high_count}")
    print(f"  [*] Moderate Risk (Monitor): {mod_count}")
    print(f"  [-] Low Risk (Caution):      {low_count}")
    print("-" * 85)
    
    # 2. DataFrame Table
    table_data = []
    for rank, r in enumerate(sorted_pairs, 1):
        table_data.append({
            "Rank": rank,
            "Drug Pair": f"{r['Pair']['Drug_A_Name']} + {r['Pair']['Drug_B_Name']}",
            "Risk Level": r["Risk_Level"],
            "Interaction Type": r["Final_Type_String"],
            "Ensemble Conf (%)": r["Ensemble_Confidence"],
            "SVM Conf (%)": r["SVM_Confidence"],
            "RF Conf (%)": r["RF_Confidence"],
            "Tanimoto Sim": r["Pair"]["Tanimoto_Similarity"],
            "Known DDI?": "Yes" if r["Is_Known_DDI"] else "Predicted",
            "Mechanism Description": r["Mechanism_Description"],
            "Clinical Action": r["Clinical_Action"]
        })
    df_report = pd.DataFrame(table_data)
    
    # Styled display
    def style_risk(val):
        if val == "HIGH RISK":
            return "background-color: #ffcccc; color: #990000; font-weight: bold;"
        elif val == "MODERATE RISK":
            return "background-color: #fff0cc; color: #b36b00; font-weight: bold;"
        else:
            return "background-color: #e6ffcc; color: #338000; font-weight: bold;"
            
    summary_cols = ["Rank", "Drug Pair", "Risk Level", "Interaction Type", "Ensemble Conf (%)", "Known DDI?"]
    display(df_report[summary_cols].style.map(style_risk, subset=["Risk Level"]))
    
    # 3. Bar Chart of Risk & Confidence
    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_pairs) * 0.4)))
    colors = [
        "#d9534f" if r["Risk_Level"] == "HIGH RISK" 
        else "#f0ad4e" if r["Risk_Level"] == "MODERATE RISK" 
        else "#5cb85c" 
        for r in sorted_pairs
    ]
    y_labels = [f"{r['Pair']['Drug_A_Name']} + {r['Pair']['Drug_B_Name']}" for r in sorted_pairs]
    confs = [r["Ensemble_Confidence"] for r in sorted_pairs]
    
    y_pos = np.arange(len(sorted_pairs))
    bars = ax.barh(y_pos, confs, color=colors, height=0.6, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Ensemble Confidence (%)", fontsize=11, fontweight="bold")
    ax.set_title("Ranked Drug-Drug Interaction Risk & Confidence", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 105)
    
    # Add percentage labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plt.show()
    
    return df_report

df_final = display_final_report(sorted_results)""")

    # =========================================================================
    # Cell 22: End-to-End Master Function Markdown
    # =========================================================================
    add_md("""---
## 🚀 End-to-End Master Execution Function

You can test **any custom list of medicines** in a single call with the master function below:
```python
report_df = run_ddi_pipeline(["Warfarin", "Fluconazole", "Atorvastatin", "Clopidogrel"])
```""")

    # =========================================================================
    # Cell 23: Master Execution Function Code
    # =========================================================================
    add_code("""def run_ddi_pipeline(medicines_list: List[str]) -> pd.DataFrame:
    \"\"\"
    Executes all 10 steps sequentially for any arbitrary input list of medicines.
    \"\"\"
    print(f"\\n{'='*70}")
    print(f"Executing 10-Step DDI Pipeline for: {medicines_list}")
    print(f"{'='*70}")
    
    # 1. Validate
    valid_m, invalid_m = validate_medicines(medicines_list)
    if len(valid_m) < 2:
        print("At least 2 valid medicines are required to predict interactions.")
        return pd.DataFrame()
        
    # 2. Retrieve molecular info
    mol_info = retrieve_molecular_info(valid_m)
    
    # 3. Represent molecules
    represented, _ = represent_molecules(mol_info)
    
    # 4. Generate pairs
    pairs = generate_drug_pairs(represented)
    
    # 5. Extract numerical features
    X, enriched = extract_numerical_features(pairs)
    
    # 6. Apply models
    outputs = apply_models(X, enriched)
    
    # 7. Predict interaction
    preds = predict_interactions(outputs)
    
    # 8. Generate confidence
    confs = generate_confidence_values(preds)
    
    # 9. Sort results
    sorted_p = sort_results(confs)
    
    # 10. Display report
    df_res = display_final_report(sorted_p)
    return df_res


# Demonstration on a cardiovascular & antifungal polypharmacy regimen:
demo_regimen = ["Warfarin", "Fluconazole", "Atorvastatin", "Clopidogrel"]
demo_report = run_ddi_pipeline(demo_regimen)""")

    # Write notebook to disk
    with open("svm_rf.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Successfully built svm_rf.ipynb with all 10 steps!")

if __name__ == "__main__":
    create_notebook()
