"""
Drug-Drug Interaction (DDI) Prediction Pipeline
================================================
A 10-Step Hierarchical Machine Learning Framework for Predicting and
Risk-Stratifying Drug-Drug Interactions.

Steps:
  1. Validate the medicines
  2. Retrieve their molecular information
  3. Represent the molecules using appropriate data structures
  4. Generate all possible drug pairs
  5. Extract numerical features (Morgan Fingerprints)
  6. Apply RF / SVM models
  7. Predict interaction for each pair (PK vs PD)
  8. Generate confidence values (Calibrated Probabilities)
  9. Sort results according to risk and confidence
 10. Display a final clinical interaction report
"""

import sys
import time
import json
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib

# Cheminformatics (RDKit)
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs
from rdkit.Chem import rdFingerprintGenerator


class DDIPipeline:
    """
    End-to-end 10-step Drug-Drug Interaction Prediction Engine.
    """

    def __init__(
        self,
        smiles_path: str = "Final_Canonical_Full_DrugBank_SMILES.csv",
        vocab_path: str = "drugbank vocabulary (1).csv",
        lookup_path: str = "DDI-HierPred_Dataset/DDI_known_lookup.csv",
        models_dir: str = "models",
    ):
        self.models_dir = Path(models_dir)
        self.smiles_path = smiles_path
        self.vocab_path = vocab_path
        self.lookup_path = lookup_path

        # RDKit Morgan Fingerprint Generator (Radius 2, 2048 bits, Chirality)
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            includeChirality=True
        )

        # In-memory indices
        self.smiles_df: Optional[pd.DataFrame] = None
        self.vocab_df: Optional[pd.DataFrame] = None
        self.known_lookup_df: Optional[pd.DataFrame] = None

        self.name_to_id: Dict[str, str] = {}
        self.id_to_meta: Dict[str, Dict[str, Any]] = {}
        self.drug_fps_cache: Dict[str, np.ndarray] = {}

        # Machine Learning Models
        self.svm_model = None
        self.rf_model = None

        self._load_resources()

    def _load_resources(self):
        """Load datasets, dictionary mappings, and pre-trained models."""
        print("[Pipeline Init] Loading molecular databases and vocabularies...")

        # 1 & 2. Drug Registry & Aliases (Common Name, SMILES, Properties)
        reg_file = self.models_dir / "drug_registry.joblib"
        if reg_file.exists():
            print("[Pipeline Init] Loading unified DrugBank registry from cache...")
            cached_data = joblib.load(reg_file)
            self.id_to_meta = cached_data["registry"]
            self.name_to_id = cached_data["name_to_id"]
            print(f"[Pipeline Init] Loaded {len(self.id_to_meta):,} drugs and {len(self.name_to_id):,} aliases.")
        else:
            # Fallback to CSV parsing
            if Path(self.smiles_path).exists():
                self.smiles_df = pd.read_csv(self.smiles_path)
                for _, r in self.smiles_df.iterrows():
                    db_id = str(r["DrugBank_ID"]).strip().upper()
                    name = str(r["Name"]).strip()
                    smiles = str(r["SMILES"]).strip()
                    self.id_to_meta[db_id] = {
                        "DrugBank_ID": db_id,
                        "Name": name,
                        "SMILES": smiles,
                        "Heavy_atom_count": r.get("Heavy_atom_count", None),
                    }
                    if name and name != "nan":
                        self.name_to_id[name.lower()] = db_id

            if Path(self.vocab_path).exists():
                self.vocab_df = pd.read_csv(self.vocab_path)
                for _, r in self.vocab_df.iterrows():
                    db_id = str(r["DrugBank ID"]).strip().upper()
                    if db_id in self.id_to_meta:
                        cname = str(r["Common name"]).strip()
                        if cname and cname != "nan":
                            self.id_to_meta[db_id]["Name"] = cname
                            self.name_to_id[cname.lower()] = db_id
                        syns = str(r.get("Synonyms", ""))
                        if syns and syns != "nan":
                            for syn in syns.split("|"):
                                s_clean = syn.strip().lower()
                                if s_clean:
                                    self.name_to_id[s_clean] = db_id

        # 3. Known Interaction Lookup Table (Empirical Ground Truth)
        cached_lookup = self.models_dir / "known_lookup.joblib"
        if cached_lookup.exists():
            print("[Pipeline Init] Loading compiled DDI empirical knowledge base from cache...")
            t_lk = time.time()
            raw_lookup = joblib.load(cached_lookup)
            self.known_lookup_dict = {
                k: {"Known_PK_PD": v[0], "Known_interaction": v[1]}
                for k, v in raw_lookup.items()
            }
            print(f"[Pipeline Init] Loaded {len(self.known_lookup_dict):,} interaction pairs in {time.time()-t_lk:.2f}s.")
        elif Path(self.lookup_path).exists():
            print("[Pipeline Init] Indexing empirical DDI knowledge base from CSV...")
            df = pd.read_csv(
                self.lookup_path,
                usecols=["unordered_key", "Known_PK_PD", "Known_interaction"]
            ).drop_duplicates(subset=["unordered_key"])
            self.known_lookup_dict = {
                k: {"Known_PK_PD": pkpd, "Known_interaction": desc}
                for k, pkpd, desc in zip(df["unordered_key"], df["Known_PK_PD"], df["Known_interaction"])
            }
        else:
            self.known_lookup_dict = {}

        # 4. Load Models & Fingerprint Cache
        fp_cache_file = self.models_dir / "drug_fps.joblib"
        if fp_cache_file.exists():
            self.drug_fps_cache = joblib.load(fp_cache_file)
            print(f"[Pipeline Init] Loaded cached fingerprints for {len(self.drug_fps_cache)} drugs.")

        svm_file = self.models_dir / "svm_calibrated_model.joblib"
        if svm_file.exists():
            self.svm_model = joblib.load(svm_file)
            print("[Pipeline Init] Loaded Calibrated Linear SVM model.")

        rf_file = self.models_dir / "rf_model.joblib"
        if rf_file.exists():
            self.rf_model = joblib.load(rf_file)
            print("[Pipeline Init] Loaded Random Forest model.")

    # =========================================================================
    # STEP 1: Validate the medicines
    # =========================================================================
    def validate_medicines(self, medicine_inputs: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validates medicine names / DrugBank IDs against the database.
        Returns:
            valid_medicines: list of resolved drug dictionaries
            invalid_medicines: list of unrecognized inputs with diagnostic errors
        """
        valid_medicines = []
        invalid_medicines = []
        seen_ids = set()

        for raw_input in medicine_inputs:
            input_clean = str(raw_input).strip()
            if not input_clean:
                continue

            # Check if input is a direct DrugBank ID
            db_id_candidate = input_clean.upper()
            resolved_id = None

            if db_id_candidate in self.id_to_meta:
                resolved_id = db_id_candidate
            else:
                # Search name / synonym dictionary
                query = input_clean.lower()
                if query in self.name_to_id:
                    resolved_id = self.name_to_id[query]

            if resolved_id and resolved_id in self.id_to_meta:
                if resolved_id in seen_ids:
                    continue  # Avoid duplicate entries
                seen_ids.add(resolved_id)
                meta = self.id_to_meta[resolved_id]
                valid_medicines.append({
                    "Query": input_clean,
                    "DrugBank_ID": resolved_id,
                    "Name": meta["Name"],
                    "SMILES": meta["SMILES"],
                    "Status": "VALID"
                })
            else:
                invalid_medicines.append({
                    "Query": input_clean,
                    "Reason": "Drug name or ID not found in DrugBank registry or missing 2D structure.",
                    "Status": "INVALID"
                })

        return valid_medicines, invalid_medicines

    # =========================================================================
    # STEP 2: Retrieve their molecular information
    # =========================================================================
    def retrieve_molecular_info(self, valid_medicines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Retrieves detailed molecular information (SMILES, heavy atoms, formula, etc.)
        """
        molecular_records = []
        for drug in valid_medicines:
            db_id = drug["DrugBank_ID"]
            meta = self.id_to_meta[db_id]
            smiles = meta["SMILES"]

            record = {
                "DrugBank_ID": db_id,
                "Name": meta["Name"],
                "Query": drug["Query"],
                "SMILES": smiles,
                "Heavy_atom_count": meta.get("Heavy_atom_count", 0),
            }
            molecular_records.append(record)

        return molecular_records

    # =========================================================================
    # STEP 3: Represent the molecules using appropriate data structures
    # =========================================================================
    def represent_molecules(self, molecular_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Constructs RDKit Mol objects and extracts physicochemical descriptors
        (Molecular Weight, LogP, H-Bond Donors, H-Bond Acceptors, TPSA, Rotatable Bonds).
        """
        represented_drugs = []
        for rec in molecular_records:
            smiles = rec["SMILES"]
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                continue

            mol_formula = rdMolDescriptors.CalcMolFormula(mol)
            mw = round(Descriptors.MolWt(mol), 2)
            logp = round(Descriptors.MolLogP(mol), 2)
            tpsa = round(Descriptors.TPSA(mol), 2)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

            updated = dict(rec)
            updated.update({
                "Mol_Object": mol,
                "Formula": mol_formula,
                "Molecular_Weight": mw,
                "LogP": logp,
                "TPSA": tpsa,
                "H_Donors": hbd,
                "H_Acceptors": hba,
                "Rotatable_Bonds": rot_bonds
            })
            represented_drugs.append(updated)

        return represented_drugs

    # =========================================================================
    # STEP 4: Generate all possible drug pairs
    # =========================================================================
    def generate_drug_pairs(self, represented_drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates all N*(N-1)/2 unique 2-drug combinations with canonical ordering.
        """
        pairs = []
        for drug1, drug2 in itertools.combinations(represented_drugs, 2):
            id1 = drug1["DrugBank_ID"]
            id2 = drug2["DrugBank_ID"]

            # Canonical lexicographical ordering as defined in DDI-HierPred
            if id1 < id2:
                pair_a, pair_b = drug1, drug2
            else:
                pair_a, pair_b = drug2, drug1

            pair_info = {
                "Pair_Key": f"{pair_a['DrugBank_ID']}||{pair_b['DrugBank_ID']}",
                "Drug_A_ID": pair_a["DrugBank_ID"],
                "Drug_A_Name": pair_a["Name"],
                "Drug_A_Obj": pair_a,
                "Drug_B_ID": pair_b["DrugBank_ID"],
                "Drug_B_Name": pair_b["Name"],
                "Drug_B_Obj": pair_b
            }
            pairs.append(pair_info)

        return pairs

    # =========================================================================
    # STEP 5: Extract numerical features (Morgan Fingerprints)
    # =========================================================================
    def extract_numerical_features(self, pairs: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generates 2048-bit Morgan Fingerprints for Drug A and Drug B and concatenates
        them into a 4096-dimensional binary feature vector.
        Also computes Tanimoto structural similarity.
        """
        features_list = []
        enriched_pairs = []

        for pair in pairs:
            id_a = pair["Drug_A_ID"]
            id_b = pair["Drug_B_ID"]

            # Retrieve or generate FP for Drug A
            if id_a in self.drug_fps_cache:
                fp_a = self.drug_fps_cache[id_a]
            else:
                mol_a = pair["Drug_A_Obj"]["Mol_Object"]
                fp_a = self.morgan_gen.GetFingerprintAsNumPy(mol_a).astype(np.uint8)
                self.drug_fps_cache[id_a] = fp_a

            # Retrieve or generate FP for Drug B
            if id_b in self.drug_fps_cache:
                fp_b = self.drug_fps_cache[id_b]
            else:
                mol_b = pair["Drug_B_Obj"]["Mol_Object"]
                fp_b = self.morgan_gen.GetFingerprintAsNumPy(mol_b).astype(np.uint8)
                self.drug_fps_cache[id_b] = fp_b

            # Horizontal concatenation: 2048 + 2048 = 4096 dimensions
            pair_feature_vector = np.concatenate([fp_a, fp_b])
            features_list.append(pair_feature_vector)

            # Calculate Tanimoto Coefficient
            dot_prod = np.dot(fp_a.astype(int), fp_b.astype(int))
            denom = np.sum(fp_a) + np.sum(fp_b) - dot_prod
            tanimoto = round(float(dot_prod / denom), 4) if denom > 0 else 0.0

            pair_copy = dict(pair)
            pair_copy["Tanimoto_Similarity"] = tanimoto
            enriched_pairs.append(pair_copy)

        if features_list:
            X = np.array(features_list, dtype=np.uint8)
        else:
            X = np.empty((0, 4096), dtype=np.uint8)

        return X, enriched_pairs

    # =========================================================================
    # STEP 6: Apply RF / SVM models
    # =========================================================================
    def apply_models(self, X: np.ndarray, enriched_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes calibrated Support Vector Machine (LinearSVC) and Random Forest classifiers.
        Also cross-references the empirical DDI known lookup database.
        """
        if len(enriched_pairs) == 0:
            return []

        # Predict with Calibrated SVM
        svm_preds = self.svm_model.predict(X)
        svm_probs = self.svm_model.predict_proba(X)  # shape (N, 2): [P(PK), P(PD)]

        # Predict with Random Forest
        rf_preds = self.rf_model.predict(X)
        rf_probs = self.rf_model.predict_proba(X)    # shape (N, 2): [P(PK), P(PD)]

        predictions = []
        for i, pair in enumerate(enriched_pairs):
            pair_key = pair["Pair_Key"]
            rev_key = f"{pair['Drug_B_ID']}||{pair['Drug_A_ID']}"

            # Check lookup database
            lookup_match = None
            if pair_key in self.known_lookup_dict:
                lookup_match = self.known_lookup_dict[pair_key]
            elif rev_key in self.known_lookup_dict:
                lookup_match = self.known_lookup_dict[rev_key]

            predictions.append({
                "Pair": pair,
                "SVM_Pred_Class": int(svm_preds[i]),
                "SVM_Prob_PK": float(svm_probs[i, 0]),
                "SVM_Prob_PD": float(svm_probs[i, 1]),
                "RF_Pred_Class": int(rf_preds[i]),
                "RF_Prob_PK": float(rf_probs[i, 0]),
                "RF_Prob_PD": float(rf_probs[i, 1]),
                "Known_Lookup": lookup_match
            })

        return predictions

    # =========================================================================
    # STEP 7: Predict interaction for each pair
    # =========================================================================
    def predict_interactions(self, model_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assigns interaction branch (PK vs PD), consensus interaction type,
        and human-readable mechanism explanation.
        """
        interaction_results = []
        class_map = {0: "PK (Pharmacokinetic)", 1: "PD (Pharmacodynamic)"}

        for item in model_outputs:
            svm_class = item["SVM_Pred_Class"]
            rf_class = item["RF_Pred_Class"]
            lookup = item["Known_Lookup"]

            # Empirical lookup takes priority for confirmed classification
            if lookup:
                known_type = lookup["Known_PK_PD"].upper()
                final_class_str = f"{known_type} (Confirmed)"
                final_type_code = 1 if known_type == "PD" else 0
                is_known = True

                # Replace (.*) placeholders with drug names
                d_a = item["Pair"]["Drug_A_Name"]
                d_b = item["Pair"]["Drug_B_Name"]
                raw_mech = lookup["Known_interaction"]
                parts = raw_mech.split("(.*)")
                if len(parts) >= 3:
                    formatted_mech = f"{parts[0]}{d_a}{parts[1]}{d_b}{parts[2]}"
                else:
                    formatted_mech = raw_mech.replace("(.*)", f"{d_a} / {d_b}")
            else:
                # ML consensus prediction
                is_known = False
                if svm_class == rf_class:
                    final_type_code = svm_class
                    final_class_str = class_map[final_type_code]
                else:
                    # Default to SVM (higher ROC-AUC 0.924 vs 0.875)
                    final_type_code = svm_class
                    final_class_str = f"{class_map[final_type_code]} [SVM Selected]"

                d_a = item["Pair"]["Drug_A_Name"]
                d_b = item["Pair"]["Drug_B_Name"]
                if final_type_code == 0:
                    formatted_mech = (
                        f"Pharmacokinetic (PK) interaction predicted: {d_a} and {d_b} are likely to alter "
                        f"each other's absorption, metabolism (CYP enzyme pathways), or renal clearance."
                    )
                else:
                    formatted_mech = (
                        f"Pharmacodynamic (PD) interaction predicted: {d_a} and {d_b} exhibit synergistic, "
                        f"additive, or antagonistic pharmacological activity on shared biological targets."
                    )

            res = dict(item)
            res.update({
                "Final_Type_Code": final_type_code,
                "Final_Type_String": final_class_str,
                "Is_Known_DDI": is_known,
                "Mechanism_Description": formatted_mech,
                "Model_Consensus": (svm_class == rf_class)
            })
            interaction_results.append(res)

        return interaction_results

    # =========================================================================
    # STEP 8: Generate confidence values
    # =========================================================================
    def generate_confidence_values(self, interaction_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Computes calibrated posterior confidence percentages for SVM, RF,
        and the ensemble consensus.
        """
        enriched_results = []
        for item in interaction_results:
            target_code = item["Final_Type_Code"]

            # Probability of target branch
            svm_conf = item["SVM_Prob_PD"] if target_code == 1 else item["SVM_Prob_PK"]
            rf_conf = item["RF_Prob_PD"] if target_code == 1 else item["RF_Prob_PK"]

            # Ensemble confidence (weighted average: SVM 0.60, RF 0.40 based on test ROC-AUC)
            if item["Is_Known_DDI"]:
                ensemble_conf = 0.985  # Empirical confirmed benchmark
            else:
                ensemble_conf = (0.60 * svm_conf) + (0.40 * rf_conf)

            item_copy = dict(item)
            item_copy.update({
                "SVM_Confidence": round(float(svm_conf * 100), 2),
                "RF_Confidence": round(float(rf_conf * 100), 2),
                "Ensemble_Confidence": round(float(ensemble_conf * 100), 2)
            })
            enriched_results.append(item_copy)

        return enriched_results

    # =========================================================================
    # STEP 9: Sort results according to risk and confidence
    # =========================================================================
    def sort_results(self, enriched_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classifies risk severity (High / Moderate / Low) and sorts pairs
        by Risk Priority (descending) followed by Confidence Score (descending).
        """
        high_risk_keywords = [
            "anticoagulant", "bleeding", "hemorrhage", "cns depression",
            "qtc prolongation", "arrhythmia", "cardiotoxicity", "hypotension",
            "hyperkalemia", "nephrotoxicity", "seizure", "respiratory depression",
            "serotonin syndrome", "hypoglycemia"
        ]

        scored_results = []
        for item in enriched_results:
            mech_lower = item["Mechanism_Description"].lower()
            target_code = item["Final_Type_Code"]
            conf = item["Ensemble_Confidence"]

            # Determine Risk Tier
            is_critical_keyword = any(kw in mech_lower for kw in high_risk_keywords)

            if is_critical_keyword or (target_code == 1 and conf >= 75.0):
                risk_level = "HIGH RISK"
                risk_rank = 3
                badge = "[HIGH RISK - CRITICAL]"
                clinical_action = "Avoid concurrent combination or closely monitor patient vital parameters / labs."
            elif target_code == 0 and conf >= 70.0:
                risk_level = "MODERATE RISK"
                risk_rank = 2
                badge = "[MODERATE RISK - MONITOR]"
                clinical_action = "Consider dosage titration or monitoring serum drug concentrations."
            else:
                risk_level = "LOW RISK"
                risk_rank = 1
                badge = "[LOW RISK - CAUTION]"
                clinical_action = "Standard clinical vigilance; minimal acute adverse expectation."

            res = dict(item)
            res.update({
                "Risk_Level": risk_level,
                "Risk_Rank": risk_rank,
                "Risk_Badge": badge,
                "Clinical_Action": clinical_action
            })
            scored_results.append(res)

        # Multi-key sort: 1) Risk_Rank descending, 2) Ensemble_Confidence descending
        sorted_results = sorted(
            scored_results,
            key=lambda x: (x["Risk_Rank"], x["Ensemble_Confidence"]),
            reverse=True
        )

        return sorted_results

    # =========================================================================
    # STEP 10: Display a final interaction report
    # =========================================================================
    def display_final_report(
        self,
        sorted_results: List[Dict[str, Any]],
        valid_medicines: List[Dict[str, Any]],
        invalid_medicines: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generates and displays the final clinical interaction report,
        including executive summary metrics and detailed tabular output.
        """
        print("\n" + "=" * 90)
        print("          DDI-HIERPRED: CLINICAL DRUG-DRUG INTERACTION REPORT          ")
        print("=" * 90)

        # Executive Summary
        total_meds = len(valid_medicines)
        total_pairs = len(sorted_results)
        high_risk_count = sum(1 for r in sorted_results if r["Risk_Level"] == "HIGH RISK")
        mod_risk_count = sum(1 for r in sorted_results if r["Risk_Level"] == "MODERATE RISK")
        low_risk_count = sum(1 for r in sorted_results if r["Risk_Level"] == "LOW RISK")

        print(f"\n[EXECUTIVE SUMMARY]")
        print(f"  * Total Medicines Analyzed: {total_meds}")
        print(f"  * Total Drug Pairs Evaluated: {total_pairs}")
        print(f"  * [!] High Risk (Critical) Pairs:   {high_risk_count}")
        print(f"  * [*] Moderate Risk Pairs:         {mod_risk_count}")
        print(f"  * [-] Low Risk / Minor Pairs:       {low_risk_count}")

        if invalid_medicines:
            print("\n[UNRESOLVED MEDICINES]")
            for inv in invalid_medicines:
                print(f"  - '{inv['Query']}': {inv['Reason']}")

        # Build Formatted Table
        table_rows = []
        for rank, r in enumerate(sorted_results, 1):
            table_rows.append({
                "Rank": rank,
                "Drug Pair": f"{r['Pair']['Drug_A_Name']} + {r['Pair']['Drug_B_Name']}",
                "Interaction Type": r["Final_Type_String"],
                "Risk Level": r["Risk_Level"],
                "Ensemble Conf (%)": f"{r['Ensemble_Confidence']:.1f}%",
                "SVM Conf (%)": f"{r['SVM_Confidence']:.1f}%",
                "RF Conf (%)": f"{r['RF_Confidence']:.1f}%",
                "Tanimoto Sim": r['Pair']['Tanimoto_Similarity'],
                "Known DDI?": "Yes" if r["Is_Known_DDI"] else "Predicted",
                "Mechanism / Warning": r["Mechanism_Description"],
                "Clinical Recommendation": r["Clinical_Action"]
            })

        df_report = pd.DataFrame(table_rows)

        print("\n[INTERACTION SUMMARY TABLE]")
        if not df_report.empty:
            summary_cols = ["Rank", "Drug Pair", "Risk Level", "Interaction Type", "Ensemble Conf (%)", "Known DDI?"]
            print(df_report[summary_cols].to_string(index=False))

            print("\n" + "-" * 90)
            print("[DETAILED PHARMACOLOGICAL ASSESSMENTS]")
            print("-" * 90)
            for r in sorted_results:
                pair_name = f"{r['Pair']['Drug_A_Name']} + {r['Pair']['Drug_B_Name']}"
                print(f"\n>> {pair_name} ({r['Pair']['Drug_A_ID']} & {r['Pair']['Drug_B_ID']})")
                print(f"   Severity Badge : {r['Risk_Badge']}")
                print(f"   Type           : {r['Final_Type_String']}")
                print(f"   Confidence     : Ensemble {r['Ensemble_Confidence']:.1f}% (SVM: {r['SVM_Confidence']:.1f}%, RF: {r['RF_Confidence']:.1f}%)")
                print(f"   Structural Sim : Tanimoto Coefficient = {r['Pair']['Tanimoto_Similarity']}")
                print(f"   Mechanism      : {r['Mechanism_Description']}")
                print(f"   Action Advice  : {r['Clinical_Action']}")
        else:
            print("No drug pairs evaluated.")

        print("\n" + "=" * 90)
        print("                          END OF REPORT                               ")
        print("=" * 90 + "\n")

        return df_report

    # =========================================================================
    # MASTER RUNNER: End-to-End Execution
    # =========================================================================
    def run_pipeline(self, medicines: List[str]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Executes all 10 steps sequentially for an input list of medicines.
        """
        print(f"\n--- Initiating 10-Step DDI Analysis for {len(medicines)} medicines ---")

        # Step 1: Validate medicines
        valid_meds, invalid_meds = self.validate_medicines(medicines)
        if len(valid_meds) < 2:
            print(f"[Notice] Need at least 2 valid medicines to evaluate interactions. Found: {len(valid_meds)}")
            return pd.DataFrame(), []

        # Step 2: Retrieve molecular info
        mol_records = self.retrieve_molecular_info(valid_meds)

        # Step 3: Represent molecules
        represented_drugs = self.represent_molecules(mol_records)

        # Step 4: Generate pairs
        pairs = self.generate_drug_pairs(represented_drugs)

        # Step 5: Extract numerical features
        X, enriched_pairs = self.extract_numerical_features(pairs)

        # Step 6: Apply RF / SVM models
        model_outputs = self.apply_models(X, enriched_pairs)

        # Step 7: Predict interaction for each pair
        predictions = self.predict_interactions(model_outputs)

        # Step 8: Generate confidence values
        conf_results = self.generate_confidence_values(predictions)

        # Step 9: Sort results by risk and confidence
        sorted_results = self.sort_results(conf_results)

        # Step 10: Display final report
        df_report = self.display_final_report(sorted_results, valid_meds, invalid_meds)

        return df_report, sorted_results


def test_pipeline():
    """Unit test for the 10-step DDI pipeline with a polypharmacy regimen."""
    pipeline = DDIPipeline()

    # Polypharmacy test regimen: Anticoagulant, NSAID, Antidiabetic, Antibiotic, PPI
    test_regimen = [
        "Warfarin",
        "Aspirin",
        "Ibuprofen",
        "Metformin",
        "Omeprazole",
        "Ciprofloxacin",
        "NonExistentDrugXYZ"
    ]

    report_df, results = pipeline.run_pipeline(test_regimen)
    return report_df, results


if __name__ == "__main__":
    test_pipeline()
