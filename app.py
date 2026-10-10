"""
DDI-HierPred Interactive Dashboard
A 10-Step Hierarchical Machine Learning Application for Predicting Drug-Drug Interactions
Powered by Calibrated Support Vector Machines (SVM), Random Forests (RF), and RDKit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from ddi_pipeline import DDIPipeline

# Page Configuration
st.set_page_config(
    page_title="DDI-HierPred | Drug Interaction Engine",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-mod {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    return DDIPipeline()

pipeline = load_pipeline()

# Sidebar: Controls & Drug Selector
with st.sidebar:
    st.image("https://img.icons8.com/color/96/pill.png", width=64)
    st.title("💊 DDI Control Hub")
    st.markdown("**10-Step ML Pipeline Navigator**")
    
    popular_drugs = [
        "Warfarin", "Aspirin", "Ibuprofen", "Metformin", "Omeprazole",
        "Ciprofloxacin", "Atorvastatin", "Clopidogrel", "Fluconazole",
        "Digoxin", "Lisinopril", "Simvastatin", "Levothyroxine", "Amlodipine"
    ]
    
    st.subheader("Select Patient Regimen")
    selected_drugs = st.multiselect(
        "Choose or search medicines:",
        options=sorted(list(set(popular_drugs + list(pipeline.name_to_id.keys())[:2000]))),
        default=["Warfarin", "Aspirin", "Ibuprofen", "Metformin", "Omeprazole", "Ciprofloxacin"],
        help="Select 2 or more medicines to evaluate potential interactions."
    )
    
    custom_entry = st.text_input(
        "Or add custom drug (Name or DrugBank ID):",
        placeholder="e.g. Paracetamol, DB00316"
    )
    if custom_entry:
        if custom_entry.strip() not in selected_drugs:
            selected_drugs.append(custom_entry.strip())
            
    run_btn = st.button("🚀 Analyze Drug Regimen", type="primary", use_container_width=True)
    
    st.divider()
    st.markdown("### 🧬 ML Architecture")
    st.markdown("""
    - **Features:** 2048-bit Morgan Circular FP
    - **SVM Model:** LinearSVC (Platt-Calibrated)
      - *ROC-AUC:* **0.9243** | *Acc:* **84.96%**
    - **RF Model:** 100 Trees (Depth 20)
      - *ROC-AUC:* **0.8747** | *Acc:* **78.72%**
    - **Knowledge Base:** 1.35M Pairs
    """)

# Main Content Header
st.markdown('<div class="main-header">🧬 DDI-HierPred: Hierarchical DDI Prediction Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evaluating Polypharmacy Safety with Machine Learning & Cheminformatics</div>', unsafe_allow_html=True)

# 10 Steps Visual Stepper
steps_cols = st.columns(10)
steps_names = [
    "1. Validate", "2. Molecular", "3. Structures", "4. Pairs",
    "5. Features", "6. Models", "7. Predict", "8. Confidence",
    "9. Sort", "10. Report"
]
for col, name in zip(steps_cols, steps_names):
    col.caption(f"**{name}**")

st.divider()

if not selected_drugs or len(selected_drugs) < 2:
    st.info("👋 Please select at least two medicines in the sidebar and click **Analyze Drug Regimen** to start.")
else:
    # Run Pipeline
    with st.spinner("Processing 10-step DDI prediction workflow..."):
        # Step 1: Validate
        valid_meds, invalid_meds = pipeline.validate_medicines(selected_drugs)
        
        # Step 2: Retrieve molecular info
        mol_records = pipeline.retrieve_molecular_info(valid_meds)
        
        # Step 3: Represent molecules
        represented = pipeline.represent_molecules(mol_records)
        
        # Step 4: Pairs
        pairs = pipeline.generate_drug_pairs(represented)
        
        # Step 5: Features
        X, enriched_pairs = pipeline.extract_numerical_features(pairs)
        
        # Step 6: Apply models
        outputs = pipeline.apply_models(X, enriched_pairs)
        
        # Step 7: Predict
        preds = pipeline.predict_interactions(outputs)
        
        # Step 8: Confidence
        confs = pipeline.generate_confidence_values(preds)
        
        # Step 9: Sort
        sorted_pairs = pipeline.sort_results(confs)

    # Tabs for Step-by-Step Transparency
    tab_report, tab_molecules, tab_pairs, tab_ml_metrics = st.tabs([
        "📑 Final Clinical Report (Steps 9 & 10)",
        "🧪 Molecular Information (Steps 1 - 3)",
        "👥 Drug Pairs & Features (Steps 4 & 5)",
        "🤖 ML Models & Metrics (Steps 6 - 8)"
    ])

    # =========================================================================
    # TAB 1: FINAL REPORT
    # =========================================================================
    with tab_report:
        # Executive Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        high_cnt = sum(1 for r in sorted_pairs if r["Risk_Level"] == "HIGH RISK")
        mod_cnt = sum(1 for r in sorted_pairs if r["Risk_Level"] == "MODERATE RISK")
        low_cnt = sum(1 for r in sorted_pairs if r["Risk_Level"] == "LOW RISK")
        
        m1.metric("Analyzed Medicines", f"{len(valid_meds)}")
        m2.metric("Critical Alerts", f"{high_cnt}", delta="High Risk" if high_cnt > 0 else "None", delta_color="inverse")
        m3.metric("Monitor Alerts", f"{mod_cnt}", delta="Moderate Risk" if mod_cnt > 0 else "None", delta_color="off")
        m4.metric("Total Pairs", f"{len(sorted_pairs)}")
        
        if invalid_meds:
            for inv in invalid_meds:
                st.warning(f"⚠️ Unresolved Input: **'{inv['Query']}'** — {inv['Reason']}")

        st.subheader("Interactive Interaction Summary")
        
        # Render Table
        table_rows = []
        for rank, r in enumerate(sorted_pairs, 1):
            badge = "🔴 Critical" if r["Risk_Level"] == "HIGH RISK" else "🟠 Monitor" if r["Risk_Level"] == "MODERATE RISK" else "🟢 Low"
            table_rows.append({
                "Rank": rank,
                "Drug Pair": f"{r['Pair']['Drug_A_Name']} + {r['Pair']['Drug_B_Name']}",
                "Risk Tier": badge,
                "Type": r["Final_Type_String"],
                "Ensemble Conf": f"{r['Ensemble_Confidence']:.1f}%",
                "SVM Conf": f"{r['SVM_Confidence']:.1f}%",
                "RF Conf": f"{r['RF_Confidence']:.1f}%",
                "Source": "Empirical DDI" if r["Is_Known_DDI"] else "ML Predicted",
                "Mechanism": r["Mechanism_Description"],
                "Action": r["Clinical_Action"]
            })
            
        df_display = pd.DataFrame(table_rows)
        st.dataframe(
            df_display[["Rank", "Drug Pair", "Risk Tier", "Type", "Ensemble Conf", "Source", "Mechanism", "Action"]],
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("Detailed Pharmacological Evaluations")
        
        for r in sorted_pairs:
            pair_title = f"{r['Pair']['Drug_A_Name']} + {r['Pair']['Drug_B_Name']}"
            with st.expander(f"{r['Risk_Badge']} {pair_title} — Ensemble Confidence: {r['Ensemble_Confidence']:.1f}%"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f"**Classification:** `{r['Final_Type_String']}`")
                    st.markdown(f"**Pharmacological Mechanism:** {r['Mechanism_Description']}")
                    st.markdown(f"**Clinical Recommendation:** {r['Clinical_Action']}")
                with c2:
                    st.progress(r["Ensemble_Confidence"] / 100.0, text=f"Ensemble Conf: {r['Ensemble_Confidence']:.1f}%")
                    st.caption(f"Linear SVM: {r['SVM_Confidence']:.1f}% | Random Forest: {r['RF_Confidence']:.1f}%")
                    st.caption(f"Tanimoto Structural Similarity: `{r['Pair']['Tanimoto_Similarity']}`")

    # =========================================================================
    # TAB 2: MOLECULAR INFORMATION
    # =========================================================================
    with tab_molecules:
        st.subheader("Step 1 & 2: Validated Molecular Registry")
        mol_summary_rows = []
        for d in represented:
            mol_summary_rows.append({
                "DrugBank ID": d["DrugBank_ID"],
                "Name": d["Name"],
                "Formula": d["Formula"],
                "Molecular Weight": f"{d['MW']} g/mol",
                "LogP": d["LogP"],
                "TPSA": f"{d['TPSA']} Å²",
                "H-Donors": d["HBD"],
                "H-Acceptors": d["HBA"],
                "Rotatable Bonds": d["Rotatable_Bonds"]
            })
        st.dataframe(pd.DataFrame(mol_summary_rows), use_container_width=True, hide_index=True)
        
        st.subheader("Step 3: 2D Chemical Structures")
        mols = [d["Mol_Object"] for d in represented]
        legends = [f"{d['Name']} ({d['DrugBank_ID']})" for d in represented]
        img = Draw.MolsToGridImage(mols, molsPerRow=min(4, len(mols)), subImgSize=(280, 180), legends=legends)
        st.image(img, use_container_width=True)

    # =========================================================================
    # TAB 3: DRUG PAIRS & FEATURES
    # =========================================================================
    with tab_pairs:
        st.subheader("Step 4: Canonical Drug Pairs")
        st.markdown(f"Generated **{len(pairs)}** unique drug combinations with lexicographical ordering.")
        
        st.subheader("Step 5: 4096-bit Feature Representations")
        st.write(f"Pair Feature Matrix Dimensions: `{X.shape[0]} pairs × {X.shape[1]} bits` (2048-bit ECFP4 Drug A + 2048-bit ECFP4 Drug B)")
        
        # Similarity Heatmap
        if len(represented) > 2:
            st.subheader("Inter-Drug Tanimoto Similarity Matrix")
            n_meds = len(represented)
            sim_matrix = np.zeros((n_meds, n_meds))
            fps = [pipeline.drug_fps_cache[d["DrugBank_ID"]] for d in represented]
            for i in range(n_meds):
                for j in range(n_meds):
                    dot = np.dot(fps[i].astype(int), fps[j].astype(int))
                    denom = np.sum(fps[i]) + np.sum(fps[j]) - dot
                    sim_matrix[i, j] = round(dot / denom, 3) if denom > 0 else 0.0
                    
            fig_sim, ax_sim = plt.subplots(figsize=(6, 5))
            med_names = [d["Name"] for d in represented]
            sns.heatmap(sim_matrix, xticklabels=med_names, yticklabels=med_names, annot=True, cmap="Blues", ax=ax_sim, fmt=".2f")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig_sim)

    # =========================================================================
    # TAB 4: ML METRICS
    # =========================================================================
    with tab_ml_metrics:
        st.subheader("Step 6: Dual Machine Learning Architecture")
        c_svm, c_rf = st.columns(2)
        with c_svm:
            st.markdown("### 🎯 Calibrated Linear SVM")
            st.markdown("- **Algorithm:** LinearSVC with Platt Sigmoid Calibration (CalibratedClassifierCV, cv=3)")
            st.markdown("- **Decision Function Transformation:** Posterior probability $P(y = 1 \mid \mathbf{x})$")
            st.markdown("- **Test ROC-AUC:** `0.9243`")
            st.markdown("- **Test Accuracy:** `84.96%`")
        with c_rf:
            st.markdown("### 🌲 Random Forest Classifier")
            st.markdown("- **Algorithm:** RandomForestClassifier (100 Trees, Depth 20)")
            st.markdown("- **Voting Scheme:** Out-of-bag probability averaging")
            st.markdown("- **Test ROC-AUC:** `0.8747`")
            st.markdown("- **Test Accuracy:** `78.72%`")
            
        st.subheader("Step 7 & 8: Consensus & Calibrated Probabilities")
        chart_df = pd.DataFrame([
            {
                "Drug Pair": f"{r['Pair']['Drug_A_Name']} + {r['Pair']['Drug_B_Name']}",
                "SVM Confidence (%)": r["SVM_Confidence"],
                "RF Confidence (%)": r["RF_Confidence"],
                "Ensemble (%)": r["Ensemble_Confidence"]
            }
            for r in sorted_pairs
        ])
        st.bar_chart(chart_df.set_index("Drug Pair")[["SVM Confidence (%)", "RF Confidence (%)", "Ensemble (%)"]])
