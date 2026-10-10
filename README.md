# 💊 Drug-to-Drug Interaction (DDI) Prediction Pipeline

A hierarchical machine learning and cheminformatics framework to predict, classify, and risk-stratify drug–drug interactions using **Support Vector Machines (SVM)**, **Random Forests (RF)**, and **RDKit**.

---

## 📌 10-Step Hierarchical Pipeline Workflow

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

---

## 🚀 Key Features

- **Cheminformatics with RDKit**: Converts chemical SMILES into 2048-bit circular Morgan Fingerprints ($r=2$, chirality preserved) and extracts molecular properties (MW, LogP, TPSA, H-Donors, H-Acceptors, Rotatable Bonds).
- **Dual Machine Learning Classifiers**:
  - **Calibrated Linear SVM**: Uses Platt sigmoid scaling (`CalibratedClassifierCV`) to output posterior probabilities ($P_{\text{target}}$). Held-out test **ROC-AUC: 0.9243**, **Accuracy: 84.96%**.
  - **Random Forest (RF)**: 100 decorrelated decision trees (max depth 20). Held-out test **ROC-AUC: 0.8747**, **Accuracy: 78.72%**.
- **Empirical Ground Truth Verification**: Integrated with a 1.35-million-pair DDI knowledge base for mechanistic confirmation.
- **Clinical Risk Stratification**:
  - 🔴 **HIGH RISK (Critical)**: Severe pharmacodynamic hazards (anticoagulant bleeding, CNS depression, QTc prolongation/arrhythmia, hypoglycemia).
  - 🟠 **MODERATE RISK (Monitor)**: Significant pharmacokinetic shifts (CYP metabolism inhibition/induction, clearance shifts).
  - 🟢 **LOW RISK (Caution)**: Mild excretion shifts or low-confidence interactions.
- **Multiple Interfaces**:
  - Interactive Jupyter Notebook: [`svm_rf.ipynb`](svm_rf.ipynb)
  - Standalone Modular Engine & CLI: [`ddi_pipeline.py`](ddi_pipeline.py)
  - Interactive Web Dashboard: [`app.py`](app.py) (Streamlit)

---

## 📁 Repository Structure

```
├── svm_rf.ipynb               # Complete 10-step interactive Jupyter notebook
├── ddi_pipeline.py            # Standalone 10-step DDI pipeline module & CLI
├── app.py                     # Streamlit web interface with 2D molecular visualizer
├── train_models.py            # Model training & serialization script
├── models/                    # Serialized models and registry caches
│   ├── svm_calibrated_model.joblib
│   ├── rf_model.joblib
│   ├── drug_fps.joblib
│   ├── drug_registry.joblib
│   ├── known_lookup.joblib
│   └── metrics.json
├── Final_Canonical_Full_DrugBank_SMILES.csv
├── drugbank vocabulary (1).csv
├── DDI-HierPred_Dataset/      # Benchmark datasets & partitions
└── README.md
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SahilParanjpe/Drug-to-drug-interaction-prediction.git
   cd Drug-to-drug-interaction-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install rdkit scikit-learn pandas numpy scipy joblib matplotlib seaborn streamlit
   ```

---

## 💻 Usage

### 1. Interactive Notebook
```bash
jupyter notebook svm_rf.ipynb
```

### 2. Python Script / CLI
```bash
python ddi_pipeline.py
```

Or import programmatically:
```python
from ddi_pipeline import DDIPipeline

pipeline = DDIPipeline()
report_df, results = pipeline.run_pipeline(["Warfarin", "Aspirin", "Metformin", "Omeprazole"])
```

### 3. Interactive Web Dashboard
```bash
streamlit run app.py
```

---

## 📊 Evaluation Metrics

| Model | Test Accuracy | Test ROC-AUC | F1-Score | Calibration |
|---|:---:|:---:|:---:|:---:|
| **Calibrated Linear SVM** | **84.96%** | **0.9243** | **0.8631** | Platt Sigmoid Scaling |
| **Random Forest (100 Trees)** | **78.72%** | **0.8747** | **0.8226** | Out-of-Bag Voting |

---

## 📜 License
This project is licensed under the MIT License.
