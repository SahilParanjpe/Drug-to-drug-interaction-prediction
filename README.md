# 🧬 Multi-Drug Interaction (DDI) Prediction Pipeline

An end-to-end machine learning system and web application for predicting **Drug-Drug Interactions (DDIs)**. Given an arbitrary list of $n$ drug names, the pipeline resolves their identities via DrugBank vocabularies, generates unique pairwise combinations ($\binom{n}{2}$), extracts chemical/graph feature representations, runs inference using pre-trained Random Forest or XGBoost classifiers, and categorizes results into interactive **Interacting** and **Non-Interacting** lists ranked by confidence scores.

---

## 🏗️ System Architecture

The application is architected as a monolith containerized web service combining a high-performance **FastAPI** backend with a responsive **Tailwind CSS** frontend dashboard.

```
┌────────────────────────────────────────────────────────┐
│                      Client (Browser)                  │
│   - Enters n drug names                                │
│   - Selects Classifier (RF / XGB) & Feature Pipeline   │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP POST /predict (JSON)
                           ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│                                                        │
│  1. Two-Step Drug Resolution:                          │
│     - Checks canonical CSV (Direct Name/ID match)      │
│     - Fallback: DrugBank Vocabulary (Synonyms/Aliases) │
│                                                        │
│  2. Pairwise Combinations:                             │
│     - Computes C(n, 2) unique drug pairs               │
│                                                        │
│  3. Feature Extraction & Inference:                    │
│     - Loads selected feature script & .joblib model    │
│     - Predicts interaction status & confidence score   │
└──────────────────────────┬─────────────────────────────┘
                           │ Returns JSON Results & Missing Drugs
                           ▼
┌────────────────────────────────────────────────────────┐
│                 Dashboard Interface                    │
│   - Amber Warning Banner for Unrecognized Drugs        │
│   - Split-Pane Cards for Interacting & Non-Interacting │
└────────────────────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
my_ddi_app/
├── main.py                                  # FastAPI application & resolution logic
├── requirements.txt                         # Python dependencies
├── Dockerfile                               # Container configuration
├── Final_Canonical_Full_DrugBank_SMILES.csv # Canonical structures & SMILES
├── drugbank vocabulary.csv                  # Vocabulary mapping for synonyms & common names
├── morgan_pipeline.py                       # Morgan fingerprint feature extractor
├── custom32_pipeline.py                     # 32-feature graph & descriptor extractor
├── custom32_wl_pipeline.py                  # 32-feature + Weisfeiler-Lehman graph kernel
├── rf_morgan.joblib                         # Trained Random Forest (Morgan)
├── xgb_morgan.joblib                        # Trained XGBoost (Morgan)
├── rf_custom32.joblib                       # Trained Random Forest (Custom32)
├── xgb_custom32.joblib                      # Trained XGBoost (Custom32)
├── rf_custom32_wl.joblib                    # Trained Random Forest (Custom32 + WL)
├── xgb_custom32_wl.joblib                   # Trained XGBoost (Custom32 + WL)
└── static/
    └── index.html                           # Tailwind CSS frontend dashboard
```

---

## 🔍 Two-Step Case-Insensitive Drug Resolution

To ensure robust user input handling, the backend executes a strict lowercase normalized (`.lower()`) two-step lookup pipeline:

1. **Direct Canonical Check:** Searches `Final_Canonical_Full_DrugBank_SMILES.csv` using the lowercase drug name or DrugBank ID.
2. **Vocabulary / Synonym Fallback:** If not found directly, it queries `drugbank vocabulary.csv` across both `Common name` and pipe-separated `Synonyms` (`|`) to resolve the official `DrugBank ID`, then fetches the canonical SMILES string.
3. **Missing Drug Handling:** Unresolved drugs are filtered out and returned in a `missing_drugs` array, triggering an amber warning banner on the frontend while allowing valid drug pairs to be evaluated successfully.

---

## 🧪 Feature Vector Pipelines Explained

The system supports three distinct feature extraction methodologies, allowing models to evaluate molecular structure from topological fingerprints, handcrafted cheminformatic descriptors, or graph neural kernel comparisons.

### 1. Morgan Fingerprint Pipeline (`morgan`)
* **Description:** Circular fingerprints (similar to ECFP4) capturing local molecular neighborhoods.
* **Extraction Process:** Uses RDKit's Morgan generator with a radius of $2$ and $2048$ bits (`n_bits`).
* **Feature Vector Composition:**
  * Drug A: $2048$ binary bits.
  * Drug B: $2048$ binary bits.
  * **Pair Vector Shape:** Concatenation of $[A, B]$ resulting in **$(1, 4096)$**.

### 2. Custom 32-Feature Pipeline (`custom32`)
* **Description:** Handcrafted molecular topology, elemental composition, and physicochemical properties.
* **Feature Vector Breakdown (32 values per drug):**
  * **Graph Metrics (10 features):** 4 BFS metrics (component count, graph diameter, mean finite shortest-path distance, max BFS eccentricity) + 6 DFS/degree metrics (max DFS tree depth, mean DFS tree depth, leaf count, branching count for degree $\ge 3$, max atom degree, mean atom degree).
  * **Elemental Counts (10 features):** Atom counts for Carbon, Nitrogen, Oxygen, Sulfur, Phosphorus, Fluorine, Chlorine, Bromine, Iodine, plus total halogens.
  * **RDKit Cheminformatic Descriptors (12 features):** Molecular Weight ($\text{MolWt}$), $\text{MolLogP}$, Topological Polar Surface Area ($\text{TPSA}$), Hydrogen Bond Donors ($\text{HBD}$), Hydrogen Bond Acceptors ($\text{HBA}$), Rotatable Bonds, Formal Charge, Fraction $\text{CSP}_3$, Aromatic Rings, Heteroatoms, Molar Refractivity ($\text{MolMR}$), and Atom Stereo Centers.
* **Pair Vector Shape:** Concatenation of $[A_{32}, B_{32}]$ resulting in **$(1, 64)$**.

### 3. Custom 32 + Weisfeiler-Lehman (WL) Kernel Pipeline (`custom32_wl`)
* **Description:** Combines the 32 custom features with iterative Weisfeiler-Lehman graph isomorphism kernel comparisons.
* **Extraction Process:** Refines atom labels iteratively (incorporating atomic number, formal charge, degree, aromaticity, and hydrogen count) across 4 rounds (iterations 0, 1, 2, 3).
* **WL Pair Comparison Features (16 values):** For each of the 4 rounds, computes 4 histogram similarity metrics between Drug A and Drug B:
  1. Cosine similarity of label-count histograms.
  2. Weighted Jaccard similarity.
  3. Normalized L1 similarity.
  4. Fraction of unique labels shared.
* **Pair Vector Shape:** 32 (Drug A) + 32 (Drug B) + 16 (WL Kernel comparisons) = **80 total features**, shaped as **$(1, 80)$**.

---

## 🚀 Installation & Local Execution

### Prerequisites
* Python 3.10+
* Pip / Conda

### 1. Install Dependencies
```bash
pip install fastapi uvicorn numpy pandas joblib scikit-learn rdkit xgboost
```

### 2. Run the FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the Dashboard
Navigate to `http://localhost:8000` in your web browser.

---

## 🐳 Docker Deployment

1. **Build the Docker Image:**
   ```bash
   docker build -t ddi-prediction-app .
   ```

2. **Run the Container Locally:**
   ```bash
   docker run -p 8000:8000 ddi-prediction-app
   ```

3. **Cloud Hosting (e.g., Render, Hugging Face Spaces):**
   * Push your repository to GitHub.
   * Create a new **Web Service** on Render, connect your repository, and select **Docker** as the environment. Render will automatically build and deploy your container with a live public URL.
