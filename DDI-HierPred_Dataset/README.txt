# DDI-HierPred: Reproducibility Resources

This repository contains the datasets, preprocessing notebooks, model-development notebooks, and validation outputs associated with DDI-HierPred, a hierarchical machine-learning framework for drug–drug interaction prediction.

## Framework Overview

DDI-HierPred consists of two hierarchical prediction levels:

- Level 1: classification of drug–drug interactions as pharmacokinetic (PK) or pharmacodynamic (PD) using a Linear SVM classifier.
- Level 2: prediction of detailed interaction subtypes using an XGBoost multiclass classifier, with subtype predictions constrained to the PK/PD branch selected by Level 1.

Drug pairs are represented using Morgan fingerprints derived from molecular SMILES.

## Data Provenance

The interaction data used in this study were derived from publicly available DeepDDI/DrugBank-derived resources.

The principal source files used during dataset construction included:

- DrugBank_known_ddi.txt
- Drug_info_combined.csv
- Interaction_information.txt
- Approved_drug_enzyme_Information.txt

The preprocessing workflow generated task-specific datasets for Level 1 and Level 2 modeling.

The reported dataset counts represent distinct preprocessing or task-specific stages and should not be interpreted as a single sequential filtering process:

| Dataset stage | Records/pairs |
|---|---:|
| Initial DDI collection | 1,420,072 |
| PK/PD-annotated dataset | 1,355,205 |
| Interaction records with valid SMILES for both drugs | 996,112 |
| Final Level 2 canonical dataset | 220,425 |

## Level 1

Level 1 performs binary PK/PD classification.

The final model uses:

- Morgan fingerprint-based drug-pair representation
- Linear SVM classifier
- Separate training and test partitions

No independent validation partition was used in the original modeling workflow.

## Level 2

The original interaction resource contained 113 documented interaction subtypes.

Interaction subtype preprocessing and class-frequency filtering retained 61 classes for Level 2 modeling.

Canonical drug pairs were generated so that A–B and B–A were treated as the same drug pair. Pair-disjoint partitioning was then applied to prevent the same canonical pair from occurring in both training and test partitions.

The final Level 2 classifier is XGBoost.

## Multi-annotated Drug Pairs

The Level 2 dataset contains 423 canonical drug pairs associated with more than one documented interaction subtype.

Each documented pair–subtype association is retained as a separate single-label record. However, all annotations associated with the same canonical drug pair are assigned to the same data partition to prevent train/test leakage.

The present Level 2 framework should therefore be interpreted as a single-label multiclass formulation rather than a formal multilabel classifier.

## Additional Validation

Additional analyses performed during manuscript revision include:

- drug-overlap assessment
- one-unseen-drug evaluation
- fully drug-disjoint evaluation
- repeated-split variability and confidence intervals
- hierarchical branch-consistency assessment
- lookup PK/PD agreement analysis
- end-to-end hierarchical evaluation
- audit of multi-annotated drug pairs

These analyses are provided in the accompanying reviewer-validation notebooks and output files.

## Prediction Scores

For Level 1, the Linear SVM decision-function output is transformed using a sigmoid function to obtain a numerical confidence score.

These values are not calibrated probabilities.

For Level 2, XGBoost class scores are used to rank candidate interaction subtypes. Following hierarchical constraint, only subtypes belonging to the Level 1-selected PK or PD branch are eligible for reporting.

The scores are reported as confidence scores and should not be interpreted as calibrated probabilities.

## Repository Structure

### Data

The deposited data files contain the curated datasets and train/test partitions used in the study.

### Scripts / Notebooks

The notebooks document the main preprocessing, model-development, and validation procedures.

See `FILE_MANIFEST.csv` for the purpose of each deposited file.

## Reproducibility Note

The repository preserves the train/test framework used in the reported study. No independent validation partition was used for algorithm selection.

The additional leakage-resistant and hierarchical validation analyses performed during revision are provided separately and do not alter the original model-development partitions.

## Data and Code Licensing

Code developed specifically for DDI-HierPred is distributed under the licence specified in this Zenodo record.

Third-party source data remain subject to the terms and licensing conditions of their original providers. Deposition of derived or processed files does not transfer ownership or override the licensing conditions of the original data sources.

Users should consult the original DeepDDI and DrugBank resources for their applicable terms of use and licensing conditions.

## Citation

If you use these resources, please cite the associated DDI-HierPred article and the corresponding Zenodo record.