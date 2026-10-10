DDI-HierPred Reproducibility Repository

Overview

This repository contains author-generated and processed resources
supporting the DDI-HierPred study, including the final Level 1 and Level
2 data partitions, preprocessing and training notebooks, and additional
validation analyses performed during peer review.

DDI-HierPred is a hierarchical drug-drug interaction (DDI) prediction
framework in which Level 1 assigns a pharmacokinetic (PK) or
pharmacodynamic (PD) branch and Level 2 predicts the detailed
interaction subtype.

Repository Structure

data/

-   DDI_known_lookup.csv — curated documented DDI lookup table used by
    the web platform.
-   Final_Canonical_Full_DrugBank_SMILES.csv — harmonized canonical
    drug/SMILES resource used for molecular representation.
-   Level1_Final_Train_Pairs.csv — Level 1 training partition.
-   Level1_Final_Test_Pairs.csv — Level 1 final test partition.
-   Level2_Train_PairDisjoint.csv — final Level 2 pair-disjoint training
    partition.
-   Level2_Test_PairDisjoint.csv — final Level 2 pair-disjoint test
    partition.

scripts/

-   Data_Preprocessing_Level1.ipynb — Level 1 dataset construction and
    PK/PD preprocessing.
-   Level2_Subtype_Preprocessing.ipynb — Level 2 subtype preprocessing
    and class filtering.
-   Level1_Final_Training.ipynb — Level 1 model development, training,
    and evaluation.
-   Level2_Final_PairDisjoint_Training.ipynb — Level 2 pair-disjoint
    model development, training, and evaluation.
-   Reviewer1_Additional_Validation.ipynb — additional
    leakage-resistant, drug-identity-disjoint, hierarchical, and
    end-to-end validation analyses performed during revision.

See FILE_MANIFEST.xlsx for the purpose of each deposited file.

Data Provenance

The study integrated several DrugBank-derived resources rather than
relying on a single downloadable dataset.

The large-scale DDI interaction table used during dataset construction
was obtained from the Kaggle dataset “drugbank-ddi”. The file used
locally was all_id_interaction.csv, containing Drug1 ID, interaction
description, and Drug2 ID. The currently displayed Kaggle data card
identifies the dataset as Version 15 and reports its licence as
“Unknown”. Because the historical download metadata were not retained,
the repository does not claim that the currently displayed version
number is necessarily identical to the historical version used during
initial dataset assembly.

Additional DrugBank ID–SMILES information was obtained from the Kaggle
dataset “Drug Bank Dataset excel file”. The file used was
drugbank_smiles.csv. The currently displayed Kaggle data card identifies
this dataset as Version 1 and reports its licence as “Unknown”.

DeepDDI-associated resources used during dataset construction included
files such as DrugBank_known_ddi.txt, Interaction_information.txt,
Approved_drug_enzyme_Information.txt, and Drug_Side_Effect.txt. These
resources contain DrugBank-derived information. The original historical
download date/version of these local source files was not preserved, and
no unsupported version or access date is asserted here.

The local project also included author-generated intermediate files
produced by cleaning, harmonization, PK/PD mapping, identifier matching,
subtype filtering, and molecular feature preparation.

Dataset Construction

For Level 1, interaction descriptions were reorganized into PK and PD
categories and processed to construct the final PK/PD classification
dataset.

For Level 2, the original interaction subtype space contained 113
classes. Subtypes represented by fewer than 100 samples were excluded,
leaving 61 retained subtype classes. Drug pairs were canonicalized so
that (Drug A, Drug B) and (Drug B, Drug A) were treated as the same
pair. Pair-disjoint splitting was then performed so that no canonical
drug pair occurred in both training and test partitions.

Molecular representations were generated from Morgan fingerprints
(radius = 2; 2048 bits per drug), yielding a 4096-dimensional pair
representation.

Reproducibility

The notebooks deposited in scripts/ document the principal
preprocessing, model-development, and evaluation procedures used in the
study.

The repository preserves the train/test framework used for the reported
analyses. No independent external validation partition was used for the
original algorithm-selection step. Additional leakage-resistant,
drug-identity-disjoint, hierarchical branch-consistency, and end-to-end
evaluations performed during revision are documented separately in
Reviewer1_Additional_Validation.ipynb and associated outputs.

Because software environments and upstream third-party resources can
change over time, exact reproduction may require compatible versions of
the Python scientific-computing and cheminformatics packages used in the
notebooks.

Licensing and Redistribution

Code written specifically by the authors for DDI-HierPred may be used
under the licence specified in LICENSE.txt and in the Zenodo record.

Third-party source data are not relicensed by the authors.
DrugBank-derived, DeepDDI-associated, Kaggle-hosted, or other externally
sourced materials remain subject to the terms, licences, and
redistribution conditions of their original providers.

Where an upstream data card reports the licence as “Unknown”, this
repository does not infer or assign a licence to that source material.
Inclusion of processed or derived research files is not intended to
transfer ownership of, or override restrictions applying to, the
underlying third-party data.

Users wishing to reuse original third-party content should consult and
comply with the corresponding source provider’s current terms.

Citation

If these resources are used, please cite the associated DDI-HierPred
article and the corresponding Zenodo record.
