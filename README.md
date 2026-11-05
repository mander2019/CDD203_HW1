# CDD 203 Homework 1: hERG Blockade Classifier
*Elliott Ou, Jieun Lee, Matthew Anderson, Hyunsoo Ryu*

## Web App
Find the online implementation of our model [here](https://cdd203hw1.streamlit.app/)

## Feature Representation
Each molecule is represented using a combination of RDKit functions / features such as Morgan fingerprint. Morgan fingerprint converts the molecule into 2048 numbers that represent small structural patterns in the molecule, and RDKit implements features that describe properties of the molecule such as molecular weight, polarity, charge, number of rings, and hydrogen-bonding ability.

## Model and Hyperparameters
The final model is a `RandomForestClassifier` with `n_estimators=250`, `class_weight="balanced"`, `random_state=42`, and `n_jobs=-1`, fit on the given data with an 80/20 train/test split. A random forest was selected because it trains quickly on CPU, works well with sparse fingerprint-style features plus molecular descriptors, does not require feature scaling, and exports cleanly to ONNX under the assignment size limit.

## Threshold Justification
The selected decision threshold is `0.35` instead of the default `0.50`. For hERG blockade prediction, false negatives are especially risky because a missed blocker could advance despite potential cardiotoxicity concerns. Lowering the threshold improves sensitivity, which prioritizes catching likely blockers, while accepting more false positives. During threshold selection on an earlier assignment-provided split, threshold `0.35` increased sensitivity from `0.848` to `0.919`. Specificity decreased from `0.831` to `0.690`, so this threshold is intentionally safety-oriented while still preserving reasonable specificity.

## LLM/AI Usage
For feature engineering AI was used to identify data which could be extracted via RDKit and used to represent the molecules. Setting up the web app and adjusting it for formatting and appearance involved heavy LLM usage. It handled unfamiliar logic and syntax and expedited the work. 

## Test Results
| Threshold | Accuracy | Sensitivity / Recall | Specificity | Precision | F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.50** | 0.840 | 0.848 | 0.831 | 0.845 | 0.846 | 0.918 | 0.925 |
| **0.35** | 0.809 | 0.919 | 0.690 | 0.764 | 0.834 | 0.918 | 0.925 |

## Checker Result
Command:
```bash
python TestModel.py Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.py
```

Result summary:
```text
PASS  NAME is filled in (ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU)
PASS  THRESHOLD is between 0 and 1 and is not 0.5 (0.35)
PASS  featurize returns a 2-D array with one row per molecule (6, 2069)
PASS  features are all finite numbers (no NaN or infinity)
PASS  PyTorch models loaded script have at most 150M parameters 
PASS  Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.onnx is at most 50 MB (34.9 MB)
PASS  the ONNX model has exactly one input
PASS  the model returns one prediction per molecule
PASS  predictions are probabilities between 0 and 1
PASS  scoring molecules one at a time gives the same predictions as scoring them together
PASS  requirements.txt lists only allowed libraries at allowed versions
PASS  Part 2.zip contains an identical copy of Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.onnx
```

The model itself passes the assignment checks, including the ONNX size limit. 

## Example Predictions From Checker
```text
P(blocker)  flagged at 0.35  SMILES
  0.156      no              CCO
  0.120      no              CC(=O)Oc1ccccc1C(=O)O
  0.928      yes             CN(C)CCOC(c1ccccc1)c1ccccc1.Cl
  0.860      yes             C[NH2+]CC[C@@H](Oc1ccc(cc1)C(F)(F)F)c1ccccc1
  0.836      yes             CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1
  0.392      yes             COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC