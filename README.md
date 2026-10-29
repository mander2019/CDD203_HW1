# CDD 203 Homework 1: hERG Blockade Classifier

Group member names: Elliott Ou, Jieun Lee, Matthew Anderson, Hyunsoo Ryu

## Web App

Find the online implementation of our model [here](https://cdd203hw1.streamlit.app/)

## Model Settings

The final model uses a scikit-learn implementation of `RandomForestClassifier`.
The submitted ONNX model is trained on all rows in `data/hw1data.tab`; no test set is held out during final model training.

## Feature Representation

TODO

## Command

```bash
conda activate cdd203-hw1
python Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.py
```

## Model Selection

The final model is a `RandomForestClassifier` with `n_estimators=250`, `class_weight="balanced"`, `random_state=42`, and `n_jobs=-1`, fit on the complete assignment dataset. A random forest was selected because it trains quickly on CPU, works well with sparse fingerprint-style features plus molecular descriptors, does not require feature scaling, and exports cleanly to ONNX under the assignment size limit.

## Threshold Justification

The selected decision threshold is `0.35` instead of the default `0.50`. For hERG blockade prediction, false negatives are especially risky because a missed blocker could advance despite potential cardiotoxicity concerns. Lowering the threshold improves sensitivity, which prioritizes catching likely blockers, while accepting more false positives.

During threshold selection on an earlier assignment-provided split, threshold `0.35` increased sensitivity from `0.846` to `0.927`. Specificity decreased from `0.837` to `0.702`, so this threshold is intentionally safety-oriented while still preserving reasonable specificity. After selecting this threshold, the final submitted model was retrained on all available rows with no held-out test set.

## Test Results

```text
threshold 0.50
  accuracy            0.842
  sensitivity/recall  0.846
  specificity         0.837
  precision           0.850
  f1                  0.848
  ROC-AUC             0.918
  PR-AUC              0.925

threshold 0.35
  accuracy            0.820
  sensitivity/recall  0.927
  specificity         0.702
  precision           0.773
  f1                  0.843
  ROC-AUC             0.918
  PR-AUC              0.925
```

## Checker Result

Command:

```bash
python TestModel.py Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.py
```

esult summary:

```text
PASS  NAME is filled in (ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU)
PASS  THRESHOLD is between 0 and 1 and is not 0.5 (0.35)
PASS  featurize returns a 2-D array with one row per molecule (6, 2069)
PASS  features are all finite numbers (no NaN or infinity)
PASS  PyTorch models loaded at the top of your script have at most 150M parameters (0.0M)
PASS  Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.onnx is at most 50 MB (43.8 MB)
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
  0.144      no              CCO
  0.172      no              CC(=O)Oc1ccccc1C(=O)O
  0.900      yes             CN(C)CCOC(c1ccccc1)c1ccccc1.Cl
  0.912      yes             C[NH2+]CC[C@@H](Oc1ccc(cc1)C(F)(F)F)c1ccccc1
  0.928      yes             CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1
  0.352      yes             COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC
```
