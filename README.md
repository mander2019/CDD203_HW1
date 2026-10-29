# CDD 203 Homework 1: hERG Blockade Classifier

Group member names: Elliott Ou, Jieun Lee, Matthew Anderson, Hyunsoo Ryu

## Web App

Find the online implementation of our model [here](https://cdd203hw1.streamlit.app/)

## Model Settings

The final model uses a scikit-learn implementation of `RandomForestClassifier`.

## Feature Representation

TODO

## Command

```bash
conda activate cdd203-hw1
python Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.py
```
## Threshold Justification

The selected decision threshold is `0.35` instead of the default `0.50`. For hERG blockade prediction, false negatives are especially risky because a missed blocker could advance despite potential cardiotoxicity concerns. Lowering the threshold improves sensitivity, which prioritizes catching likely blockers, while accepting more false positives.

On the assignment-provided test split, threshold `0.35` increased sensitivity from `0.761` to `0.950`. Specificity decreased from `0.677` to `0.291`, so this threshold is intentionally safety-oriented rather than specificity-oriented.

## Test Results

```text
threshold 0.50
  accuracy            0.721
  sensitivity/recall  0.761
  specificity         0.677
  precision           0.720
  f1                  0.740
  ROC-AUC             0.792
  PR-AUC              0.803

threshold 0.35
  accuracy            0.635
  sensitivity/recall  0.950
  specificity         0.291
  precision           0.594
  f1                  0.731
  ROC-AUC             0.792
  PR-AUC              0.803
```

## Checker Result

Command:

```bash
python TestModel.py Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.py
```

Result summary:

```text
PASS  NAME is filled in (ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU)
PASS  THRESHOLD is between 0 and 1 and is not 0.5 (0.35)
PASS  featurize returns a 2-D array with one row per molecule (6, 200)
PASS  features are all finite numbers (no NaN or infinity)
PASS  PyTorch models loaded at the top of your script have at most 150M parameters (0.0M)
PASS  Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.onnx is at most 50 MB (26.9 MB)
PASS  the ONNX model has exactly one input
PASS  the model returns one prediction per molecule
PASS  predictions are probabilities between 0 and 1
PASS  scoring molecules one at a time gives the same predictions as scoring them together
PASS  README.md exists and links to your web app
PASS  requirements.txt lists only allowed libraries at allowed versions
FAIL  Part 2.zip contains an identical copy of Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.onnx
```

The model itself passes the assignment checks, including the ONNX size limit. The remaining failure means `Part 2.zip` currently contains an older or different ONNX file and should be rebuilt with the final group model before submission.

## Example Predictions From Checker

```text
P(blocker)  flagged at 0.35  SMILES
  0.341      no              CCO
  0.274      no              CC(=O)Oc1ccccc1C(=O)O
  0.586      yes             CN(C)CCOC(c1ccccc1)c1ccccc1.Cl
  0.770      yes             C[NH2+]CC[C@@H](Oc1ccc(cc1)C(F)(F)F)c1ccccc1
  0.738      yes             CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1
  0.257      no              COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC
```
