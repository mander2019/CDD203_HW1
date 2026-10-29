# CDD 203 Homework 1: hERG Blockade Classifier

Group member names: Elliott Ou, Jieun Lee, Matthew Anderson, Hyunsoo Ryu

## Web App

Find the online implementation of our model [here](https://cdd203hw1.streamlit.app/)

## Model Settings

The final model uses a scikit-learn implementation of `RandomForestClassifier`.

- n_estimators = 300
- max_depth = 20
- min_samples_leaf = 2
- class_weight = balanced
- random_state = 42
- threshold = 0.35


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
