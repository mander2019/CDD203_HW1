"""
Homework 1, Part 1 - hERG blockade classifier.

Rename to Part1_[GROUP_NAMES].py, fill in the four TODOs, 
then run: python Part1_[GROUP_NAMES].py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Lipinski, rdMolDescriptors


# TODO 1: your group's names -> Part1_FIRST1-LAST1-FIRST2-LAST2.onnx
NAME = "ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU"


# TODO 2: your own threshold; justify it in README.md
THRESHOLD = 0.5


def featurize(smiles):
    """TODO 3: SMILES strings -> 2D float32 NumPy array, one row per molecule.
    How you turn a SMILES string into numbers is up to you.
    Include any custom imports needed in this file and don't forget to
    update the requirements.txt with the library and exact version,
    which will need to be pre-approved by Peter if deviating from the
    versions indicated in that file.
    """
    
    features = []

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=True,
    )

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi!r}")

        # Morgan fingerprint
        fp = gen.GetFingerprint(mol)
        # fp = np.zeros(2048, dtype=np.float32)
        
        # Global physicochemical features
        descriptors = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Chem.GetFormalCharge(mol),
        ], dtype=np.float32)

        features.append(np.concatenate([fp, descriptors]))
    
    return np.asarray(features, dtype=np.float32)


## Parameter Grid - set to the optimal single values 
TREES = 250
DEPTH = 100


def build_model():
    ## TODO 4: Return your model using RandomForestClassifier
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=TREES,
            max_depth=DEPTH,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    )


def save_onnx(model, sample):
    ## reduce tree to fit less than 50MB onnx model
    from skl2onnx import to_onnx
    Path("Part1_%s.onnx" % NAME).write_bytes(
        to_onnx(model, sample, options={"zipmap": False}, target_opset=12).SerializeToString())
    print("\nwrote Part1_%s.onnx" % NAME)


def load_split():
    df = pd.read_csv("data/hw1data.tab", sep="\t")
    df.columns = ["Drug_ID", "Drug", "Y"]
    test = df.sample(frac=0.2, random_state=42)
    rest = df.drop(test.index)
    valid = rest.sample(frac=0.125, random_state=1)
    train = rest.drop(valid.index)
    return {
        "train": train,
        "valid": valid,
        "test": test
    }


def prepare(frame):
    return np.asarray(featurize(frame["Drug"].tolist()), np.float32), frame["Y"].to_numpy()


def report(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print("\nthreshold %.2f" % threshold)
    for name, value in [("accuracy", accuracy_score(y, pred)),
                        ("sensitivity/recall", recall_score(y, pred)),
                        ("specificity", tn / (tn + fp)),
                        ("precision", precision_score(y, pred)),
                        ("f1", f1_score(y, pred)),
                        ("ROC-AUC", roc_auc_score(y, probability)),
                        ("PR-AUC", average_precision_score(y, probability))]:
        print("  %-20s%.3f" % (name, value))


def main():
    split = load_split()
    X_train, y_train = prepare(pd.concat([split["train"], split["valid"]]))
    X_test, y_test = prepare(split["test"])

    model = build_model().fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    report(y_test, probability, 0.5)
    report(y_test, probability, THRESHOLD)
    save_onnx(model, X_test[:1])


if __name__ == "__main__":
    main()
    
  
