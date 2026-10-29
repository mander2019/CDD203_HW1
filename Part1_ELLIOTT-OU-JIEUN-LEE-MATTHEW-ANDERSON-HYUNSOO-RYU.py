"""
Homework 1, Part 1 - hERG blockade classifier.

Rename to Part1_[GROUP_NAMES].py, fill in the four TODOs, 
then run: python Part1_[GROUP_NAMES].py
"""

from pathlib import Path
import gzip
import shutil
import zipfile
import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Lipinski, rdMolDescriptors


RANDOM_SEED = 42

# TODO 1: your group's names -> Part1_FIRST1-LAST1-FIRST2-LAST2.onnx
NAME = "ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU"

# TODO 2: your own threshold; justify it in README.md
THRESHOLD = 0.35

# A threshold of 0.35 will make it easier to label a compound as a hERG blocker. This will lose us some true negatives, but will help us find more true positives. This was chosen because the result of a false negative is someone being harmed by an incorrectly labeled drug, causing active harm. A false positive will only prevent a drug from being used out of an abundance of caution, which is a lost good.

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

        fp = np.asarray(gen.GetFingerprint(mol), dtype=np.float32)

        atoms = mol.GetAtoms()
        positive_atoms = sum(atom.GetFormalCharge() > 0 for atom in atoms)
        negative_atoms = sum(atom.GetFormalCharge() < 0 for atom in atoms)
        halogen_count = sum(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in atoms)

        descriptors = np.asarray([
            # Size, lipophilicity, and polarity
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolMR(mol),
            rdMolDescriptors.CalcTPSA(mol),
            mol.GetNumHeavyAtoms(),

            # Hydrogen-bonding and heteroatom features
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHeteroatoms(mol),
            Lipinski.NHOHCount(mol),
            Lipinski.NOCount(mol),

            # Topology and flexibility
            Lipinski.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            Lipinski.NumAliphaticRings(mol),
            Lipinski.NumSaturatedRings(mol),
            Lipinski.NumAromaticCarbocycles(mol),
            Lipinski.NumAromaticHeterocycles(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),

            # Charge and atom-composition features
            Chem.GetFormalCharge(mol),
            positive_atoms,
            negative_atoms,
            halogen_count,
        ], dtype=np.float32)

        features.append(np.concatenate([fp, descriptors]))
    
    return np.asarray(features, dtype=np.float32)


def build_model(seed=RANDOM_SEED):
    ## TODO 4: Return your model using RandomForestClassifier
    
    model = RandomForestClassifier(
                n_estimators=250,
                random_state=seed,
                class_weight='balanced',
                n_jobs=-1
            )

    return model


def save_onnx(model, sample):
    ## reduce tree to fit less than 50MB onnx model
    onnx_path = Path("Part1_%s.onnx" % NAME)
    onnx_path.write_bytes(
        to_onnx(model, sample, options={"zipmap": False}, target_opset=12).SerializeToString())
    print("\nwrote %s" % onnx_path)
    return onnx_path


def package_web_app(model, sample):
    """Export the trained model, refresh the Space files, and build Part 2.zip."""
    onnx_path = save_onnx(model, sample)
    space_dir = Path("space-static")
    if not space_dir.is_dir():
        raise FileNotFoundError("Expected the web app folder at %s" % space_dir)

    space_model = space_dir / "model.onnx"
    shutil.copy2(onnx_path, space_model)
    with space_model.open("rb") as source, gzip.open(space_dir / "model.onnx.gz", "wb", compresslevel=9) as target:
        shutil.copyfileobj(source, target)

    archive_path = Path("Part 2.zip")
    web_app_files = [
        Path("streamlit_app.py"),
        Path("requirements.txt"),
        Path("runtime.txt"),
        Path("Part1_%s.py" % NAME),
        onnx_path,
    ]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in web_app_files:
            if not path.is_file():
                raise FileNotFoundError("Expected the Streamlit web app file at %s" % path)
            archive.write(path, path.name)
    print("wrote %s" % archive_path)


def load_split(seed=RANDOM_SEED):
    df = pd.read_csv("data/hw1data.tab", sep="\t")
    df.columns = ["Drug_ID", "Drug", "Y"]
    test = df.sample(frac=0.2, random_state=seed)
    rest = df.drop(test.index)
    valid = rest.sample(frac=0.125, random_state=seed)
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
    package_web_app(model, X_test[:1])


if __name__ == "__main__":
    main()
