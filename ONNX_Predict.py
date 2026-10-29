"""Print ONNX hERG-blocker probabilities and classifications for SMILES strings.

Run from the repository root:
    python TestStreamlit.py
    python TestStreamlit.py "CCO"
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
PART1_PATH = ROOT / "Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.py"
MODEL_PATH = ROOT / "Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.onnx"
SAMPLES = [
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN(C)CCOC(c1ccccc1)c1ccccc1.Cl",
    "C[NH2+]CC[C@@H](Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",
    "CC(C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO",
]


def load_part1():
    spec = importlib.util.spec_from_file_location("part1", PART1_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_smiles(smiles, part1, session, model_input):
    features = part1.featurize([smiles])
    dtype = {"tensor(double)": np.float64, "tensor(int64)": np.int64}.get(model_input.type, np.float32)
    output = np.asarray(session.run(None, {model_input.name: np.asarray(features, dtype)})[-1], float)
    return float(output[0, 1] if output.ndim == 2 and output.shape[1] == 2 else output.ravel()[0])


part1 = load_part1()
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
model_input = session.get_inputs()[0]

print("P(blocker)  class                  SMILES")
for smiles in SAMPLES + sys.argv[1:]:
    probability = score_smiles(smiles, part1, session, model_input)
    classification = "potential blocker" if probability >= part1.THRESHOLD else "not flagged"
    print(f"{probability:.3f}      {classification:<22} {smiles}")
