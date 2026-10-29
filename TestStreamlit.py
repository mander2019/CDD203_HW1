"""Verify that the Streamlit scoring path matches direct ONNX Runtime output.

Run from the repository root:
    python TestStreamlit.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
PART1_PATH = ROOT / "Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.py"
MODEL_PATH = ROOT / "Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.onnx"
PREDICTOR_PATH = ROOT / "streamlit" / "predictor.py"
SAMPLES = [
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN(C)CCOC(c1ccccc1)c1ccccc1.Cl",
    "C[NH2+]CC[C@@H](Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(passed, message):
    print("PASS " if passed else "FAIL ", message)


part1 = load_module("part1", PART1_PATH)
predictor = load_module("streamlit_predictor", PREDICTOR_PATH)
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
model_input = session.get_inputs()[0]
dtype = {"tensor(double)": np.float64, "tensor(int64)": np.int64}.get(model_input.type, np.float32)

for smiles in SAMPLES:
    features = part1.featurize([smiles])
    direct_output = np.asarray(session.run(None, {model_input.name: np.asarray(features, dtype)})[-1], float)
    direct_probability = float(direct_output[0, 1] if direct_output.ndim == 2 else direct_output.ravel()[0])
    app_probability = predictor.score_smiles(smiles, part1, session, model_input)
    check(np.isclose(app_probability, direct_probability, atol=1e-8),
          "Streamlit score matches ONNX for %s" % smiles[:35])
    check((app_probability >= part1.THRESHOLD) == (direct_probability >= part1.THRESHOLD),
          "Streamlit class matches ONNX at threshold %.2f" % part1.THRESHOLD)
