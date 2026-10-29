"""Public Streamlit interface for the CDD 203 hERG classifier."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rdkit import Chem
import streamlit as st


ROOT = Path(__file__).resolve().parent
PART1_PATH = ROOT / "Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.py"
MODEL_PATH = ROOT / "Part1_ELLIOTT-OU-JIEUN-LEE-MATTHEW-ANDERSON-HYUNSOO-RYU.onnx"


@st.cache_resource
def load_model():
    """Load the submitted featurizer and ONNX model once per app process."""
    spec = spec_from_file_location("part1", PART1_PATH)
    part1 = module_from_spec(spec)
    spec.loader.exec_module(part1)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return part1, session, session.get_inputs()[0]


def score_smiles(smiles, part1, session, model_input):
    """Return P(hERG blocker) from the exact submitted feature function and ONNX model."""
    features = part1.featurize([smiles])
    dtype = {"tensor(double)": np.float64, "tensor(int64)": np.int64}.get(model_input.type, np.float32)
    output = np.asarray(session.run(None, {model_input.name: np.asarray(features, dtype)})[-1], float)
    return float(output[0, 1] if output.ndim == 2 and output.shape[1] == 2 else output.ravel()[0])


st.set_page_config(page_title="hERG Blockade Classifier", page_icon="🧬", layout="centered")
st.title("hERG Blockade Classifier")
st.caption("Enter a SMILES string to estimate hERG-blocker probability.")

try:
    part1, session, model_input = load_model()
except Exception as error:
    st.error(f"The model could not load: {error}")
    st.stop()

smiles = st.text_area("SMILES", value="CC(=O)Oc1ccccc1C(=O)O", height=100)
smiles = smiles.strip()
mol = Chem.MolFromSmiles(smiles) if smiles else None
is_valid = mol is not None

if not smiles:
    st.info("Enter a SMILES string to enable prediction.")
elif not is_valid:
    st.error("Enter a valid SMILES string. Prediction is disabled until the structure is valid.")
else:
    st.success("Valid SMILES. Ready to predict.")

if st.button("Predict hERG blockade", type="primary", disabled=not is_valid):
    try:
        st.session_state["probability"] = score_smiles(smiles, part1, session, model_input)
        st.session_state["scored_smiles"] = smiles
    except Exception as error:
        st.error(f"The model could not score this molecule: {error}")

if is_valid and st.session_state.get("scored_smiles") == smiles and "probability" in st.session_state:
    probability = st.session_state["probability"]
    prediction = "Potential hERG blocker" if probability >= part1.THRESHOLD else "Not flagged as a hERG blocker"
    left, right = st.columns(2)
    left.metric("Blocker probability", f"{probability:.1%}")
    right.metric("Predicted class", prediction)
    st.caption(f"Decision threshold: {part1.THRESHOLD:.2f}")
