"""Public Streamlit interface for the CDD 203 hERG classifier."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random

import onnxruntime as ort
from rdkit import Chem
from rdkit.Chem import Draw
import streamlit as st
from predictor import score_smiles


ROOT = Path(__file__).resolve().parents[1]
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


def random_smiles():
    """Generate a varied, RDKit-valid SMILES string."""
    substituents = ["C", "CC", "CCC", "O", "OC", "N", "NC", "F", "Cl", "Br", "C(F)(F)F", "C(=O)O", "C(=O)N"]
    terminal_groups = ["", "O", "N", "F", "Cl", "Br", "C(=O)O", "C(=O)N"]
    for _ in range(40):
        choice = random.randrange(6)
        if choice == 0:
            candidate = "C" * random.randint(2, 12) + random.choice(terminal_groups)
        elif choice == 1:
            candidate = "C" * random.randint(1, 5) + "C(C)" + "C" * random.randint(1, 5)
        elif choice == 2:
            candidate = f"c1cc({random.choice(substituents)})c({random.choice(substituents)})cc1{random.choice(substituents)}"
        elif choice == 3:
            candidate = f"n1cc({random.choice(substituents)})ccc1{random.choice(substituents)}"
        elif choice == 4:
            candidate = "C1CCCCC1" + random.choice(terminal_groups)
        else:
            candidate = "N1CCNCC1" + random.choice(terminal_groups)
        molecule = Chem.MolFromSmiles(candidate)
        if molecule is not None:
            return Chem.MolToSmiles(molecule)
    raise RuntimeError("Could not generate a valid SMILES string.")


def set_random_smiles():
    st.session_state["smiles"] = random_smiles()


st.set_page_config(page_title="hERG Blockade Classifier", page_icon="🧬", layout="centered")
st.title("hERG Blockade Classifier")
st.caption("Enter a SMILES string to estimate hERG-blocker probability.")

try:
    part1, session, model_input = load_model()
except Exception as error:
    st.error(f"The model could not load: {error}")
    st.stop()

if "smiles" not in st.session_state:
    st.session_state["smiles"] = "CC(=O)Oc1ccccc1C(=O)O"
smiles = st.text_area("SMILES", height=100, key="smiles")
smiles = smiles.strip()
mol = Chem.MolFromSmiles(smiles) if smiles else None
is_valid = mol is not None

if not smiles:
    st.info("Enter a SMILES string to enable prediction.")
elif not is_valid:
    st.error("Enter a valid SMILES string. Prediction is disabled until the structure is valid.")
else:
    st.success("Valid SMILES. Ready to predict.")

randomize_column, predict_column = st.columns(2)
randomize_column.button("Randomize SMILES", on_click=set_random_smiles, use_container_width=True)
predict_clicked = predict_column.button("Predict hERG blockade", type="primary", disabled=not is_valid,
                                        use_container_width=True)

if is_valid:
    st.image(Draw.MolToImage(mol, size=(420, 260)), caption="SMILES structure")

if predict_clicked:
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
