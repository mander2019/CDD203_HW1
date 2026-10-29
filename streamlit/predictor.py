"""Shared scoring logic used by the Streamlit interface and its regression test."""

import numpy as np


def score_smiles(smiles, part1, session, model_input):
    """Return P(hERG blocker) using the submitted Python featurizer and ONNX model."""
    features = part1.featurize([smiles])
    dtype = {"tensor(double)": np.float64, "tensor(int64)": np.int64}.get(model_input.type, np.float32)
    output = np.asarray(session.run(None, {model_input.name: np.asarray(features, dtype)})[-1], float)
    return float(output[0, 1] if output.ndim == 2 and output.shape[1] == 2 else output.ravel()[0])
