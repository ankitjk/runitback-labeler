"""The small scene model: preprocessing and ONNX inference.

Preprocessing copies RunItBack's runitback/cv/scene_classifier.py so the model
sees frames exactly as it did in training.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from labeler.classes import CLASSES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """One OpenCV BGR frame -> a (3, 224, 224) float32 array."""
    resized = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)


def class_names_from_mapping(mapping: dict, n_outputs: int) -> list[str]:
    """The model's output order, from class_mapping.json ({"0": "both_airborne", ...})."""
    if len(mapping) != n_outputs:
        raise ValueError(f"class_mapping.json has {len(mapping)} classes but the model has {n_outputs} outputs")
    names = [mapping[str(i)] for i in range(n_outputs)]
    unknown = set(names) - set(CLASSES)
    if unknown:
        raise ValueError(f"class_mapping.json names classes this project does not know: {sorted(unknown)}")
    return names


class SceneModel:
    def __init__(self, model_dir: Path):
        model_file = model_dir / "scene_classifier.onnx"
        self.session = ort.InferenceSession(str(model_file))
        # The output order comes from the file training wrote, never from a list in code.
        mapping = json.loads((model_dir / "class_mapping.json").read_text())
        self.class_names = class_names_from_mapping(mapping, self.session.get_outputs()[0].shape[1])
        self.fingerprint = hashlib.sha256(model_file.read_bytes()).hexdigest()[:12]

    def predict(self, frames_bgr: list[np.ndarray]) -> list[tuple[str, float]]:
        """Return (class name, confidence) for each frame."""
        batch = np.stack([preprocess(frame) for frame in frames_bgr])
        logits = self.session.run(None, {"input": batch})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        best = probabilities.argmax(axis=1)
        return [(self.class_names[i], float(probabilities[row, i])) for row, i in enumerate(best)]
