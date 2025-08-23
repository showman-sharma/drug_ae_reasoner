"""Utilities for encoding text using a SentenceTransformer model."""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
from sentence_transformers import SentenceTransformer


# Default location where a SapBERT model is expected to live. The library no
# longer downloads models automatically; users must provide the path to a
# pre-downloaded model.
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "local_models", "sapbert")

logger = logging.getLogger(__name__)

# Global handle to the loaded model. It will be lazily initialised by
# ``encode_text`` or can be explicitly loaded via :func:`load_model`.
model: SentenceTransformer | None = None

# Simple cache to avoid redundant embeddings.
embedding_cache: Dict[str, np.ndarray] = {}


def load_model(path: str | None = None) -> SentenceTransformer:
    """Load a SapBERT model from ``path``.

    Parameters
    ----------
    path:
        Filesystem path to a directory containing a pre-trained
        :class:`~sentence_transformers.SentenceTransformer` model. If ``None``
        is provided, ``MODEL_DIR`` is used. The function raises a
        :class:`FileNotFoundError` if the path does not exist or does not look
        like a valid model directory.

    Returns
    -------
    SentenceTransformer
        The loaded model instance.
    """

    global model

    model_path = path or MODEL_DIR
    config_file = os.path.join(model_path, "config.json")

    if not os.path.isdir(model_path) or not os.path.exists(config_file):
        raise FileNotFoundError(
            f"SapBERT model not found at '{model_path}'. "
            "Download the model separately and provide its path to `load_model`."
        )

    logger.info("Loading SapBERT model from %s", model_path)
    model = SentenceTransformer(model_path)
    return model


def encode_text(text: str) -> np.ndarray:
    """Encode ``text`` into a normalised vector.

    The underlying model is loaded lazily on first use. Subsequent calls reuse
    both the model and cached embeddings when available.
    """

    global model

    if text in embedding_cache:
        return embedding_cache[text]

    if model is None:
        load_model()

    assert model is not None  # For type checkers

    vec = model.encode([text])[0]
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    normalized = vec / (norm if norm > 0 else 1.0)
    embedding_cache[text] = normalized
    return normalized


__all__ = ["encode_text", "load_model"]

