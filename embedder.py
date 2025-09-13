"""
embedder.py
Embedding utilities for MBTI and content text using LaBSE.
Supports optional hybrid user vectors (MBTI + Posts + Genre).
"""

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text  # noqa: F401  # required by LaBSE
import numpy as np
from typing import List, Optional
from config import W_GENRE  # 장르 가중치 (config.py에서 관리)


class LaBSEEmbedder:
    """
    Singleton LaBSE encoder for MBTI text, user posts, or content overview.
    """

    _singleton = None

    def __init__(self):
        # smaller_LaBSE : multilingual (15 languages including English/Korean)
        self.encoder = hub.KerasLayer(
            "https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1"
        )
        self.preprocessor = hub.KerasLayer(
            "https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang_preprocess/1"
        )
        self.model = self._build_model()

    @classmethod
    def get(cls):
        """Singleton pattern to avoid reloading heavy model."""
        if cls._singleton is None:
            cls._singleton = LaBSEEmbedder()
        return cls._singleton

    def _build_model(self):
        input_text = tf.keras.layers.Input(shape=(), dtype=tf.string)
        encoder_inputs = self.preprocessor(input_text)
        outputs = self.encoder(encoder_inputs)["pooled_output"]
        norm_outputs = tf.nn.l2_normalize(outputs, axis=-1)
        return tf.keras.Model(input_text, norm_outputs)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized LaBSE embeddings for a list of texts."""
        texts = [t if isinstance(t, str) else "" for t in texts]
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = tf.constant(texts[i:i + batch_size])
            batch_embeddings = self.model(batch).numpy()
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)


# === Basic Interfaces ===
def get_user_embedding(mbti_str: str) -> np.ndarray:
    """Embed a single MBTI string."""
    embedder = LaBSEEmbedder.get()
    return embedder.embed([mbti_str])[0]


def get_user_embedding_from_posts(posts_text: str) -> np.ndarray:
    """Embed full user posts text."""
    embedder = LaBSEEmbedder.get()
    return embedder.embed([posts_text])[0]


# === Hybrid Interface ===
def get_user_embedding_hybrid(
    mbti_str: str,
    posts_text: str = "",
    genre_vector: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Hybrid user vector
    = alpha * MBTI embedding + (1-alpha) * Posts embedding (+ optional Genre).
    - If posts_text is empty, fall back to MBTI only.
    - If genre_vector is provided, it is concatenated (weighted) then re-normalized.
    """
    v1 = get_user_embedding(mbti_str)

    # Posts 임베딩 (없으면 MBTI로 대체)
    if posts_text and posts_text.strip():
        v2 = get_user_embedding_from_posts(posts_text)
    else:
        v2 = v1

    vec = alpha * v1 + (1.0 - alpha) * v2

    # 장르 벡터(멀티핫 또는 임베딩 평균)가 있으면 concat 후 normalize
    if genre_vector is not None:
        vec = np.concatenate([vec, W_GENRE * genre_vector])

    n = np.linalg.norm(vec) + 1e-12
    return vec / n
