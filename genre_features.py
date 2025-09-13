"""
genre_features.py
Create multi-hot or weighted genre vectors from content data.

- Default: L2 normalized multi-hot vectors
- Optional: weight each genre (e.g., popularity or custom weight)
"""

import pandas as pd
import numpy as np
import re
import pickle
from config import GENRE_FEATURE_PATH, W_GENRE  # W_GENRE: 장르 임베딩 가중치


def _split_genres(g):
    if pd.isna(g):
        return []
    return [s.strip() for s in re.split(r"[,\|/]", str(g)) if s.strip()]

def build_genre_matrix(
    content: pd.DataFrame,
    normalize: str = "l2",
    save: bool = True,
    weight: float = W_GENRE,
):
    """
    Build genre feature matrix.

    Parameters
    ----------
    content : pd.DataFrame
        Must contain a "Genres" column.
    normalize : {"l2","none"}, default="l2"
        Normalization strategy.
    save : bool, default=True
        Save the result to GENRE_FEATURE_PATH.
    weight : float
        Global multiplier for later hybrid recommendation.

    Returns
    -------
    vocab : list[str]
        Sorted list of unique genres.
    mat : np.ndarray
        (n_contents, n_genres) matrix, normalized if requested.
    """
    all_genres, genre_lists = [], []
    for g in content["Genres"]:
        gl = _split_genres(g)
        genre_lists.append(gl)
        all_genres.extend(gl)

    vocab = sorted(set(all_genres))
    idx = {g: i for i, g in enumerate(vocab)}
    mat = np.zeros((len(content), len(vocab)), dtype=np.float32)

    for row, gl in enumerate(genre_lists):
        for g in gl:
            mat[row, idx[g]] = 1.0

    # === Normalization ===
    if normalize == "l2":
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be 'l2' or 'none'")

    # === Optional Save ===
    if save:
        with open(GENRE_FEATURE_PATH, "wb") as f:
            pickle.dump(
                {"titles": content["Title"].tolist(),
                 "genres": vocab,
                 "matrix": weight * mat},
                f,
            )
        print(f"[INFO] Saved genre features to {GENRE_FEATURE_PATH}")

    return vocab, mat
