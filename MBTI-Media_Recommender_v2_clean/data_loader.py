"""
data_loader.py
Load and preprocess content and user data with extended cleaning
and feature preparation (genre multi-hot, rating reliability).
"""

import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from config import (
    MEDIA_DATA_PATH,
    MBTI_USER_PATH,
    GENRE_FEATURE_PATH,
    RATING_SCORE_PATH,
)


def load_content_data(save_features: bool = True) -> pd.DataFrame:
    """
    Load media_data.csv and perform extended cleaning.
    - Drop rows missing critical fields
    - Cast numerical types
    - Generate genre multi-hot vectors
    - Compute rating reliability (optional)
    """
    df = pd.read_csv(MEDIA_DATA_PATH)

    # 필수 컬럼 결측 제거
    required_cols = [
        "Title",
        "Genres",
        "Overview",
        "Rating Value",
        "Rating Count",
    ]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # 타입 캐스팅
    df["Rating Value"] = df["Rating Value"].astype(float)
    df["Rating Count"] = df["Rating Count"].astype(int)

    # === 장르 멀티핫 벡터 생성 ===
    genres_split = df["Genres"].apply(
        lambda x: [g.strip().lower() for g in str(x).split(",")]
    )
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(genres_split)

    if save_features:
        with open(GENRE_FEATURE_PATH, "wb") as f:
            pickle.dump({"titles": df["Title"].tolist(),
                         "genres": mlb.classes_,
                         "matrix": genre_matrix}, f)
        print(f"[INFO] Saved genre features to {GENRE_FEATURE_PATH}")

    # === 평점 신뢰도 계산 (간단 비율) ===
    if save_features:
        reliability = df["Rating Count"] / df["Rating Count"].max()
        with open(RATING_SCORE_PATH, "wb") as f:
            pickle.dump(reliability.to_dict(), f)
        print(f"[INFO] Saved rating reliability to {RATING_SCORE_PATH}")

    return df


def load_user_data() -> pd.DataFrame:
    """
    Load MBTI user posts data.
    - Drop rows without posts or type
    - Reset index
    """
    df = pd.read_csv(MBTI_USER_PATH)
    df = df.dropna(subset=["type", "posts"]).reset_index(drop=True)
    return df
