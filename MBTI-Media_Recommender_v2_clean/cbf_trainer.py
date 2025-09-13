"""
cbf_trainer.py
Scale numerical features (Rating Value, Rating Count) for content-based filtering
and save to pickle for fast loading.
Improved to support rating reliability and optional MinMax scaling.
"""

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import MEDIA_DATA_PATH, CBF_SCALED_INPUT_PATH, RATING_SCORE_PATH


def compute_rating_reliability(df: pd.DataFrame) -> pd.Series:
    """
    평점 신뢰도 계산
    - 단순 예시: Rating Count / max(Rating Count)
    - 추후 Bayesian 평균 등으로 교체 가능
    """
    count = df["Rating Count"].fillna(0)
    return count / (count.max() if count.max() > 0 else 1)


def build_cbf_features(use_minmax: bool = False):
    # 1. 콘텐츠 데이터 로드
    df = pd.read_csv(MEDIA_DATA_PATH)

    # === 평점 신뢰도 계산 ===
    rating_reliability = compute_rating_reliability(df)
    with open(RATING_SCORE_PATH, "wb") as f:
        pickle.dump(rating_reliability.to_dict(), f)
    print(f"[INFO] Saved rating reliability to {RATING_SCORE_PATH}")

    # === 기본 평점 Feature 스케일링 ===
    features = ["Rating Value", "Rating Count"]
    X = df[features].fillna(0).values

    # 선택적 스케일러
    scaler = MinMaxScaler() if use_minmax else StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cbf_scaled = {
        "titles": df["Title"].tolist(),
        "scaled_features": X_scaled,
        "scaler": scaler,
        "feature_names": features
    }

    with open(CBF_SCALED_INPUT_PATH, "wb") as f:
        pickle.dump(cbf_scaled, f)

    print(f"[INFO] Saved scaled CBF features to {CBF_SCALED_INPUT_PATH}")


if __name__ == "__main__":
    # 기본적으로 StandardScaler 사용
    build_cbf_features()
