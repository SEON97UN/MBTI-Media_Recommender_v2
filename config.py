"""
config.py
Central configuration of data and model file paths.
Update these paths if you change folder structure.
"""

# === Data Files ===
MEDIA_DATA_PATH = "data/media_data.csv"       # 콘텐츠 메타데이터
MBTI_USER_PATH  = "data/MBTI 500.csv"         # 사용자 MBTI+posts 데이터

# === Pre-computed Embeddings ===
CONTENT_EMBEDDINGS_PATH = "data/content_embeddings.pkl"    # 콘텐츠 임베딩 (LaBSE)
MBTI_EMBEDDINGS_PATH    = "data/mbti_embeddings.pkl"       # MBTI 임베딩 (사전 계산)
USER_CLUSTERS_PATH      = "data/user_clusters.pkl"         # (선택) 사용자 군집 결과

# === Feature Outputs ===
GENRE_FEATURE_PATH      = "data/genre_features.pkl"        # 장르 멀티핫/임베딩 특징
RATING_SCORE_PATH       = "data/rating_scores.pkl"         # 평점 신뢰도 보정 점수
CBF_SCALED_INPUT_PATH   = "data/cbf_model_input_scaled.pkl"# 콘텐츠 기반 추천 스케일링 데이터

# === Recommendation Weights ===
# 초기에는 임의값 → 추후 Optuna 등으로 자동 최적화 가능
W_MBTI    = 0.5   # 사용자 MBTI 임베딩 유사도
W_SIMILAR = 0.3   # 사용자가 선택한 콘텐츠 유사도 (재추천 시)
W_GENRE   = 0.2   # 장르 선호도
W_RATING  = 0.05  # 평점 신뢰도 가중