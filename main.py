"""
main.py
Hybrid Recommendation (MBTI + 장르 + 평점 + 초기 선호 콘텐츠)
- 사용자 입력: MBTI / 선호 장르 / 초기 선호 콘텐츠(자동완성 1회)
- 1차 추천: MBTI + 장르 + 평점 + 초기 선호 콘텐츠 유사도
- 2차 추천: 1차 추천 + 사용자가 선택한 콘텐츠와의 유사도
"""

import pickle
import pandas as pd
from embedder import get_user_embedding_hybrid
from recommender_engine import recommend_contents_combined_excluding_with_popularity_filter
from genre_features import build_genre_matrix
from scoring import add_weighted_rating
from config import (
    MEDIA_DATA_PATH, CONTENT_EMBEDDINGS_PATH,
    USER_CLUSTERS_PATH, W_MBTI, W_GENRE, W_RATING, W_SIMILAR
)

# ---------- 데이터 로드 ----------
print("[INFO] Loading content data...")
content = pd.read_csv(MEDIA_DATA_PATH)

with open(CONTENT_EMBEDDINGS_PATH, "rb") as f:
    content_embeddings_dict = pickle.load(f)

# 장르 멀티핫 벡터 & 평점 점수
vocab, genre_matrix = build_genre_matrix(content)
content = add_weighted_rating(content, q=0.70)

# 사용자 클러스터 (선택적)
try:
    with open(USER_CLUSTERS_PATH, "rb") as f:
        user_clusters = pickle.load(f)
    centroids = user_clusters.get("centroids", None)
    print("[INFO] User cluster centroids loaded.")
except Exception:
    centroids = None

# ---------- 사용자 입력 ----------
user_mbti = input("당신의 MBTI를 입력하세요 (예: INFP): ").strip().upper()
preferred_genres = [
    g.strip() for g in input(
        "선호 장르를 입력하세요 (쉼표로 구분, 없으면 Enter): "
    ).split(",") if g.strip()
]

# 🔍 간단한 자동완성 함수
def fuzzy_search_titles(query: str, all_titles: list[str], limit: int = 10):
    q = query.lower()
    return [t for t in all_titles if q in t.lower()][:limit]

# ----- 초기 선호 콘텐츠 입력 (단일 단계) -----
initial_titles = []
query = input("\n초기에 좋아하는 콘텐츠 제목 일부를 입력하세요 (없으면 Enter): ").strip()
if query:
    candidates = fuzzy_search_titles(query, list(content['Title'].values))
    if candidates:
        print(f"후보: {', '.join(candidates)}")
        chosen_list = [
            t.strip() for t in input(
                "추가할 정확한 제목을 모두 입력하세요 (쉼표 구분, 없으면 Enter): "
            ).split(",") if t.strip()
        ]
        initial_titles = [t for t in chosen_list if t in content['Title'].values]
        for t in initial_titles:
            print(f"추가됨 → {t}")
    else:
        print("일치하는 후보가 없습니다.")

# ---------- 사용자 임베딩 ----------
user_embedding = get_user_embedding_hybrid(user_mbti, posts_text="", alpha=1.0)

# ---------- 1차 추천 ----------
initial_recs = recommend_contents_combined_excluding_with_popularity_filter(
    user_embedding=user_embedding,
    preferred_contents=initial_titles,
    previous_recommendations=[],
    initial_preferred_contents=initial_titles,
    content_embeddings_dict=content_embeddings_dict,
    content=content,
    genre_matrix=genre_matrix,
    user_cluster_centroids=centroids,
    top_n=10,
    w_mbti=W_MBTI,
    w_genre=W_GENRE,
    w_rating=W_RATING,
    w_similar=W_SIMILAR,
    enable_similar=bool(initial_titles),
    preferred_genres=preferred_genres,
    genre_vocab=vocab
)

print("\n=== 1차 추천 결과 ===")
for idx, (title, score) in enumerate(initial_recs, 1):
    print(f"{idx}. {title} (Score: {score:.3f})")

# ---------- 2차 재추천 ----------
liked_titles = [
    t.strip() for t in input(
        "\n첫 추천 중 마음에 드는 콘텐츠를 입력하세요 (쉼표로 구분, 없으면 Enter): "
    ).split(",") if t.strip()
]

if liked_titles:
    second_recs = recommend_contents_combined_excluding_with_popularity_filter(
        user_embedding=user_embedding,
        preferred_contents=liked_titles,
        previous_recommendations=[cid for cid, _ in initial_recs],
        initial_preferred_contents=initial_titles,
        content_embeddings_dict=content_embeddings_dict,
        content=content,
        genre_matrix=genre_matrix,
        user_cluster_centroids=centroids,
        top_n=10,
        w_mbti=W_MBTI,
        w_genre=W_GENRE,
        w_rating=W_RATING,
        w_similar=W_SIMILAR,
        enable_similar=True,
        preferred_genres=preferred_genres,
        genre_vocab=vocab
    )
    print("\n=== 2차 재추천 결과 ===")
    for idx, (title, score) in enumerate(second_recs, 1):
        print(f"{idx}. {title} (Score: {score:.3f})")
else:
    print("\n재추천을 건너뛰었습니다.")
