import numpy as np

def _cos(a, b):
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def recommend_contents_combined_excluding_with_popularity_filter(
    user_embedding,
    preferred_contents,
    previous_recommendations,
    content_embeddings_dict,
    content,
    genre_matrix,
    user_cluster_centroids=None,
    top_n=20,
    w_mbti=0.2,
    w_genre=0.3,
    w_rating=0.2,
    w_similar=0.3,
    enable_similar=False,
    preferred_genres=None,
    genre_vocab=None,
    initial_preferred_contents=None
):
    """
    Hybrid recommendation
    - 1차 추천: enable_similar=False → MBTI + 장르 + 평점 (+초기 선호 콘텐츠)
    - 2차 추천: enable_similar=True  → + 사용자 선택 콘텐츠와의 유사도
    """
    all_preferred_contents = list(
        set(preferred_contents + (initial_preferred_contents or []))
    )

    # --- Candidate pool ---
    popular_embeddings_dict = {
        title: emb for title, emb in content_embeddings_dict.items()
        if title in content['Title'].values or title in all_preferred_contents
    }

    # --- 1. MBTI similarity ---
    mbti_scores = {title: _cos(user_embedding, emb)
                   for title, emb in popular_embeddings_dict.items()}

    # --- 2. Similar content (재추천 전용) ---
    similar_scores = {}
    if enable_similar and preferred_contents:
        for p in preferred_contents:
            if p in content_embeddings_dict:
                p_emb = content_embeddings_dict[p]
                for o, o_emb in popular_embeddings_dict.items():
                    if o not in preferred_contents:
                        sim = _cos(p_emb, o_emb)
                        similar_scores[o] = similar_scores.get(o, 0.0) + sim

    # --- 3. Candidate set ---
    candidates = set(mbti_scores) | set(similar_scores)
    candidates -= set(previous_recommendations) | set(all_preferred_contents)

    # --- 4. Genre similarity ---
    title_to_idx = {t: i for i, t in enumerate(content['Title'].values)}
    if preferred_genres and genre_vocab is not None:
        idx = {g: i for i, g in enumerate(genre_vocab)}
        user_genre_vec = np.zeros((len(genre_vocab),), dtype=np.float32)
        for g in preferred_genres:
            g = g.strip()
            if g in idx:
                user_genre_vec[idx[g]] = 1.0
        user_genre_vec /= (np.linalg.norm(user_genre_vec) + 1e-12)
    else:
        pref_idx = [title_to_idx[p] for p in preferred_contents if p in title_to_idx]
        if pref_idx:
            user_genre_vec = np.mean(genre_matrix[pref_idx, :], axis=0)
            user_genre_vec /= (np.linalg.norm(user_genre_vec) + 1e-12)
        else:
            user_genre_vec = np.zeros((genre_matrix.shape[1],), dtype=np.float32)

    genre_scores = {cid: _cos(user_genre_vec, genre_matrix[title_to_idx[cid]])
                    if cid in title_to_idx else 0.0 for cid in candidates}

    # --- 5. Weighted rating ---
    rating_scores = {cid: float(content.loc[
        content['Title'] == cid, 'WeightedRatingZ'].values[0])
        if cid in content['Title'].values else 0.0 for cid in candidates}

    # --- 6. Cluster affinity ---
    cluster_scores = {}
    if user_cluster_centroids is not None:
        boost = max(_cos(user_embedding, c) for c in user_cluster_centroids)
        cluster_scores = {cid: boost for cid in candidates}
    else:
        cluster_scores = {cid: 0.0 for cid in candidates}

    # --- 7. Z-normalization ---
    def z_norm(d):
        arr = np.array(list(d.values()), dtype=np.float32)
        if len(arr) == 0:
            return {}
        mu, sd = arr.mean(), arr.std() + 1e-8
        return {k: (v - mu) / sd for k, v in d.items()}

    mbtiZ = z_norm(mbti_scores)
    simZ  = z_norm(similar_scores) if enable_similar else {}
    genreZ = z_norm(genre_scores)

    # --- 8. Final combined score ---
    combined = {
        cid: (
            w_mbti * mbtiZ.get(cid, 0.0)
            + (w_similar * simZ.get(cid, 0.0) if enable_similar else 0.0)
            + w_genre * genreZ.get(cid, 0.0)
            + w_rating * rating_scores.get(cid, 0.0)
        )
        for cid in candidates
    }

    # --- 9. Minimum rating filter ---
    final = {cid: score for cid, score in combined.items()
             if content.loc[content['Title'] == cid, 'Rating Value'].values[0] >= 3.5}

    return sorted(final.items(), key=lambda x: x[1], reverse=True)[:top_n]
