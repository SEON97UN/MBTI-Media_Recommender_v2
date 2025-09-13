"""
cluster_test.py
---------------
MBTI 500.csv의 posts 텍스트를 LaBSE 임베딩으로 벡터화한 뒤
KMeans 군집 품질 지표와 시각화를 수행하여
user_clusters.pkl 생성 여부를 판단하기 위한 독립 실행 스크립트입니다.

실행 결과:
1) Silhouette Score / Davies-Bouldin Index 출력
2) t-SNE 2D 산점도(군집 색상) 시각화
3) (옵션) 품질이 충분하면 user_clusters.pkl 저장
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from embedder import get_user_embedding_from_posts   # LaBSE 임베딩
from config import MBTI_USER_PATH, USER_CLUSTERS_PATH

# ==============================
# 1. 데이터 로드 & 임베딩 생성
# ==============================
print("[INFO] Loading MBTI user data...")
df = pd.read_csv(MBTI_USER_PATH).dropna(subset=["posts"]).reset_index(drop=True)

print("[INFO] Generating LaBSE embeddings for posts...")
embs = np.vstack([get_user_embedding_from_posts(text) for text in df["posts"]])

# ==============================
# 2. KMeans 군집 및 품질 지표
# ==============================
k = 16  # MBTI 유형 수와 동일하게 설정 (실험적으로 조정 가능)
print(f"[INFO] Running KMeans clustering with k={k}...")
km = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = km.fit_predict(embs)

sil = silhouette_score(embs, labels)
dbi = davies_bouldin_score(embs, labels)
print("\n=== Cluster Quality Metrics ===")
print(f"Silhouette Score     : {sil:.3f}  (0.3 이상이면 양호)")
print(f"Davies-Bouldin Index : {dbi:.3f}  (낮을수록 양호)")

# ==============================
# 3. 2D 시각화 (t-SNE)
# ==============================
print("[INFO] Running t-SNE for 2D visualization (may take a while)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(embs)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab20", s=25)
plt.title("KMeans Clusters on MBTI Posts (t-SNE 2D)")
plt.colorbar(scatter, label="Cluster ID")
plt.tight_layout()
plt.savefig("cluster_visualization.png", dpi=300)
plt.show()

# ==============================
# 4. 저장 여부 선택
# ==============================
save_choice = input(
    "\nSave cluster centroids for recommendation boosting? (y/n): "
).strip().lower()

if save_choice == "y":
    out = {"centroids": km.cluster_centers_}
    with open(USER_CLUSTERS_PATH, "wb") as f:
        pickle.dump(out, f)
    print(f"[INFO] Saved cluster centroids to {USER_CLUSTERS_PATH}")
else:
    print("[INFO] Clusters not saved. You can rerun this script anytime.")
