import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
from embedder import get_user_embedding_from_posts

def build_user_clusters(user_csv_path="data/MBTI 500.csv",
                        k=16, random_state=42):
    df = pd.read_csv(user_csv_path).dropna(subset=['posts']).reset_index(drop=True)
    embs = [get_user_embedding_from_posts(t) for t in df['posts']]
    X = np.vstack(embs)
    km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    km.fit(X)
    out = {"centroids": km.cluster_centers_}
    with open("data/user_clusters.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"Saved cluster centroids to data/user_clusters.pkl (k={k})")

if __name__ == "__main__":
    build_user_clusters()
