import pandas as pd
import numpy as np
import os

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

results = []

# ---------- PREPROCESS ----------
def preprocess(texts):
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    X = tfidf.fit_transform(texts)
    return X, tfidf

# ---------- TOP WORDS ----------
def top_words(model, tfidf, n=10):
    terms = tfidf.get_feature_names_out()
    if hasattr(model, 'cluster_centers_'):
        for i, center in enumerate(model.cluster_centers_):
            words = [terms[ind] for ind in center.argsort()[-n:]]
            print(f"\nCluster {i} top words:", words)

# ---------- ELBOW ----------
def elbow(X, name):
    sse = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        sse.append(km.inertia_)

    plt.figure()
    plt.plot(range(2, 8), sse, marker='o')
    plt.title(f"Elbow - {name}")
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.show()

# ---------- DENDROGRAM ----------
def plot_dendrogram(X, name):
    Z = linkage(X.toarray()[:300], method='ward')
    plt.figure(figsize=(8,4))
    dendrogram(Z)
    plt.title(f"Dendrogram - {name}")
    plt.show()

# ---------- VISUALIZATION ----------
def visualize(X, labels, title):
    X_small = X.toarray()[:1000]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_small)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_small)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels[:1000])
    plt.title(f"PCA - {title}")

    plt.subplot(1,2,2)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=labels[:1000])
    plt.title(f"t-SNE - {title}")

    plt.show()

# ---------- MODELS ----------
def run_models(X, tfidf, name):
    print("\n==========", name, "==========")

    # KMeans
    k = 5
    km = KMeans(n_clusters=k, random_state=42)
    km_labels = km.fit_predict(X)
    km_score = silhouette_score(X, km_labels)
    print("KMeans Silhouette:", km_score)
    top_words(km, tfidf)
    visualize(X, km_labels, "KMeans")
    results.append([name, "KMeans", k, km_score])

    # Hierarchical
    hc = AgglomerativeClustering(n_clusters=k)
    hc_labels = hc.fit_predict(X.toarray())
    hc_score = silhouette_score(X.toarray(), hc_labels)
    print("Hierarchical Silhouette:", hc_score)
    visualize(X, hc_labels, "Hierarchical")
    results.append([name, "Hierarchical", k, hc_score])

    plot_dendrogram(X, name)

    # DBSCAN
    db = DBSCAN(eps=1.5, min_samples=5)
    db_labels = db.fit_predict(X.toarray())

    if len(set(db_labels)) > 1:
        db_score = silhouette_score(X.toarray(), db_labels)
    else:
        db_score = -1

    print("DBSCAN Silhouette:", db_score)
    visualize(X, db_labels, "DBSCAN")
    results.append([name, "DBSCAN", "auto", db_score])

    elbow(X, name)

# ---------- DATASET 1 ----------
data1 = fetch_20newsgroups(subset='all')
texts1 = data1.data[:2000]

X1, tfidf1 = preprocess(texts1)
run_models(X1, tfidf1, "20 Newsgroups")

# ---------- DATASET 2 ----------
imdb_path = r"C:\Users\pragn\OneDrive\Desktop\IMDB Dataset.csv"

if os.path.exists(imdb_path):
    df2 = pd.read_csv(imdb_path)
    texts2 = df2['review'][:2000]

    X2, tfidf2 = preprocess(texts2)
    run_models(X2, tfidf2, "IMDb Reviews")
else:
    print("\n❌ IMDb dataset not found. Fix path.")

# ---------- DATASET 3 ----------
twitter_path = r"C:\Users\pragn\OneDrive\Desktop\training.1600000.processed.noemoticon.csv"

if os.path.exists(twitter_path):
    df3 = pd.read_csv(twitter_path, encoding='latin-1', header=None)
    texts3 = df3[5][:2000]

    X3, tfidf3 = preprocess(texts3)
    run_models(X3, tfidf3, "Sentiment140 Tweets")
else:
    print("\n❌ Twitter dataset not found. Fix path or extract ZIP.")

# ---------- FINAL TABLE ----------
df_results = pd.DataFrame(results, columns=[
    "Dataset", "Algorithm", "Clusters", "Silhouette Score"
])

print("\n===== FINAL COMPARISON TABLE =====")
print(df_results)

# Save results
df_results.to_csv("clustering_results.csv", index=False)