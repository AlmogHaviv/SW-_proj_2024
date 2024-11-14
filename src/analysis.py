import sys
import numpy as np
from sklearn.metrics import silhouette_score
from symnmf import symnmf_construct
from kmeans import kmeans_calc, euclidean_dist

# Default number of iterations for KMeans
DEFAULT_ITER = 300

def apply_kmeans(data, k, vectors):
    """
    Apply KMeans clustering and return the silhouette score.
    """
    # Calculate KMeans centroids
    centroids = kmeans_calc(k, DEFAULT_ITER, data)

    # Assign clusters based on the closest centroids
    clusters = get_clusters(centroids, vectors)

    # Compute silhouette score
    score = silhouette_score(data, clusters)
    return score


def get_clusters(centroids, vectors):
    """
    Assign each data point to the closest centroid to form clusters.
    """
    labels = np.zeros(len(vectors), dtype=int)

    for i, vector in enumerate(vectors):
        # Calculate the distance to each centroid and find the closest one
        distances = [euclidean_dist(vector, centroid) for centroid in centroids]
        labels[i] = np.argmin(distances)

    return labels


def main():
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)

    k = int(sys.argv[1])
    file_path = sys.argv[2]

    # Load data from the specified file
    try:
        data = np.loadtxt(file_path, delimiter=',')
        vectors = data.tolist()
    except:
        print("An Error Has Occurred")
        sys.exit(1)

    # Get the number of data points
    n = len(vectors)

    # Perform SymNMF clustering and compute the silhouette score
    symnmf_mat = np.array(symnmf_construct(vectors, k, n))
    nmf_clusters = np.argmax(symnmf_mat, axis=1)
    nmf_score = silhouette_score(data, nmf_clusters)

    # Compute silhouette score for KMeans clustering
    kmeans_score = apply_kmeans(data, k, vectors)

    # Print the results
    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")


if __name__ == "__main__":
    main()
