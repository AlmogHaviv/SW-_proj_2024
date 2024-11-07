import sys
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf_module
from kmeans import kmeans_calc, euclidean_dist

# Default number of iterations for KMeans
DEFAULT_ITER = 300

def apply_kmeans(data, k, vectors):
    """
    Apply KMeans clustering and return the silhouette score.

    Parameters:
    - data (ndarray): Array of data points, shape (n_samples, n_features).
    - k (int): Number of clusters.
    - vectors (list): Data points in list format.

    Returns:
    - score (float): Silhouette score for KMeans clustering.
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

    Parameters:
    - centroids (ndarray): Array of centroid coordinates, shape (k, n_features).
    - vectors (ndarray): Array of data points, shape (n_samples, n_features).

    Returns:
    - labels (ndarray): Array of cluster assignments, where each index i
      contains the cluster number for point i in vectors.
    """
    labels = np.zeros(len(vectors), dtype=int)

    for i, vector in enumerate(vectors):
        # Calculate the distance to each centroid and find the closest one
        distances = [euclidean_dist(vector, centroid) for centroid in centroids]
        labels[i] = np.argmin(distances)

    return labels


def main():
    # Check for correct command-line usage
    if len(sys.argv) != 3:
        print("Usage: python3 analysis.py <number_of_clusters> <file_path>")
        sys.exit(1)

    # Parse arguments
    k = int(sys.argv[1])
    file_path = sys.argv[2]

    # Load data from the specified file
    try:
        data = np.loadtxt(file_path, delimiter=',')
        vectors = data.tolist()
    except:
        print("An error occurred while reading the data file.")
        sys.exit(1)

    # Get the number of data points
    n = len(vectors)

    # Perform SymNMF clustering and compute the silhouette score
    symnmf = np.array(symnmf_module.symnmf_construct(vectors, k, n))
    nmf_clusters = np.argmax(symnmf, axis=1)
    nmf_score = silhouette_score(data, nmf_clusters)

    # Compute silhouette score for KMeans clustering
    kmeans_score = apply_kmeans(data, k, vectors)

    # Print the results
    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")


if __name__ == "__main__":
    main()
