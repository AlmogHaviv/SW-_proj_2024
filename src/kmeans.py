import math
import sys

EPSILON = 0.0001

def kmeans_calc(k, itr, datapoints):
    try:
        global EPSILON
        centroids = datapoints[:k]
        converged = False

        while not converged and itr > 0:
            # k groups
            groups = [[] for _ in range(len(centroids))]

            for point in datapoints:
                point_min = [0, float("inf")]
                for idx, centroid in enumerate(centroids):
                    dist = euclidean_dist(centroid, point)
                    if dist < point_min[1]:
                        point_min[0] = idx
                        point_min[1] = dist
                groups[point_min[0]].append(point)

            # Update the centroids
            centroids, converged = update_centroids(centroids, groups)
            itr -= 1

        return centroids
    
    except Exception:
        # Catch any exception and print an error message
        print(f"An Error Has Occurred")


def euclidean_dist(centroid, point):
    lst_of_diff = [math.pow((centroid[i] - point[i]), 2) for i in range(len(point))]
    dist = math.sqrt(sum(lst_of_diff))
    return dist


def update_centroids(centroids, groups):
    max_change = 0
    for i in range(len(groups)):
        length = len(groups[i])
        tmp = [0 for i in range(len(centroids[0]))]
        for j in range(len(groups[i])):
            for m in range(len(groups[i][j])):
                tmp[m] += groups[i][j][m]
        groups[i] = [point / length for point in tmp]

    for idx, centroid in enumerate(centroids):
        max_change = max(max_change, euclidean_dist(centroid, groups[idx]))

    if max_change < EPSILON:
        return groups, True

    return groups, False



