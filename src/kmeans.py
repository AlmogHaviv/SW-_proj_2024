import math
import sys

EPSILON = 0.0001


def main(k, itr, txt):
    try:
        global EPSILON

        datapoints = create_datapoints(txt)

        if not is_valid(k, itr, len(datapoints)):
            return
        k = float(k)
        itr = float(itr)
        k = int(k)
        itr = int(itr)
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

        for point in centroids:
            # Format each number to 4 decimal places and join them with spaces
            formatted_numbers = ",".join(f"{num:.4f}" for num in point)

            # Print the formatted numbers
            print(formatted_numbers)
    except Exception:
        # Catch any exception and print an error message
        print(f"An Error Has Occurred")


def kmeans_calc(k, itr, datapoints):
    try:
        global EPSILON

        if not is_valid(k, itr, len(datapoints)):
            return
        k = float(k)
        itr = float(itr)
        k = int(k)
        itr = int(itr)
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


def create_datapoints(txt):
    datapoints = []
    # Open the file in read mode
    with open(txt, 'r') as file:
        # Read the file line by line
        for line in file:
            line_data = line.strip()
            lst_of_elem = line_data.split(",")
            float_data = [float(i) for i in lst_of_elem]
            datapoints.append(float_data)

    return datapoints


def is_valid(k, itr, n):
    def is_integer(value):
        try:
            return float(value).is_integer()
        except:
            return False

    try:
        k = float(k)
        if not is_integer(k) or not 1 < k < n:
            print("Invalid number of clusters!")
            return False

    except:
        print("Invalid number of clusters!")
        return False

    try:
        itr = float(itr)
        if not is_integer(itr) or not 1 < itr < 1000:
            print("Invalid maximum iteration!")
            return False

    except:
        print("Invalid maximum iteration!")
        return False

    return True


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


# Execution ###
if __name__ == "__main__":
    if len(sys.argv) > 4 or len(sys.argv) < 3:
        print("An Error Has Occurred")
        sys.exit(1)
    nums = sys.argv[1]
    if len(sys.argv) == 3:
        iterations = 200
        input_file = sys.argv[2]
    else:
        iterations = sys.argv[2]
        input_file = sys.argv[3]

    main(nums, iterations, input_file)
