import numpy as np
import argparse
import warnings
import symnmf_module # Import the C extension module


def parse_arguments():
    parser = argparse.ArgumentParser(description="SymNMF Clustering")
    parser.add_argument('k', type=int, help="Number of required clusters")
    parser.add_argument('goal', type=str, choices=['symnmf', 'sym', 'ddg', 'norm'], help="Goal of the operation")
    parser.add_argument('file_name', type=str, help="Path to the input file (.txt)")
    return parser.parse_args()


def load_data(file_name):
    # Load data points from the file, assuming one data point per line
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        data = np.loadtxt(file_name, delimiter=',')
        res = data.tolist()
        return res

def initialize_H(W, n, k):
    # Set random seed for reproducibility
    np.random.seed(1234)

    # Calculate the average of all entries in W
    m = np.mean(W)

    # Calculate the upper bound for random values in H
    upper_bound = 2 * np.sqrt(m / k)

    # Randomly initialize H with values in the interval [0, upper_bound]
    H = np.random.uniform(0, upper_bound, size=(n, k))

    return H


def symnmf_construct(data, k, n):
     # Initialize the similarity matrix using the C extension
    W_norm = symnmf_module.norm(data)
    # Initialize H
    H_init = initialize_H(np.array(W_norm), n, k).tolist()
    # Call the C function to perform SymNMF and get the final H
    H_final = symnmf_module.symnmf(n, k, W_norm, H_init)

    return H_final


def main():
    try:
        args = parse_arguments()
        data = load_data(args.file_name)
        data = np.ascontiguousarray(data, dtype=np.float64).tolist()
        n = len(data)
        if int(args.k) >= n:
            print("An Error Has Occurred")
            return

        if args.goal == 'sym':
            # Call the C function to compute the similarity matrix
            sym_mat = symnmf_module.sym(data)
            print_matrix(sym_mat)

        elif args.goal == 'ddg':
            # Call the C function to compute the diagonal degree matrix
            ddg_mat = symnmf_module.ddg(data)
            print_matrix(ddg_mat)

        elif args.goal == 'norm':
            # Call the C function to compute the normalized similarity matrix
            W_norm = symnmf_module.norm(data)
            print_matrix(W_norm)

        elif args.goal == 'symnmf':
            H_final = symnmf_construct(data, args.k, n)
            print_matrix(H_final)

        else:
            print("An Error Has Occurred")

    except Exception as e:
        print("An Error Has Occurred")


def print_matrix(matrix):
    # Print the matrix with 4 decimal places, each row in a new line
    for row in matrix:
        print(','.join(f'{value:.4f}' for value in row))


if __name__ == "__main__":
    main()
