import numpy as np
import argparse
import symnmf  # Import the C extension module


def parse_arguments():
    parser = argparse.ArgumentParser(description="SymNMF Clustering")
    parser.add_argument('k', type=int, help="Number of required clusters")
    parser.add_argument('goal', type=str, choices=['symnmf', 'sym', 'ddg', 'norm'], help="Goal of the operation")
    parser.add_argument('file_name', type=str, help="Path to the input file (.txt)")
    return parser.parse_args()


def load_data(file_name):
    # Load data points from the file, assuming one data point per line
    data = np.loadtxt(file_name, delimiter=',')
    res = data.tolist()
    return res


def initialize_H(W, k):
    np.random.seed(1234)
    m = np.mean(W)
    H_init = np.random.uniform(0, 2 * np.sqrt(m / k), (W.shape[0], k))
    return H_init


def main():
    try:
        args = parse_arguments()
        data = load_data(args.file_name)
        data = np.ascontiguousarray(data, dtype=np.float64).tolist()
        n = len(data)
        d = len(data[0])
        if int(args.k) >= len(data):
            print("An Error Has Occurred!")
            return

        if args.goal == 'sym':
            # Call the C function to compute the similarity matrix
            W = symnmf.sym(data, n, d)
            print_matrix(W)

        elif args.goal == 'ddg':
            # Call the C function to compute the diagonal degree matrix
            D = symnmf.ddg(data, n, d)
            print_matrix(D)

        elif args.goal == 'norm':
            # Call the C function to compute the normalized similarity matrix
            sym_mat = symnmf.sym(data, n, d)
            ddg_mat = symnmf.ddg(data, n, d)
            W_norm = symnmf.norm(sym_mat, ddg_mat, n)
            print_matrix(W_norm)

        elif args.goal == 'symnmf':
            # Initialize the similarity matrix using the C extension
            sym_mat = symnmf.sym(data, n, d)
            ddg_mat = symnmf.ddg(data, n, d)
            W_norm = symnmf.norm(sym_mat, ddg_mat, n)
            # Initialize H
            H_init = initialize_H(W, args.k)
            # Call the C function to perform SymNMF and get the final H
            H_final = symnmf.symnmf(n, args.k, W_norm, data)
            print_matrix(H_final)

        else:
            print("An Error Has Occurred!")

    except Exception as e:
        print(e)
        print("An Error Has Occurred!")


def print_matrix(matrix):
    # Print the matrix with 4 decimal places, each row in a new line
    for row in matrix:
        print(','.join(f'{value:.4f}' for value in row))



if __name__ == "__main__":
    main()
