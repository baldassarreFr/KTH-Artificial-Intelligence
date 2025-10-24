# HMM0 – Next Emission Distribution
# Computes the next emission distribution as (pi * A) * B

import sys  # read from stdin, write to stdout

def read_matrix_line():
    # read one line, split on whitespace
    tokens = sys.stdin.readline().strip().split()
    # first two items are row and column counts
    r = int(tokens[0])
    c = int(tokens[1])
    # remaining r*c values are the matrix
    flat = list(map(float, tokens[2:]))
    # reshape into a list of rows
    mat = [flat[i * c:(i + 1) * c] for i in range(r)]
    return mat, r, c

def row_times_matrix(row, mat, out_cols):
    # multiply a 1×N row vector by an N×M matrix get 1×M row vector
    out = [0.0] * out_cols
    # for each output column j, sum row[i] * mat[i][j]
    for j in range(out_cols):
        s = 0.0
        for i, vi in enumerate(row):
            s += vi * mat[i][j]
        out[j] = s
    return out

def fmt(x):
    # print with enough precision but without trailing zeros
    s = f"{x:.10f}"       # start with 10 decimals for safety
    s = s.rstrip('0').rstrip('.') if '.' in s else s
    return '0' if s == '-0' else s  # normalize negative zero

def main():
    # read A (transition), B (emission), pi (initial state distribution)
    A, rA, cA = read_matrix_line()
    B, rB, cB = read_matrix_line()
    PI, rP, cP = read_matrix_line()

    # PI is given as a 1×N matrix, we want the single row as a flat list
    pi_row = PI[0] if rP == 1 else [PI[i][0] for i in range(rP)]

    # predict next state's distribution: pi' = pi * A
    next_state = row_times_matrix(pi_row, A, cA)

    # next emission distribution: e = (pi * A) * B
    emission = row_times_matrix(next_state, B, cB)

    # 1×M row matrix
    print("1", cB, *[fmt(x) for x in emission])

if __name__ == "__main__":
    main()
