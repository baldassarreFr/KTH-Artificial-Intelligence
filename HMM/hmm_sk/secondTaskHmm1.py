#HMM1: Probability of Emission Sequence
# Computes P(O_1:T | A, B, pi) using the forward (alpha-pass) algorithm.

import sys  # we'll read everything from standard input

def read_matrix_line():
    # grab a line, split into pieces
    tokens = sys.stdin.readline().strip().split()
    # first two values are row and column counts
    r = int(tokens[0]); c = int(tokens[1])
    # the rest are the matrix entries, row-major order
    flat = list(map(float, tokens[2:]))
    # reshape into a list-of-rows matrix
    mat = [flat[i * c:(i + 1) * c] for i in range(r)]
    return mat, r, c

def read_observation_line():
    # read the line with T followed by T observation indices
    tokens = sys.stdin.readline().strip().split()
    # first number is how many observations there are
    T = int(float(tokens[0]))  
    # the rest are the observation symbols (as integers 0..M-1)
    obs = [int(float(x)) for x in tokens[1:]]
    return T, obs

def fmt(x):
    # print a float with enough precision but without trailing zeros
    s = f"{x:.12f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return '0' if s == '-0' else s

def main():
    # read A (transition), B (emission), and pi (initial distribution) — in this order
    A, rA, cA = read_matrix_line()   # A is N×N
    B, rB, cB = read_matrix_line()   # B is N×M
    PI, rP, cP = read_matrix_line()  # pi is 1×N single row
    # read number of observations and the observation sequence
    T, obs = read_observation_line()

    # the number of states N is rA (and equals cA, rB, and cP)
    N = rA

    # pull pi as a flat list (single row)
    pi = PI[0]

# ------------ Forward algorithm  ------------------------------

    # t = 0 (first observation): alpha_1(i) = pi[i] * B[i][o_1]
    o0 = obs[0]
    alpha = [pi[i] * B[i][o0] for i in range(N)]

    # t = 1..T-1: alpha_t(i) = B[i][o_t] * sum_j alpha_{t-1}(j) * A[j][i]
    for t in range(1, T):
        ot = obs[t]
        new_alpha = [0.0] * N
        # compute each state's alpha via the inner sum over previous states
        for i in range(N):
            s = 0.0
            for j in range(N):
                s += alpha[j] * A[j][i]
            new_alpha[i] = s * B[i][ot]
        alpha = new_alpha  # move to the next time step

    # total probability is the sum over final alphas
    prob = sum(alpha)
    # output the result
    print(fmt(prob))

if __name__ == "__main__":
    main()
