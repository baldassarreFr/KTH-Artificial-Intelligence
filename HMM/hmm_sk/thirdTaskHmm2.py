# HMM2: estimatşng the most probable hidden-state sequence

import sys 

def read_matrix_line():
    line = sys.stdin.readline().strip().split() # read one matrix line
    r, c = int(float(line[0])), int(float(line[1])) # rows and cols, both 4 and 4.0 accepted
    flat = list(map(float, line[2:])) # r*c numbers in row-major order
    mat = [flat[i * c:(i + 1) * c] for i in range(r)] # reshape to list-of-rows
    return mat, r, c # return matrix and dims

def read_observation_line():
    toks = sys.stdin.readline().strip().split() # read T and the T observation ids
    T = int(float(toks[0])) # number of observations
    obs = [int(float(x)) for x in toks[1:]]# each observation id as int
    return T, obs # return both

def main():
    # parse inputs, then observations
    A, rA, cA = read_matrix_line() # transition matrix A (N x N)
    B, rB, cB = read_matrix_line()# emission matrix B (N x M)
    PI, rP, cP = read_matrix_line() # initial distribution pi (1 x N)
    T, obs = read_observation_line() # number of emissions and the sequence
    N = rA # number of hidden states
    pi = PI[0]# flatten the 1xN row to a list

    # Viterbi initialization at t = 0
    o0 = obs[0]# first observed symbol
    delta = [pi[i] * B[i][o0] for i in range(N)]# best path prob to each state at t=0
    backptr = [[-1] * N for _ in range(T)]# backpointers (T x N), first row not used

    # Viterbi recursion,t = 1..T-1
    for t in range(1, T):# loop over remaining time steps
        ot = obs[t] # observed symbol at time t
        new_delta = [0.0] * N # next time-step best probs
        for i in range(N): # compute best path ending in state i
            best_prob = -1.0 # current max value holder
            best_prev = 0 # argmax predecessor state
            for j in range(N): # try every predecessor j
                cand = delta[j] * A[j][i] # extend best path to j with transition j->i
                if cand > best_prob: # update max and argmax
                    best_prob = cand
                    best_prev = j
            new_delta[i] = best_prob * B[i][ot] # multiply by emission prob for state i
            backptr[t][i] = best_prev # remember which j led to the max
        delta = new_delta  # roll to next time step

    # Terminate, choose the best final state
    last_state = max(range(N), key=lambda i: delta[i]) # argmax over final delta_T

    # Backtrack the state sequence using backpointers
    path = [0] * T # allocate space for the output path
    path[T - 1] = last_state # set final state
    for t in range(T - 2, -1, -1): # walk backwards t = T-2 .. 0
        path[t] = backptr[t + 1][path[t + 1]] # previous state is the stored backpointer

    # Output
    print(" ".join(str(x) for x in path))
if __name__ == "__main__":
    main()
