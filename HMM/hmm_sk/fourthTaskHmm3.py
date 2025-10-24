# Trains HMM (A, B) from a single observation sequence using Baum–Welch 
import sys      
import math     

# I/O helpers
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

def fmt(x):
    s = f"{x:.6f}" # 6 decimals is neat/accepted
    if "." in s: s = s.rstrip("0").rstrip(".") # trim trailing zeros/point
    return s if s != "-0" else "0" # normalize negative zero

# forward (alpha) pass
def alpha_pass(A, B, pi, obs):
    N = len(A)  # number of states
    T = len(obs) # sequence length
    alpha = [[0.0]*N for _ in range(T)] # T x N table
    c = [0.0]*T # scaling factors per time

    # t = 0: alpha[0][i] = pi[i] * B[i][o0]
    o0 = obs[0] # first observed symbol
    s = 0.0 # sum for scaling
    for i in range(N):# loop states
        alpha[0][i] = pi[i] * B[i][o0] # raw alpha
        s += alpha[0][i]  # accumulate for scaling
    c[0] = 1.0/s if s > 0 else 1.0  # c0 = 1/s, fallback 1 if s==0
    for i in range(N): # scale row 0 so it sums to 1
        alpha[0][i] *= c[0]
    # t >= 1: alpha[t][i] = B[i][ot] * sum_j alpha[t-1][j] * A[j][i], then scale
    for t in range(1, T):# each time step
        ot = obs[t]# current observed symbol
        s = 0.0 # reset row sum
        for i in range(N): # compute each state's alpha
            sm = 0.0# inner sum over j
            for j in range(N): # try all predecessors
                sm += alpha[t-1][j] * A[j][i] # alpha_{t-1}(j) * a_{j,i}
            alpha[t][i] = sm * B[i][ot]# times emission prob
            s += alpha[t][i]   # accumulate for scaling
        c[t] = 1.0/s if s > 0 else 1.0  # scaling factor for row t
        for i in range(N): # apply scaling
            alpha[t][i] *= c[t]
    return alpha, c # return scaled alphas + scales

# backward (beta) pass using the same scales
def beta_pass(A, B, obs, c):
    N = len(A) # number of states
    T = len(obs) # sequence length
    beta = [[0.0]*N for _ in range(T)] # T x N table

    # t = T-1: beta[T-1][i] = c[T-1]  (so alpha[T-1]*beta[T-1] ~ posterior up to norm)
    for i in range(N): # last time step
        beta[T-1][i] = c[T-1] # scaled terminal beta

    # t down to 0: beta[t][i] = c[t] * sum_j A[i][j] * B[j][o_{t+1}] * beta[t+1][j]
    for t in range(T-2, -1, -1): # go backwards
        ot1 = obs[t+1] # next observation
        for i in range(N): # each current state i
            s = 0.0 # sum over next states j
            for j in range(N): # loop all j
                s += A[i][j] * B[j][ot1] * beta[t+1][j] # a_{i,j} * b_j(ot1) * beta_{t+1}(j)
            beta[t][i] = c[t] * s # scale to match alpha's scheme
    return beta # return betas

# gamma / digamma computation
def compute_gammas(A, B, obs, alpha, beta):
    N = len(A) # states
    T = len(obs) # length
    gamma = [[0.0]*N for _ in range(T)]# gamma[t][i]
    digamma = [[[0.0]*N for _ in range(N)] for _ in range(T-1)] # digamma[t][i][j]

    # for t = 0..T-2: digamma[t][i][j] proportional to alpha[t][i]*a[i][j]*b[j][o_{t+1}]*beta[t+1][j]
    for t in range(T-1): # up to T-2
        ot1 = obs[t+1] # next symbol
        row_sum = 0.0 # denominator, will also sum gamma
        for i in range(N): # current state i
            for j in range(N): # next state j
                val = alpha[t][i] * A[i][j] * B[j][ot1] * beta[t+1][j]# unnormalized xi
                digamma[t][i][j] = val # store
                row_sum += val # accumulate
        inv = 1.0/row_sum if row_sum > 0 else 0.0 # guard against zero
        for i in range(N): # normalize to a prob. distribution
            s = 0.0 # gamma is the row-sum over j
            for j in range(N): # sum over next states
                digamma[t][i][j] *= inv # normalize xi
                s += digamma[t][i][j] # accumulate gamma
            gamma[t][i] = s # gamma[t][i]
    # special case t = T-1: gamma[T-1] equals alpha[T-1], already normalized
    for i in range(N): # last time step
        gamma[T-1][i] = alpha[T-1][i] # posterior over last state
    return gamma, digamma # posteriors

# re-estimation of parameters
def reestimate(A, B, pi, obs, gamma, digamma):
    N = len(A) # states
    M = len(B[0]) # symbols
    T = len(obs) # length

    # pi: gamma at t=0
    new_pi = [gamma[0][i] for i in range(N)] # pi_i = gamma_0(i)

    # A[i][j] = sum_{t=0..T-2} digamma[t][i][j] / sum_{t=0..T-2} gamma[t][i]
    new_A = [[0.0]*N for _ in range(N)] # allocate
    for i in range(N): # for each row i
        denom = 0.0 # sum gamma over t
        for t in range(T-1): # up to T-2
            denom += gamma[t][i] # accumulate
        if denom <= 0: # degenerate: fall back to uniform
            for j in range(N): new_A[i][j] = 1.0/N # uniform row
        else:
            for j in range(N): # compute numerator per j
                numer = 0.0
                for t in range(T-1): # sum digamma over t
                    numer += digamma[t][i][j]
                new_A[i][j] = numer / denom# normalize row

    # B[i][k] = sum_{t: obs[t]==k} gamma[t][i] / sum_{t=0..T-1} gamma[t][i]
    new_B = [[0.0]*M for _ in range(N)] # allocate
    for i in range(N): # for each state i
        denom = 0.0 # sum gamma over all t
        for t in range(T): # all time steps
            denom += gamma[t][i] # accumulate
        if denom <= 0: # degenerate: uniform emissions
            for k in range(M): new_B[i][k] = 1.0/M # uniform row
        else:
            # count-weight gamma by which symbol appears at t
            counts = [0.0]*M  # tmp per symbol
            for t in range(T): # each time step
                k = obs[t] # emitted symbol index
                counts[k] += gamma[t][i]  # add posterior weight for that symbol
            for k in range(M): # finalize row normalization
                new_B[i][k] = counts[k] / denom # proper distribution

    return new_A, new_B, new_pi # return updated model

# log-likelihood from scales
def log_likelihood(c):
    ll = 0.0 # start at 0
    for ct in c: # for each scale
        ll -= math.log(ct) # sum -log(c_t)
    return ll # equals log P(O | lambda)

# main training loop
def train(A, B, pi, obs, max_iters=100, tol=1e-4):
    # initialize bookkeeping
    old_ll = float("-inf") # previous log-likelihood
    iters = 0 # iteration counter

    while iters < max_iters: # loop until cap
        # compute alphas, betas, then posteriors
        alpha, c = alpha_pass(A, B, pi, obs) # forward with scaling
        beta = beta_pass(A, B, obs, c) # backward with same scales
        gamma, digamma = compute_gammas(A, B, obs, alpha, beta) # posteriors

        #re-estimate parameters
        A, B, pi = reestimate(A, B, pi, obs, gamma, digamma) # update model

        # evaluate current log-likelihood for convergence
        ll = log_likelihood(c) # log P(O | lambda)
        iters += 1 # bump iter count

        # stop if improvement is tiny or NaN
        if not math.isfinite(ll) or (ll - old_ll) < tol: # convergence or bad numerics
            break # exit loop
        old_ll = ll # accept new value and continue

    return A, B # return learned A and B

def main():
    # read model guess (A, B, pi) and observations
    A, rA, cA = read_matrix_line() # transition (N x N)
    B, rB, cB = read_matrix_line() # emission (N x M)
    PI, rP, cP = read_matrix_line() # initial (1 x N)
    T, obs = read_observation_line() # length and sequence

    # flatten pi's single row
    pi = PI[0][:] # copy row to list

    # train with Baum-Welch
    A_learn, B_learn = train(A, B, pi, obs) # run EM

    # print A then B
    flatA = [fmt(x) for row in A_learn for x in row] # row-major flatten
    print(len(A_learn), len(A_learn[0]), " ".join(flatA)) # A line

    flatB = [fmt(x) for row in B_learn for x in row] # row-major flatten
    print(len(B_learn), len(B_learn[0]), " ".join(flatB)) # B line

if __name__ == "__main__":
    main()
