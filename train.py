import numpy as np
from dataprocessing import *

def latex_matrix(matrix):
    matrix_str = '\\begin{bmatrix}\n'
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix_str += str("{0:.3f}".format(matrix[i][j])) + ' & '
        matrix_str = matrix_str[:-2] + '\\\\\n'
    matrix_str += '\\end{bmatrix} \n'
    return matrix_str


# Define our arguments and their respective sizes

# state_space is all possible states y (length L)
# obs_space is all possible observations x (length m)
# start_probs are the start probabiliies (length L)
# observs is our sequence of observations (length M)
# transition is matrix of transition probabilities between y_j, y_i (size LxL)
# emission is matrix of prob of observing y from x (size Lxm)
# E is our matrix of P(y_i)'s. E[i, j] = P(y_j = i), size LxM

def MStep(state_space, obs_space, observs, E):
    L = len(state_space)
    m = len(obs_space)

    transition = np.zeros((L, L))
    emission   = np.zeros((L, m))

    # Calculate transition and emission matrix
    for i in range(L):
        # Normalization factor for both matrices
        norm = np.sum(E[i, :])

        for j in range(M-1):
            transition[i, :] += E[i, j] * E[:, j+1].T
        transition[i, :] /= norm

        for j in range(M):
            val = observs[j] # jth emission in sequence
            emission[i, val] += E[i, j]
        emission[i, :] /= norm

    return transition, emission


def EStep(state_space, obs_space, observs, transition, emission):
    M = len(observs)
    L = len(state_space)
    m = len(obs_space)

    # Run fwd-bckwd with uniform start probabilities
    uniform     = [1./L  for i in range(L)]
    fwd_probs   = forward(state_space, uniform, observs, transition, emission)
    bckwd_probs = backward(state_space, observs, transition, emission)

    # Calculate P(y_i) for each y = (y1, ..., yM)
    E = np.zeros([L, M])

    for i in range(M):
        dotprod = np.dot(fwd_probs[:, i].T, bckwd_probs[:, i])
        E[:, i] = fwd_probs[:, i] * bckwd_probs[:, i] / dotprod
    
    return E


# Performs the forward algorithm, returning matrix of probabilities
def forward(state_space, start_probs, observs, transition, emission):
    M = len(observs)
    fwd_probs = np.array(start_probs).T

    # Do the iterative forward algorithm
    for i in range(M):
        fwd_next = np.dot(transition, np.diag(emission[:, observs[i]]))
        fwd_next = np.dot(fwd_next, fwd_probs[:, -1])

        fwd_probs = np.append(fwd_probs, fwd_next, 1)

    return fwd_probs


# Performs the backward algorithm, returning matrix of probabilities
def backward(state_space, observs, transition, emission):
    M = len(observs)
    bwd_probs = np.array([1] * L).T

    # i is reversed from the forward algorithm
    for i in reversed(range(M)):
        bwd_prev = np.dot(transition, np.diag(emission[:, observs[i]]))
        bwd_prev = np.dot(bwd_prev, bwd_probs[:,0])

        # append in the opposite order so prev column is always in front
        bwd_probs = np.append(bwd_prev, bwd_probs, 1)
        
    return bwd_probs

# eps is our stopping condition
# observs is all of our training examples
def EM_algorithm(state_space, obs_space, transition, emission, observs, eps, epoch_size):
    norm_diff = eps + 1

    while norm_diff > eps:
        transition_new = np.copy(transition)
        emission_new   = np.copy(emission)
        for i in range(epoch_size):
            E = EStep(state_space, obs_space, observs, transition_new, emission_new)
            transition_new, emission_new = MStep(state_space, obs_space, observs, E)

        norm_diff  = np.linalg.norm(transition - transition_new) + \
                     np.linalg.norm(emission - emission_new)
        transition = transition_new
        emission   = emission_new

    return transition, emission
