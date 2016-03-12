import numpy as np
from dataprocessing import *
import random as rand

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
# emission is matrix of prob of observing x from y (size mxL)
# E is our matrix of P(y_i)'s. E[i, j] = P(y_j = i), size LxM

def MStep(state_space, obs_space, observs, E, F):
    M = len(observs)
    L = len(state_space)
    m = len(obs_space)
    transition = np.zeros((L, L))
    emission   = np.zeros((m, L))

    for b in range(L):
        for a in range(L):
            transition[b, a] = sum([F[i, b, a] for i in range(0, M-1)])

    # Both simultaneously
    for i in range(L):
        for j in range(M):
            val = observs[j] # jth emission in sequence
            emission[val, i] += E[i, j]

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

    # Calculate P(y_i-1=a, y_i = b) for each y = (y1, ..., yM)
    F = np.zeros([M, L, L])

    for i in range(0,M-1):
        for b in range(L):
            for a in range(L):
                F[i][b][a] = fwd_probs[a, i] * transition[b,a] * emission[observs[i+1], b] * bckwd_probs[b, i+1]
        F[i] /= F[i].sum()
        # print F[i]
    return E, F


# Performs the forward algorithm, returning matrix of probabilities
def forward(state_space, start_probs, observs, transition, emission):
    M = len(observs)
    #fwd_probs = np.array(start_probs).T
    fwd_probs = np.array(start_probs).reshape((1, len(start_probs))).T
    # Do the iterative forward algorithm
    for i in range(M):
        fwd_next = np.dot(transition, np.diag(emission[observs[i], :]))
        fwd_next = np.dot(fwd_next, fwd_probs[:, -1]).reshape((len(start_probs), 1))
        fwd_probs = np.append(fwd_probs, fwd_next/float(sum(fwd_next)), 1)
    #print fwd_probs
    return fwd_probs


# Performs the backward algorithm, returning matrix of probabilities
def backward(state_space, observs, transition, emission):
    M = len(observs)
    L = len(state_space)
    bwd_probs = np.array([1] * L).reshape((1, L)).T

    # i is reversed from the forward algorithm
    for i in reversed(range(M)):
        bwd_prev = np.dot(transition, np.diag(emission[observs[i], :]))
        bwd_prev = np.dot(bwd_prev, bwd_probs[:,0]).reshape((L, 1))

        # append in the opposite order so prev column is always in front
        bwd_probs = np.append(bwd_prev/float(sum(bwd_prev)), bwd_probs, 1)
    #print bwd_probs
    return bwd_probs

# eps is our stopping condition
# observs is all of our training examples
def EM_algorithm(state_space, obs_space, transition, emission, observs, eps, epoch_size):
    L = len(state_space)
    M = len(observs)
    norm_diff = eps + 1

    while norm_diff > eps:

        transition_new = np.zeros(transition.shape)
        emission_new   = np.zeros(emission.shape)
        for i in range(epoch_size):
            print 'epoch', i

            norm = np.zeros(L)

            # Make a pass thru the data
            for observ in observs:
                E, F = EStep(state_space, obs_space, observ, transition, emission)

                transition_epoch, emission_epoch = MStep(state_space, obs_space, observ, E, F)
                # if np.max(transition_epoch) > 1.5:
                #     print transition_epoch
                emission_new += emission_epoch
                transition_new += transition_epoch
                
                norm += E.sum(axis=1)
            # Normalize
            emission_new /= norm
            transition_new /= norm
                
        norm_diff  = np.linalg.norm(transition - transition_new) + \
                     np.linalg.norm(emission - emission_new)
        print 'transition------\n', transition_new
        print transition_new.sum(axis=0), transition_new.sum()
        print 'emission--------\n', emission_new
        # print emission_new.sum(axis=0), emission_new.sum()
        print '----------- \n', norm_diff
        transition = np.copy(transition_new)
        emission   = np.copy(emission_new)

    return transition, emission

if __name__ == '__main__':
    EM_in = outputStream()
    flat_obs = [item for sublist in EM_in for item in sublist] 
    unique_obs = len(set(flat_obs))
    num_internal = 5
    Trans = np.random.rand(num_internal, num_internal)
    Emiss = np.random.rand(unique_obs, num_internal)
    
    for i in range(Trans.shape[1]):
        Trans[:, i] /= np.sum(Trans[:, i])
    for i in range(Emiss.shape[1]):
        Emiss[:, i] /= np.sum(Emiss[:, i])

    final_out = EM_algorithm(np.array(range(num_internal)), \
                             np.array(list(set(flat_obs))), Trans, Emiss, EM_in, .005, 1)

