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
            transition[b, a] = sum([F[i, b, a] for i in range(0, M)])

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

    for i in range(0,M):
        for b in range(L):
            for a in range(L):
                F[i][b][a] = fwd_probs[a, i] * transition[b, a] * emission[observs[i], b] * bckwd_probs[b, i+1]
        F[i] /= F[i].sum()
    return E, F


# Performs the forward algorithm, returning matrix of probabilities
def forward(state_space, start_probs, observs, transition, emission):
    M = len(observs)
    L = len(state_space)

    fwd_probs = np.array(start_probs).reshape((L, 1))
    # Do the iterative forward algorithm
    for i in range(M):
        fwd_next = np.dot(transition.T, np.diag(emission[observs[i], :]))
        fwd_next = np.dot(fwd_next, fwd_probs[:, -1]).reshape((L, 1))

        fwd_probs = np.append(fwd_probs, fwd_next/float(sum(fwd_next)), 1)

    return fwd_probs


# Performs the backward algorithm, returning matrix of probabilities
def backward(state_space, observs, transition, emission):
    M = len(observs)
    L = len(state_space)
    bwd_probs = np.ones((L, 1)) / float(L)

    # i is reversed from the forward algorithm
    for i in reversed(range(M)):
        bwd_prev = np.dot(transition.T, np.diag(emission[observs[i], :]))
        bwd_prev = np.dot(bwd_prev, bwd_probs[:,0]).reshape((L, 1))

        # append in the opposite order so prev column is always in front
        bwd_probs = np.append(bwd_prev/float(sum(bwd_prev)), bwd_probs, 1)

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
                # if np.max(transition_epoch) > 1:
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
        print emission_new.sum(axis=0), emission_new.sum()
        print '----------- \n', norm_diff
        transition = np.copy(transition_new)
        emission   = np.copy(emission_new)

    return transition, emission



def predictSequence(transition, emission, length):
    sequence = np.zeros(length)

    Emiss_total = emission.sum(axis=0)
    Trans_total = transition.sum(axis=0)
    print Trans_total
    rando = np.random.uniform(0, Emiss_total.sum())
    cumulative, i = emission[0][0], 0
    while(rando > cumulative):
        i += 1
        cumulative += emission[i/len(emission[0])][i % len(emission[0])]
    print len(emission[0])
    sequence[0], state = i/len(emission[0]), i % len(emission[0])

    for j in range(1, length):
        rando = np.random.uniform(0, Trans_total[state])
        cumulative, i = transition[0][state], 0
        while(rando > cumulative):
            i += 1
            cumulative += transition[i][state]

        state = i

        rando = np.random.uniform(0, Emiss_total[state])
        cumulative, i = emission[0][state], 0
        while(rando > cumulative):
            i += 1
            cumulative += emission[i][state]

        sequence[j] = i 


    return sequence


if __name__ == '__main__':
    EM_in, worddict = outputStream()
    flat_obs = [item for sublist in EM_in for item in sublist]
    unique_obs = len(set(flat_obs))
    num_internal = 
    Trans = np.random.rand(num_internal, num_internal)
    Emiss = np.random.rand(unique_obs, num_internal)
    
    Trans /= Trans.sum(axis=0)
    Emiss /= Emiss.sum(axis=0)

    final_t, final_e = EM_algorithm(np.array(range(num_internal)), \
                             np.array(list(set(flat_obs))), Trans, Emiss, EM_in, .005, 1)

    prediction = predictSequence(final_t, final_e, 100)

    # print prediction


    iddict = {y:x for x,y in worddict.iteritems()}
 
    poem = ""

    for i in prediction:
        poem += iddict[int(i + 2)] + " "

    print poem




