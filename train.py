import numpy as np
from operator import itemgetter as get
from dataprocessing import *

def latex_matrix(matrix):
    matrix_str = '\\begin{bmatrix}\n'
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix_str += str("{0:.3f}".format(matrix[i][j])) + ' & '
        matrix_str = matrix_str[:-2] + '\\\\\n'
    matrix_str += '\\end{bmatrix} \n'
    return matrix_str


# Uses a single Maximization step to compute A (state-transition) and
# O (observation) matrices. See Lecture 6 Slide 65.
def MStep(pos_states, pos_observations, state_seq, obs_seq):

    A = [[1. / len(pos_states) for i in pos_states] for j in pos_states]
    O = [[1. / len(pos_observations) for i in pos_observations]
         for j in pos_states]

    # create transition matrix
    for prev_state in pos_states:
        for state in pos_states:
            num = 0.0
            den = 0.0
            for j in range(len(state_seq) - 1):
                if (state_seq[j] == pos_states[prev_state]):
                    den += 1
                    if state_seq[j + 1] == pos_states[state]:
                        num += 1
            A[pos_states[state]][pos_states[prev_state]] = num / \
                den if den != 0 else 0

    # create observation matrix
    for state in pos_states:
        for obs in pos_observations:
            num = 0.0
            den = 0.0
            num = sum([int(obs_seq[j] == pos_observations[obs]) and
                       int(state_seq[j] == pos_states[state]) for j in range(len(state_seq))])
            den = sum([int(state_seq[j] == pos_states[state])
                       for j in range(len(state_seq))])

            O[pos_states[state]][pos_observations[obs]] = float(num) / den

    return A, O


def EStep(state_space, observs, transition, emission):
    m = emission.shape[1]
    M = len(observs)
    L = len(state_space)

    # Run fwd-bckwd with uniform start probabilities
    uniform = [1./L  for i in range(L)]
    fwd_probs = forward(state_space, uniform, observs, transition, emission)
    bckwd_probs = backward(state_space, observs, transition, emission)

    # Calculate P(y_i) for each y = (y1, ..., yM)
    E = np.zeros([L, M])

    for i in range(M):
        dotprod = np.dot(fwd_probs[:, i].T, bckwd_probs[:, i])
        E[:, i] = fwd_probs[:, i] * bckwd_probs[:, i] / dotprod
    
    return E


# state_space is all possible states x (length L)
# start_probs are the start probabiliies (length L)
# observs is our sequence of observations (length M)
# transition is matrix of transition probabilities between y_j, y_i (size LxL)
# emission is matrix of prob of observing y from x (size Lxm)

# Performs the forward algorithm, returning matrix of probabilities of
# internal states
def forward(state_space, start_probs, observs, transition, emission):
    M = len(observs)
    fwd_probs = np.array(start_probs).T

    # Do the iterative forward algorithm
    for i in range(M):
        fwd_next = np.dot(transition, np.diag(emission[:, observs[i]]))
        fwd_next = np.dot(fwd_next, fwd_probs[:, -1])

        fwd_probs = np.append(fwd_probs, fwd_next, 1)

    return fwd_probs

# Performs the backward algorithm, returning matrix of probabilities of
# internal states
def backward(state_space, observs, transition, emission):
    M = len(observs)
    bwd_probs = np.array([1] * L).T

    # i is reversed from the forward algorithm
    for i in reversed(range(M)):
        bwd_prev = np.dot(transition, np.diag(emission[:, observs[i]]))
        bwd_prev = np.dot(bwd_prev, bwd_probs[:,0])
        
        # append in the opposite order so that the prev column is always in the 
        # front
        bwd_probs = np.append(bwd_prev, bwd_probs, 1)
        
    return bwd_probs


def Forward(num_states, obs, A, O):
    """Computes the probability a given HMM emits a given observation using the
        forward algorithm. This uses a dynamic programming approach, and uses
        the 'prob' matrix to store the probability of the sequence at each length.
        Arguments: num_states the number of states
                   obs        an array of observations
                   A          the transition matrix
                   O          the observation matrix
        Returns the probability of the observed sequence 'obs'
    """
    len_ = len(obs)                   # number of observations
    # stores p(seqence)
    prob = [[[0.] for i in range(num_states)] for i in range(len_)]
    
    # initializes uniform state distribution, factored by the
    # probability of observing the sequence from the state (given by the
    # observation matrix)
    prob[0] = [(1. / num_states) * O[j][obs[0]] for j in range(num_states)]
    # We iterate through all indices in the data
    for length in range(1, len_):   # length + 1 to avoid initial condition
        for state in range(num_states):
            # stores the probability of transitioning to 'state'
            p_trans = 0

            # probabilty of observing data in our given 'state'
            p_obs = O[state][obs[length]]

            # We iterate through all possible previous states, and update
            # p_trans accordingly.
            for prev_state in range(num_states):
                p_trans += prob[length - 1][prev_state] * A[prev_state][state]

            prob[length][state] = p_trans * p_obs  # update probability

        prob[length] = prob[length][:]  # copies by value
    # start backwards code
    prob_back = [[0.] for i in range(len_)]    
    prob_back[-1] = [1 for j in range(num_states)]    
    for length in reversed(range(0, len_ - 1)):
        for state in range(num_states):
            prob_back[length][state] = sum([prob[length + 1][j] * A[j][state] * O[j][obs[length]] for j in range(num_states)])
    # end backwards code
    # start finding the actual probabilities
    P = [[[0.] for i in range(num_states)] for i in range(len_)]
    
    for length in range(len_):
        for state in range(num_states):
            P[length][state] = prob[length][state] * prob_back[length][state] / \
                sum([prob[length][j] * prob_back[length][j] for j in range(num_states)])
    # end computing the probabilities
    # return total probability
    return P