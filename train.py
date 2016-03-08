import numpy as np
from operator import itemgetter as get


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


def EStep(pos_states, pos_observations, state_seq, A, O):

    E = np.zeros([len(state_seq), len(pos_states)])

    E[0] = [O[i][state_seq[0]] for i in range(len(pos_states))]

    for x in range(1, M):
        for i in range(L):
            E[x][i] = max([A[j][i] * O[i][X[x]]*E[j][0] for j in range(len(pos_states))])

    return np.transpose(E)

