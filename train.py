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
# emission is matrix of prob of observing x from y (size mxL)
# E is our matrix of P(y_i)'s. E[i, j] = P(y_j = i), size LxM

def MStep(state_space, obs_space, observs, E, F):
    M = len(observs)
    L = len(state_space)
    m = len(obs_space)
    transition = np.zeros((L, L))
    emission   = np.zeros((m, L))

    transition = F.sum(axis=0)

    # Emission
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

    E = np.multiply(fwd_probs[:, :-1], bckwd_probs[:, :-1])
    E /= E.sum(axis=0)

    # Calculate P(y_i-1=a, y_i = b) for each y = (y1, ..., yM)
    F = np.zeros([M, L, L])

    for i in range(M):
        for b in range(L):
            F[i][b] = fwd_probs[:, i] * transition[b, :].T * emission[observs[i], b] * bckwd_probs[b, i+1]
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

        norm = np.zeros(L)

        # Make a pass thru the data
        for observ in observs:
            E, F = EStep(state_space, obs_space, observ, transition, emission)

            transition_epoch, emission_epoch = MStep(state_space, obs_space, observ, E, F)

            emission_new += emission_epoch
            transition_new += transition_epoch
            
            norm += E.sum(axis=1)
        # Normalize
        emission_new /= norm
        transition_new /= norm
                
        norm_diff  = np.linalg.norm(transition - transition_new) + \
                     np.linalg.norm(emission - emission_new)
        print "normdiff: ", norm_diff

        # print "Transition norm:"
        # print transition_new.sum(axis=0)
        # print "Emission norm:"
        # print emission_new.sum(axis=0)

        transition = np.copy(transition_new)
        emission   = np.copy(emission_new)

    return transition, emission



def predictSequence(transition, emission, length, start_seed=None):
    sequence = np.zeros(length)
    states = []

    state = list(np.random.multinomial(1, transition[:, 0])).index(1)
    sequence[0] = list(np.random.multinomial(1, emission[:, state])).index(1)

    if start_seed != None:
        sequence[0] = start_seed
        sample = list(np.random.multinomial(1, emission[start_seed, :]))
        state = sample.index(1)

    states.append(state)

    for j in range(1, length):
        state = list(np.random.multinomial(1, transition[:, state])).index(1)
        sequence[j] = list(np.random.multinomial(1, emission[:, state])).index(1)
        states.append(state)

    if start_seed != None:
        sequence = sequence[::-1]
        states = states[::-1]
    return sequence, states


def computeMatrices(num_internal, backwards=False):
    EM_in, worddict, rhymes = outputStream()
    iddict = {y:x for x,y in worddict.iteritems()}

    if backwards:
        EM_in = [sample[::-1] for sample in EM_in]

    flat_obs = [item for sublist in EM_in for item in sublist]
    unique_obs = len(set(flat_obs))
    Trans = np.random.rand(num_internal, num_internal)
    Emiss = np.random.rand(unique_obs, num_internal)
    
    Trans /= Trans.sum(axis=0)
    Emiss /= Emiss.sum(axis=0)

    final_t, final_e = EM_algorithm(np.array(range(num_internal)), \
                             np.array(list(set(flat_obs))), Trans, Emiss, EM_in, .001, 1)

    direction = ''
    if backwards:
        direction = 'rev'


    tFile = open(os.getcwd() + "/data/trans" + str(num_internal) + direction + ".npy", "w")
    eFile = open(os.getcwd() + "/data/emiss" + str(num_internal) + direction +".npy", "w")
    dictFile = open(os.getcwd() + "/data/iddict.npy", "w+")

    np.save(tFile, final_t)
    np.save(eFile, final_e)
    np.save(dictFile, iddict)

    tFile.close()
    eFile.close()
    dictFile.close()


# Count syllables where poem is a string with whitespaces
def count_syllables(poem):
    return sum([nsyl(x) for x in poem.split(' ')[:-1]])


def philosophize(iddict, trans, emiss, length, end_seed=None):
    prediction, states = predictSequence(trans, emiss, length, end_seed)
    poem = ""
    for i in prediction:
        poem += iddict[int(i)] + " "
    return poem, states


def philosophize_syls(iddict, trans, emiss, length, syllables):
    poem, states = philosophize(iddict, trans, emiss, length)
    syls = count_syllables(poem)
    while syls != syllables:
        poem, states = philosophize(iddict, T, E, length)
        syls = count_syllables(poem)
    return poem, states


def philosophize_syls_and_rhyme(iddict, trans, emiss, length, syllables, end_seed):
    poem, states = philosophize(iddict, trans, emiss, length, end_seed)
    syls = count_syllables(poem)
    while syls != syllables:
        poem, states = philosophize(iddict, T, E, length, end_seed)
        syls = count_syllables(poem)
    return poem, states


def generate_sonnet(iddict, trans, emiss, samples):
    # 14 lines with 10 syllables each
    poem = ['']*14
    for line in range(14):
        length = len(np.random.choice(samples))
        poem_line, states = philosophize_syls(iddict, trans, emiss, length, 10)
        # print poem_line
        poem[line] = poem_line

    print ",\n".join(poem) + "."


def generate_rhyming_sonnet(iddict, trans, emiss, samples, rhymes):
    poem = [''] * 14
    # 14 lines with abab cdcd efef gg
    for line in [0, 1, 4, 5, 8, 9, 12]:
        seed1 = np.random.choice(rhymes.keys())
        seed2 = rhymes[seed1]

        poem_line, states = philosophize_syls_and_rhyme(iddict, trans, emiss, \
            len(np.random.choice(samples)), 10, seed1)
        poem_line2, states2 = philosophize_syls_and_rhyme(iddict, trans, emiss, \
            len(np.random.choice(samples)), 10, seed2)

        poem[line] = poem_line
        if line == 12:
            poem[line+1] = poem_line2
        else:
            poem[line+2] = poem_line2

    print ",\n".join(poem) + "."


def haiku_split(poem):
    words = poem.split()
    lines = []

    syl_count = 0
    tmp = []
    i = 0
    while True:
        if syl_count + nsyl(words[i]) > 5:
            break
        tmp.append(words[i])
        syl_count += nsyl(words[i])
        i += 1

    if syl_count != 5:
        return False, lines

    lines.append(tmp)

    syl_count = 0
    tmp = []
    while True:
        if syl_count + nsyl(words[i]) > 7:
            break
        tmp.append(words[i])
        syl_count += nsyl(words[i])
        i += 1

    if syl_count != 7:
        return False, lines

    lines.append(tmp)
    lines.append(words[i:])

    return True, lines


def generate_haiku(iddict, trans, emiss):
    # 5 7 5 syllable scheme, 17 total
    while True:
        length = np.random.randint(12, 15)
        poem, states = philosophize_syls(iddict, trans, emiss, length, 17)
        haiku_correct, lines = haiku_split(poem)
        if haiku_correct:
            for line in lines:
                print ' '.join(line)
            break


if __name__ == '__main__':
    
    num_internal = 25
    length = 100

    # computeMatrices(num_internal=num_internal, backwards=False)
    # computeMatrices(num_internal=num_internal, backwards=True)
    
    iddict = np.load(os.getcwd() + "/data/iddict.npy").item()
    T = np.load(os.getcwd() + "/data/trans" + str(num_internal) + ".npy")
    E = np.load(os.getcwd() + "/data/emiss" + str(num_internal) + ".npy")

    T_rev = np.load(os.getcwd() + "/data/trans" + str(num_internal) + "rev.npy")
    E_rev = np.load(os.getcwd() + "/data/emiss" + str(num_internal) + "rev.npy")

    EM_in, worddict, rhymes = outputStream()

    print "Haiku: "
    generate_haiku(iddict, T, E)

    print "\nSonnet:"
    generate_sonnet(iddict, T, E, EM_in)

    print "\nRhyming sonnet:"
    generate_rhyming_sonnet(iddict, T_rev, E_rev, EM_in, rhymes)

    print "\n" + str(length) + " words: "
    poem, states = philosophize(iddict, T, E, length)
    print poem

    # trollpoem = True
    # p = 0
    # if trollpoem:
    #     while p != 10:
    #         poem, states = philosophize(iddict, T, E, 7)
    #         p = sum([nsyl(x) for x in poem.split(' ')[:-1]])

    # print poem

    visualize = False
    if visualize:
        num_states = 7
        EM_in, worddict = outputStream()
        iddict = {y:x for x,y in worddict.iteritems()}
        topdict, syls, pos = {}, {}, {}
        flat_obs = [item for sublist in EM_in for item in sublist]
        countdict = {}
        for word in list(set(flat_obs)):
            countdict[word] = flat_obs.count(word)
        speechdict = partsofSpeech(worddict)    
        E = np.load(os.getcwd() + "/data/emiss" + str(num_states) + ".npy")
        E = E.transpose()
        for elem in range(len(E[0])):
            for state in range(num_states):
                # divide the E matrix by the frequency of the words
                a = 1
                E[state][elem] /= countdict[elem]
        for state in range(num_states):
            # returns an array sorted by the size of the argument.  Min is first
            topwords = E[state].argsort()[-10:]
            topwords1 = E[state].argsort()[-100:]
            print topwords
            topdict[state] = [iddict[x] for x in topwords] 
            syls[state] = [nsyl(iddict[x]) for x in topwords1]
            pos[state] = [speechdict[iddict[x]] for x in topwords1]
            print 'average syllables for top 100 words', \
                  sum(syls[state]) / float(np.count_nonzero(syls[state]))
            print 'nouns', pos[state].count('NOUN')
            print 'verbs', pos[state].count('VERB')
            print 'adjectives', pos[state].count('ADJ')
            print 'adverbs', pos[state].count('ADV')