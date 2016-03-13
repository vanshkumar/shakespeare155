import numpy as np
from dataprocessing import *

# Define our arguments and their respective sizes

# state_space is all possible states y (length L)
# obs_space is all possible observations x (length m)
# start_probs are the start probabiliies (length L)
# observs is our sequence of observations (length M)
# transition is matrix of transition probabilities between y_j, y_i (size LxL)
# emission is matrix of prob of observing x from y (size mxL)
# E is our matrix of P(y_i)'s. E[i, j] = P(y_j = i), size LxM
# F is our matrix of P(y_i = b, y_i-1 = a), size MxLxL

def MStep(state_space, obs_space, observs, E, F):
    M = len(observs)
    L = len(state_space)
    m = len(obs_space)
    transition = np.zeros((L, L))
    emission   = np.zeros((m, L))

    # Compute the numerators of the transition matrix for a single example
    # See Lecture 8 slide 68
    transition = F.sum(axis=0)

    # Compute the numerators of the emission matrix for a single example
    # See Lecture 8 slide 68
    for i in range(L):
        for j in range(M):
            # Add into the emission matrix for observation j
            emission[observs[j], i] += E[i, j]

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


# Performs the forward algorithm, returned matrix of probs is size M+1xL
def forward(state_space, start_probs, observs, transition, emission):
    M = len(observs)
    L = len(state_space)

    fwd_probs = np.array(start_probs).reshape((L, 1))
    for i in range(M):
        fwd_next = np.dot(transition.T, np.diag(emission[observs[i], :]))
        fwd_next = np.dot(fwd_next, fwd_probs[:, -1]).reshape((L, 1))

        fwd_probs = np.append(fwd_probs, fwd_next/float(sum(fwd_next)), 1)

    return fwd_probs


# Performs the backward algorithm, returned matrix of probs is size M+1xL
def backward(state_space, observs, transition, emission):
    M = len(observs)
    L = len(state_space)
    bwd_probs = np.ones((L, 1)) / float(L)

    # i is reversed from the forward algorithm
    for i in reversed(range(M)):
        bwd_prev = np.dot(transition.T, np.diag(emission[observs[i], :]))
        bwd_prev = np.dot(bwd_prev, bwd_probs[:,0]).reshape((L, 1))

        # Append in the opposite order so prev column is always in front
        bwd_probs = np.append(bwd_prev/float(sum(bwd_prev)), bwd_probs, 1)

    return bwd_probs


# eps is our stopping condition
# observs is all of our training examples
def EM_algorithm(state_space, obs_space, transition, emission, observs, eps):
    L = len(state_space)
    M = len(observs)
    norm_diff = eps + 1

    while norm_diff > eps:
        transition_new = np.zeros(transition.shape)
        emission_new   = np.zeros(emission.shape)

        # The denominators of our calculation of new matrices
        denoms = np.zeros(L)

        # Make a pass thru the data
        for observ in observs:
            # Perform an EM step to update matrices and save numerators & denoms
            E, F = EStep(state_space, obs_space, observ, transition, emission)
            transition_epoch, emission_epoch = MStep(state_space, obs_space, observ, E, F)

            emission_new += emission_epoch
            transition_new += transition_epoch
            
            denoms += E.sum(axis=1)

        # Do the actual division to get updated matrices
        emission_new /= denoms
        transition_new /= denoms
                
        norm_diff  = np.linalg.norm(transition - transition_new) + \
                     np.linalg.norm(emission - emission_new)
        print "normdiff: ", norm_diff

        transition = np.copy(transition_new)
        emission   = np.copy(emission_new)

    return transition, emission


# Compute the transition and emission matrices with num_internal hidden states
def computeMatrices(num_internal):
    EM_in, worddict, rhymes = outputStream()
    iddict = {y:x for x,y in worddict.iteritems()}

    # We will train our HMM to generate lines from right to left
    # (this makes rhyming significantly easier as we just seed state 1)
    EM_in = [sample[::-1] for sample in EM_in]

    flat_obs = [item for sublist in EM_in for item in sublist]
    unique_obs = len(set(flat_obs))

    # Initialize transition & emission to random normalized matrices
    transition = np.random.rand(num_internal, num_internal)
    emission = np.random.rand(unique_obs, num_internal)
    transition /= Trans.sum(axis=0)
    emission /= Emiss.sum(axis=0)

    final_t, final_e = EM_algorithm(range(num_internal), list(set(flat_obs)),\
                                    transition, emission, EM_in, .001)

    # Save our calculate matrices so we don't have to recalculate every time
    tFile = open(os.getcwd() + "/data/trans" + str(num_internal) + ".npy", "w")
    eFile = open(os.getcwd() + "/data/emiss" + str(num_internal) +".npy", "w")
    dictFile = open(os.getcwd() + "/data/iddict.npy", "w+")

    np.save(tFile, final_t)
    np.save(eFile, final_e)
    np.save(dictFile, iddict)

    tFile.close()
    eFile.close()
    dictFile.close()


# Perform a random walk thru our trained Markov model, with an optional
# first_obs. If first_obs is not None, it is guaranteed the first element
# of our sequence of observations we return
def predictSequence(transition, emission, length, first_obs=None):
    sequence = np.zeros(length)
    states = []

    # We can take 1 sample from a multinomial distribution to pick a random
    # state and associated observation (sampling will take into account the
    # transition & emission probabilities proportionally)
    state = list(np.random.multinomial(1, transition[:, 0])).index(1)
    sequence[0] = list(np.random.multinomial(1, emission[:, state])).index(1)

    if first_obs != None:
        sequence[0] = first_obs
        state = list(np.random.multinomial(1, emission[first_obs, :])).index(1)

    states.append(state)

    for j in range(1, length):
        state = list(np.random.multinomial(1, transition[:, state])).index(1)
        sequence[j] = list(np.random.multinomial(1, emission[:, state])).index(1)
        states.append(state)

    # Reverse the sequence and states because our HMM is trained to generate
    # lines from right to left
    return sequence[::-1], states[::-1]


# Count syllables where poem_line is a line of a poem - a string with words
# separated by spaces and no punctuation
def count_syllables(poem_line):
    return sum([nsyl(x) for x in poem_line.split(' ')[:-1]])


# Return the results of a single random walk thru our Markov model
def philosophize(iddict, trans, emiss, length, first_obs=None):
    prediction, states = predictSequence(trans, emiss, length, first_obs)
    poem = ""
    for i in prediction:
        poem += iddict[int(i)] + " "
    return poem, states


# Return the results of a single random walk thru our Markov model, with the
# constraint that the number of syllables must be exactly 'syllables'. If
# first_obs is not None, then we also guarantee that the last word of the
# returned poem is the word associated with first_obs
def philosophize_syls(iddict, trans, emiss, length, syllables, first_obs=None):
    poem, states = philosophize(iddict, trans, emiss, length, first_obs)
    syls = count_syllables(poem)
    while syls != syllables:
        poem, states = philosophize(iddict, T, E, length, first_obs)
        syls = count_syllables(poem)
    return poem, states


# Generate a sonnet from trained Markov model
# samples is a list of lists, where each list is a training example (a single
# line from a sonnet)
def generate_sonnet(iddict, trans, emiss, samples):
    # 14 lines with 10 syllables each
    poem = [''] * 14
    for line in range(14):
        length = len(np.random.choice(samples))
        poem_line, states = philosophize_syls(iddict, trans, emiss, length, 10)
        poem[line] = poem_line

    print ",\n".join(poem) + "."


# Generate a rhyming sonnet from trained Markov model
# samples is a list of lists as in generate_sonnet
# rhymes is a dictionary where rhymes[i] = j means that the words associated
# i and j rhyme
def generate_rhyming_sonnet(iddict, trans, emiss, samples, rhymes):
    poem = [''] * 14
    # 14 lines with rhyme scheme abab cdcd efef gg and 10 syllables each line
    for line in [0, 1, 4, 5, 8, 9, 12]:
        # Pick a rhyming pair to seed end of our lines with
        seed1 = np.random.choice(rhymes.keys())
        seed2 = rhymes[seed1]

        poem_line, states = philosophize_syls(iddict, trans, emiss, \
            len(np.random.choice(samples)), 10, seed1)
        poem_line2, states2 = philosophize_syls(iddict, trans, emiss, \
            len(np.random.choice(samples)), 10, seed2)

        poem[line] = poem_line

        if line == 12:
            # Last couplet, so rhyme scheme is different
            poem[line+1] = poem_line2
        else:
            poem[line+2] = poem_line2

    print ",\n".join(poem) + "."


# Checks if a given poem (string with words separated by whitespace) which must
# have 17 syllables can be a haiku or not - if it can be split into the 5 7 5
# structure while preserving words
def haiku_split(poem):
    words = poem.split()
    lines = []

    # Check if we can get exactly 5 syllables for the first line of our haiku
    syl_count, i, line = 0, 0, []
    while True:
        if syl_count + nsyl(words[i]) > 5:
            break
        line.append(words[i])
        syl_count += nsyl(words[i])
        i += 1

    # If not return false
    if syl_count != 5:
        return False

    lines.append(line)

    # Check if we can get exactly 7 syllables for second line of our haiku
    syl_count, line = 0, []
    while True:
        if syl_count + nsyl(words[i]) > 7:
            break
        line.append(words[i])
        syl_count += nsyl(words[i])
        i += 1

    # If not return false
    if syl_count != 7:
        return False

    lines.append(line)
    lines.append(words[i:])

    return lines


# Generate a haiku from trained Markov model, with the 5 7 5 syllable scheme
def generate_haiku(iddict, trans, emiss):
    while True:
        length = np.random.randint(12, 15) # Pick reasonable length for haiku
        
        # 17 total syllables
        poem, states = philosophize_syls(iddict, trans, emiss, length, 17)
        
        # If it's a valid haiku, we're done; otherwise, keep going
        haiku = haiku_split(poem)
        if haiku:
            for line in haiku:
                print ' '.join(line)
            return


if __name__ == '__main__':
    # 50 internal states for this Markov model
    num_internal = 50

    # Compute and save transition/emission matrices
    # computeMatrices(num_internal)
    
    # Load in computed transition matrices
    iddict = np.load(os.getcwd() + "/data/iddict.npy").item()
    T = np.load(os.getcwd() + "/data/trans" + str(num_internal) + ".npy")
    E = np.load(os.getcwd() + "/data/emiss" + str(num_internal) + ".npy")

    # Get training examples (lines of sonnets) and rhyming dictionary
    EM_in, worddict, rhymes = outputStream()

    print "\nHaiku: "
    generate_haiku(iddict, T, E)

    print "\nSonnet:"
    generate_sonnet(iddict, T, E, EM_in)

    print "\nRhyming sonnet:"
    generate_rhyming_sonnet(iddict, T, E, EM_in, rhymes)

    length = 7
    print "\n" + str(length) + " words sampled from model: "
    poem, states = philosophize(iddict, T, E, length)
    print poem

    # Do some visualization
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