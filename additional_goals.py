import numpy as np
from dataprocessing import *
from sklearn import hmm


if __name__ == '__main__':
    num_internal = 50
    length = 100

    EM_in, worddict = outputStream()
    iddict = {y:x for x,y in worddict.iteritems()}
    iddict[0] = ','
    iddict[1] = '.'

    flat_obs = [item for sublist in EM_in for item in sublist]
    unique_obs = len(set(flat_obs))

    Trans = np.random.rand(num_internal, num_internal)
    Emiss = np.random.rand(unique_obs, num_internal)
    
    Trans /= Trans.sum(axis=0)
    Emiss /= Emiss.sum(axis=0)

    model = hmm.MultinomialHMM(n_components=num_internal)
    model._set_startprob(np.array([1./num_internal  for i in range(num_internal)]))
    model._set_transmat(Trans.T)
    model._set_emissionprob(Emiss.T)

    model.fit(np.array(list(set(flat_obs))))