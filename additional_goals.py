import numpy as np
from dataprocessing import *
from train import predictSequence


if __name__ == '__main__':
    num_internal = 50
    length = 100

    iddict = np.load(os.getcwd() + "/data/iddict.npy").item()
    T = np.load(os.getcwd() + "/data/trans" + str(num_internal) + ".npy")
    E = np.load(os.getcwd() + "/data/emiss" + str(num_internal) + ".npy")

