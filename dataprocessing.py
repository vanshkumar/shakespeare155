import os
import numpy
import nltk
from nltk.corpus import cmudict
 

d = cmudict.dict()
def nsyl(word):
  '''finds the number of syllables in a word'''
  return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]] 


def loadShakespeare():
  '''Returns a list of sonnets, with each line being a separate element'''
  f = open(os.getcwd() + '/../project2data/shakespeare.txt')
  lines = f.readlines()
  sonnets = []
  sonnet = []
  for line in lines:
    line = line.strip()
    if line.isdigit():
      sonnets.append(sonnet)
      sonnet = []
    else:
      sonnet.append(line)
  sonnets.append(sonnet)
  del sonnets[0]
  return sonnets



