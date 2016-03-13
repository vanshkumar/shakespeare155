import os
import numpy
import nltk
from nltk.corpus import cmudict

d = cmudict.dict()
def nsyl(word):
  '''finds the number of syllables in a word'''
  try:
    out = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
  except KeyError:
    return 0
  return out[0]


def loadShakespeare():
  '''Returns a list of sonnets, with each line being a separate element'''

  # Pick which dataset to train on
  # f = open(os.getcwd() + '/../project2data/shakespeare.txt')
  f = open(os.getcwd() + '/../project2data/shakespeare_spenser.txt')
  lines = f.readlines()
  sonnets = []
  sonnet = []
  for line in lines:
    line = line.strip()
    for punct in [',', '.', '?', '!', ':', ';']:
      line = line.replace(punct, '')
    if line.isdigit():
      sonnets.append(sonnet)
      sonnet = []
    elif line.strip() == '':
      pass
    else:
      sonnet.append(line)
  sonnets.append(sonnet)
  del sonnets[0]
  return sonnets

def processWord(word):
  word = word.replace("'", "")
  word = word.replace('(', '')
  word = word.replace(')', '')
  
  word = word.lower()  
  return word

def createBag():
  '''Converts each word to an id'''
  a = loadShakespeare()
  words = {}
  dictid = 0
  for sonnet in a:
    for line in sonnet:
      line = line.split(' ')
      for word in line:
        word = processWord(word)
        if word not in words:
          words[word] = dictid
          dictid += 1
  return words

def partsofSpeech(bagdict):
  speechdict = {}
  for word in bagdict.keys():
    speechdict[word] = str(nltk.pos_tag([word], tagset = 'universal')[0][1])
  return speechdict

def convertToBag(line, bagdict):
  '''Converts line to bag representation'''
  bagrep = []
  line = line.split(' ')
  for word in line:
    word = processWord(word)
    bagrep.append(bagdict[word])
  return bagrep

def outputStream():
  a = loadShakespeare()
  bagdict = createBag()
  output = []
  rhyme_dict = {}
  for sonnet in a:
    sonnet_bagged = []
    for line in sonnet:
      bagged = convertToBag(line, bagdict)
      sonnet_bagged.append(bagged)
      output.append(bagged)

    for i in [0, 4, 8]:
      rhyme_dict[sonnet_bagged[i][-1]] = sonnet_bagged[i+2][-1]
      rhyme_dict[sonnet_bagged[i+1][-1]] = sonnet_bagged[i+3][-1]

    if len(sonnet_bagged) == 14:
      rhyme_dict[sonnet_bagged[12][-1]] = sonnet_bagged[13][-1]

  return output, bagdict, rhyme_dict