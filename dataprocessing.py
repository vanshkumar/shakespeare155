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
    for punct in [',', '.', '?', '!', ':', ';']:
      line = line.replace(punct, ' ' + punct)    
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
    speechdict[word] = nltk.pos_tag([word])[0][1]
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
  for sonnet in a:
    for line in sonnet:
      output.append(convertToBag(line, bagdict))
  return output, bagdict
