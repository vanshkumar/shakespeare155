import os
import numpy
# import nltk
# from nltk.corpus import cmudict
import re

# d = cmudict.dict()
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
    elif line.strip() == '':
      pass
    else:
      sonnet.append(line)
  sonnets.append(sonnet)
  del sonnets[0]
  return sonnets

def processWord(word):
  word = word.strip(',')
  word = word.strip('.')
  word = word.strip('!')
  word = word.strip('?')
  word = word.replace("'", "")
  word = word.lower()  
  return word

def createBag():
  '''Converts each word to an id'''
  a = loadShakespeare()
  words = {}
  dictid = 2
  for sonnet in a:
    for line in sonnet:
      line = re.findall(r"[\w']+", line)
      for word in line:
        word = processWord(word)
        if word not in words:
          words[word] = dictid
          dictid += 1
  return words

def convertToBag(line, bagdict):
  '''Converts line to bag representation'''
  bagrep = []
  line = re.findall(r"[\w']+", line)
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
      # 0 is a comma, 1 is a period
      output[-1] += [0]
    output[-1][-1] = 1
  return output

def exampleUsage():
  a = loadShakespeare()
  print a[100][0]
  print convertToBag(a[100][0])