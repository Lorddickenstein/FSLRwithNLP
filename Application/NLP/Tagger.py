#####################################################################
# Author: Jerson Destacamento, Joshua Cruzat, Rocella Legaspi       #
# Date: October-December                                            #
# Program Title: Tagger.py                                          #
# Description: Transforms the inputs into a format that is          #
#              recognizable by the program.                         #
# General System Design: Data Conversion, NLP Part                  #
# Data structures, Algorithms, Controls: List, Dictionary, File     #
#              Handling, Try Except                                 #
# Requirements: None		                                        #
#####################################################################

import os
from NLP.Utilities import read_dictionary

def separate_words(sentence):
  return sentence.split(' ')


def is_alpha(letter):
  """ Checks if the recognized sign belongs to the english alphabet or not. """
  letters = ['A', 'B', 'C', 'D', 'E',
             'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O',
             'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y',
             'Z']
  return True if letter in letters else False


def build_letters(sentence):
  """ Groups all the recognized letters to form a word. As long as the next recognized sign is an alphabet, it
      will be appended as one word.
  """
  index = 0
  new_sentence = []
  while index < len(sentence):
    word = sentence[index].upper()

    # Convert fingerspell into one word
    if is_alpha(word):
      lett_index = index
      lett_concat = ''
      while lett_index < len(sentence):
        word = sentence[lett_index].upper()
        if not is_alpha(word):
          break
        lett_concat += word
        lett_index += 1
      word = lett_concat
      index = lett_index - 1
    new_sentence.append(word)
    index += 1
  return new_sentence


def anotate(sentence):
  """ Expands or gives meaning to the recognized sign language if it has other meanings other than the meaning of
      the word that was used to sign the actual word.
  """
  new_sentence = []
  index = 0

  while index < len(sentence):
    word = sentence[index]
    if word == 'HO':
      word = 'HELLO'
    elif word == 'GF':
      word = 'GIRLFRIEND'
    elif word == 'BF':
      word = 'BOYFRIEND'
    elif word == 'OK':
      word = 'OKAY'
    elif word in persons:
      try:
        if sentence[index + 1] == 'PERSON':
          word = persons[word]
          index += 1
      except Exception:
        pass

    new_sentence.append(word)
    index += 1

  return new_sentence


def tokenization(tokens):
  """ Creates a list of all the words that are recognized and will be treated as tokens. """
  tokens = build_letters(tokens)
  tokens = anotate(tokens)
  return tokens


cwd = os.getcwd()
cwd = cwd + '\\NLP' if '\\NLP' not in cwd else cwd
path = os.path.join(cwd, 'persons.txt')
persons = {occupation: person for occupation, person in read_dictionary(path, dict_format=True)}

if __name__ == '__main__':
  sentence = ['I-Me', 'J', 'E', 'R', 'S', 'O', 'N', 'and', 'I-Me', 'Live', 'in', 'C', 'U', 'B', 'A', 'O', 'HO', 'I-Me', 'DRAWING', 'PERSON']
  sentence = build_letters(sentence)
  print(sentence)
  sentence = anotate(sentence)
  print(sentence)
