import nltk
from Application.NLP.Repos import read_file
from easynmt import EasyNMT
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

def is_alpha(letter):
  letters = ['A', 'B', 'C', 'D', 'E',
             'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O',
             'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y',
             'Z']
  return True if letter in letters else False


def tokenizer(sentence):
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
  person = {'STUDY': 'STUDENT', 'LAW': 'LAWYER', 'TEACH': 'TEACHER', 'DRAWING': 'ARCHITECT'}
  new_sentence = []
  index = 0
  while index < len(sentence):
    word = sentence[index]
    if word == 'HO':
      word = 'HELLO'
    elif word == 'GF':
      word = 'G'
    elif word == 'BF':
      word = 'BF'
    elif word == 'OK':
      word = 'Okay'
    elif word in person:
      if sentence[index + 1] == 'OCCUPATION':
        word = person[word]
        index += 1

    new_sentence.append(word)
    index += 1
  return new_sentence


def pos_tag(tokens):
  return nltk.pos_tag(tokens)


def main():

  text = "Good Morning"
  tokens = tokenizer(text)
  print(tokens)
  pos_tagged = pos_tag(tokens)

def test():
  dictionary = read_file('D:\Documents\Thesis\FSLRwithNLP\Application\\NLP\dictionary.txt')
  print(pos_tag(dictionary))

if __name__ == '__main__':
  sentence = ['I-Me', 'J', 'E', 'R', 'S', 'O', 'N', 'and', 'I-Me', 'Live', 'in', 'C', 'U', 'B', 'A', 'O', 'HO', 'I-Me', 'DRAWING', 'OCCUPATION']
  sentence = tokenizer(sentence)
  print(sentence)
  sentence = anotate(sentence)
  print(sentence)
  sentence = pos_tag(sentence)
  print(sentence)
