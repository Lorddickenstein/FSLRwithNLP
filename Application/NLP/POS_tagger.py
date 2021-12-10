import nltk
from Application.NLP.Utilities import read_file
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

def tokenizer(text):
  tokens = nltk.word_tokenize(text)
  return tokens

def pos_tag(tokens):
  pos_tagged = nltk.pos_tag(tokens)
  return pos_tagged

def main(text):
  tokens = tokenizer(text)
  pos_tagged = pos_tag(tokens)

  print("POS TAGGED: ", pos_tagged)

persons = {}
# for occupation, person in read_file('NLP\persons.txt'):
#   persons[occupation] = person


def is_alpha(letter):
  letters = ['A', 'B', 'C', 'D', 'E',
             'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O',
             'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y',
             'Z']
  return True if letter in letters else False


def build_letters(sentence):
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
    elif word in persons:
      if sentence[index + 1] == 'OCCUPATION':
        word = persons[word]
        index += 1

    new_sentence.append(word)
    index += 1
  return new_sentence


# def pos_tag(tokens):
#   tokens = build_letters(tokens)
#   tokens = anotate(tokens)
#   return tokens


if __name__ == '__main__':
  # sentence = ['I-Me', 'J', 'E', 'R', 'S', 'O', 'N', 'and', 'I-Me', 'Live', 'in', 'C', 'U', 'B', 'A', 'O', 'HO', 'I-Me', 'DRAWING', 'OCCUPATION']
  # sentence = build_letters(sentence)
  # print(sentence)
  # sentence = anotate(sentence)
  # print(sentence)

  main('Good Morning')
