from Application.NLP.Repos import read_file


def separate_words(sentence):
  return sentence.split(' ')


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
      word = 'GIRLFRIEND'
    elif word == 'BF':
      word = 'BOYFRIEND'
    elif word == 'OK':
      word = 'OKAY'
    elif word in persons:
      if sentence[index + 1] == 'OCCUPATION':
        word = persons[word]
        index += 1

    new_sentence.append(word)
    index += 1
  return new_sentence


def tokenize(tokens):
  tokens = build_letters(tokens)
  tokens = anotate(tokens)
  return tokens


persons = {}
for occupation, person in read_file('persons.txt'):
  persons[occupation] = person


if __name__ == '__main__':
  sentence = ['I-Me', 'J', 'E', 'R', 'S', 'O', 'N', 'and', 'I-Me', 'Live', 'in', 'C', 'U', 'B', 'A', 'O', 'HO', 'I-Me', 'DRAWING', 'OCCUPATION']
  sentence = build_letters(sentence)
  print(sentence)
  sentence = anotate(sentence)
  print(sentence)