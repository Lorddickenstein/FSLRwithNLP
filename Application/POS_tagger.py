import nltk
from nltk.tag import pos_tag
from easynmt import EasyNMT


def tokenizer(text):

  tokens = nltk.word_tokenize(text)

  return tokens

def pos_tag(tokens):

  pos_tagged = nltk.pos_tag(tokens)

  return pos_tagged


def main():

  text = "Good Morning"
  tokens = tokenizer(text)
  pos_tagged = pos_tag(tokens)

  print("POS TAGGED: ", pos_tagged)

main()