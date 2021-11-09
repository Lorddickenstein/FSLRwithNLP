import nltk
nltk.download('averaged_perceptron_tagger')
from easynmt import EasyNMT

def tokenizer(text):
  return nltk.word_tokenize(text)

def pos_tag(tokens):
  return nltk.pos_tag(tokens)


def main():

  text = "Good Morning"
  tokens = tokenizer(text)
  pos_tagged = pos_tag(tokens)

  print("POS TAGGED: ", pos_tagged)

main()