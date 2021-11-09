import nltk
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

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