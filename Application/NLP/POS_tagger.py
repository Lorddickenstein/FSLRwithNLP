import nltk
<<<<<<< HEAD:Application/POS_tagger.py
from nltk.tag import pos_tag
=======
nltk.download('averaged_perceptron_tagger')
>>>>>>> f9ad38df7212fbcb0acacc3caaac68d668dae15f:Application/NLP/POS_tagger.py
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