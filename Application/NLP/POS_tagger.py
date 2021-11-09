import nltk
from Application.NLP.Repos import read_file
from easynmt import EasyNMT
nltk.download('averaged_perceptron_tagger')

def tokenizer(text):
  return nltk.word_tokenize(text)

def pos_tag(tokens):
  return nltk.pos_tag(tokens)


def main():

  text = "Good Morning"
  tokens = tokenizer(text)
  # print(tokens)
  pos_tagged = pos_tag(tokens)

  print("POS TAGGED: ", pos_tagged)

# main()

def test():
  dictionary = read_file('D:\Documents\Thesis\FSLRwithNLP\Application\\NLP\dictionary.txt')
  print(pos_tag(dictionary))

test()