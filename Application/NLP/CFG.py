import nltk
import re
from Utilities import read_file
from nltk import Tree
from nltk.util import flatten

word_list = read_file('dictionary.txt')

def update_grammar(word='lagoon'):
  grammar = f"""
    S -> QP | SP VP | SP JJ | SP NNP | VP PP | SP
    QP -> SP WQ | SP PP WQ | SP VP WQ
    SP -> PRP NN | PRP | NN | NN SP 
    VP -> VB RB | VB 
    PP -> IN SP | IN
    WQ -> WP | WRB
    NN -> 'name' | 'egg' | 'office' | 'work' | 'person' | 'student' 
    PRP -> 'you' | 'i-me' | 'he-she'
    WP -> 'what' | 'who' | 'when'
    WRB -> 'how' | 'where'
    JJ -> 'GOOD' | 'okay'
    IN -> 'from' | 'to'
    RB -> 'here'
    VB -> 'live' | 'cook' | 'go' | 'study' | 'come'
    NNP -> {word} 
  """
  return grammar

class ParseError(Exception):
  pass

# Tokenize a string.
# Tokens yielded are of the form (type, string)
# Possible values for 'type' are '(', ')' and 'WORD'
def tokenize(s, reg_ex):
    toks = re.compile(reg_ex)
    for match in toks.finditer(s):
        s = match.group(0)
        if s[0] == ' ':
            continue
        if s[0] in '()':
            yield (s, s)
        else:
            yield ('word', s)


# Parse once we're inside an opening bracket.
def parse_inner(toks):
    ty, name = next(toks)
    if ty != 'word': raise ParseError
    children = []
    while True:
        ty, s = next(toks)
        if ty == '(':
            children.append(parse_inner(toks))
        elif ty == ')':
            return (name, children)

# Parse this grammar:
# ROOT ::= '(' INNER
# INNER ::= WORD ROOT* ')'
# WORD ::= [A-Za-z]+
def parse_root(toks):
    ty, _ = next(toks)
    if ty != '(': raise ParseError
    return parse_inner(toks)

def show_children(tree, pattern):
    name, children = tree
    if not children: return ""
    pattern = "(%s -> %s) " % (name, ' '.join(child[0] for child in children))
    for child in children:
      pattern += show_children(child, pattern)
    # return children, ("(%s -> %s) " % (name, ' '.join(child[0] for child in children)))
    return pattern

def get_terminal(tree, terminals):
  name, children = tree
  # print(children)
  if not children:
    # print(name)
    terminals.append(name)
    return terminals
  for child in children:
    terminals = get_terminal(child, terminals)
  return terminals

def gen_sentence(terminals, pattern):
  tree = ""
  string = "unrecognized"

  # you name what
  if pattern == '(S -> QP) (QP -> SP WQ) (SP -> PRP NN) (WQ -> WP)':
    prps, nn, wp = terminals
    prps = 'your' if prps.split(' ')[1] == 'you' else 'my' if prps.split(' ')[1] == 'i-me' else 'his-her' 
    string = f'(S (QP (WQ (WP {wp.split()[1]}) (FWA is)) (SP (PRPS {prps}) (NN {nn.split()[1]}))))'
  # you live where
  elif pattern == '(S -> QP) (QP -> SP VP WQ) (SP -> PRP) (VP -> VB) (WQ -> WRB)':
    prp, vb, wrb = terminals
    fwc = 'does' if prp.split(' ')[1] == 'he-she' else 'do' 
    string = f'(S (QP (WQ (WRB {wrb.split()[1]}) (FWC {fwc})) (SP (PRP {prp.split()[1]})) (VP (VB {vb.split()[1]}))))'
  # you from where
  elif pattern == '(S -> QP) (QP -> SP PP WQ) (SP -> PRP) (PP -> IN) (WQ -> WRB)':
    prp, ins, wrb = terminals
    prp = prp.split(' ')[1]
    fwa = 'am' if prp == 'i-me' else 'are' if prp == 'you' else 'is'
    string = f'(S (QP (WQ (WRB {wrb.split()[1]}) (FWA {fwa})) (SP (PRP {prp})) (PP (IN {ins.split()[1]}))))'
  # i-me name +
  elif pattern == '(S -> SP NNP) (SP -> PRP NN)':
    prps, nn, nnp = terminals
    prps = 'your' if prps.split(' ')[1] == 'you' else 'my' if prps.split(' ')[1] == 'i-me' else 'his-her' 
    string = f'(S (SP (PRPS {prps}) (NN {nn.split()[1]}) (FWA is)) (NNP {nnp.split()[1]}))'
  # i-me good
  elif pattern == '(S -> SP JJ) (SP -> PRP)':
    prp, jj = terminals
    fwa = 'am' if prp.split(' ')[1] == 'i-me' else 'are' if prp.split(' ')[1] == 'you' else 'is'
    string = f'(S (SP (PRP {prp.split()[1]}) (FWA {fwa})) (JJ {jj.split()[1]}))'
  # egg i-me cook
  elif pattern == '(S -> SP VP) (SP -> NN SP) (SP -> PRP) (VP -> VB)':
    nn, prp, vb = terminals
    fwa = 'am' if prp.split(' ')[1] == 'i-me' else 'are' if prp.split(' ')[1] == 'you' else 'is'
    vb = vb.split(' ')[1] 
    prp = prp.split(' ')[1]
    vb = vb if prp == 'you' or prp == 'i-me' else vb + 's'
    string = f'(S (SP (PRP {prp})) (VP (VB {vb}) (NN {nn.split()[1]})))'
  # go to office
  elif pattern == '(S -> VP PP) (VP -> VB) (PP -> IN SP) (SP -> NN)':
    vb, ins, nn = terminals
    string = f'(S (VP (VB {vb.split()[1]})) (PP (IN {ins.split()[1]}) (SP (NN {nn.split()[1]}))))'

  tree = Tree.fromstring(string)
  return ' '.join(flatten(tree))

def naturalized_sentence(text):
  tree = None
  sent = text.split()

  pnn_list = [word for word in sent if word not in word_list]
  pnn_list = '\'' + '\' | \''.join(pnn_list) + '\'' if len(pnn_list) else ''
  grammar = nltk.CFG.fromstring(update_grammar(word=pnn_list))

  # rd_parser = nltk.parse.shiftreduce.ShiftReduceParser(grammar6, trace=3)
  rd_parser = nltk.parse.recursivedescent.RecursiveDescentParser(grammar)
  for tree in rd_parser.parse(sent):
    tree = tree
    print(tree)
  try: 
    tree = parse_root(tokenize(str(tree),' +|[A-Z a-z -]+|[()]'))
    pattern = show_children(parse_root(tokenize(str(tree), ' +|[A-Za-z-]+|[()]')), "")
    print(pattern)
    terminals = get_terminal(tree,[])
    sentence = gen_sentence(terminals, pattern.strip())
  except Exception as EXE:
    sentence ="unrecognized"

  return sentence

<<<<<<< HEAD
grammar1 = nltk.CFG.fromstring("""
  S -> QP | VP SP | QP SP | O VBP | WP | O JJ | VBP SP
  SP -> PRP NN | PRP IN | NN | PRP |
  QP -> WP NN | WRB A2 | WP | WRB
  O -> SP A1 | SP A2 A3 | SP
  VBP -> VB RB | VB WRB | VB TO | QP
  A1 -> "is"
  A2 -> "are"
  A3 -> "am"
  IN -> "from"
  JJ -> "good"
  WP -> "what" | "when"
  WDT -> "which"
  WRB -> "where" | "how"
  NN -> "name" | "okay" | "office" | "work"
  PRP -> "you" | "i"
  RB -> "here" 
  TO -> "to"
  VB -> "live" | "go"
  """)
# (S (QP (WP What)) (SP (PRP you) (NN name)))
grammar2 = nltk.CFG.fromstring("""
  S -> QP | VP SP | QP A1 SP | O SP | O VBP
  SP -> NN | PRP NN 
  QP ->  WP
  WP -> "what"
  A1 -> "is"
  NN -> "name"
  PRP -> "your"
  """)

grammar4 = nltk.CFG.fromstring("""
  S -> A
  A -> QP SP  
  SP -> NN
  QP -> WP O
  O -> PRP
  WP -> "what"
  PRP -> "your"
  NN -> "name"
  A1 -> "is"
  """)

grammar7 = nltk.CFG.fromstring("""
  S -> A
  A -> QP SP  
  SP -> NN
  QP -> WP O
  O -> PRP
  WP -> "what"
  PRP -> "your"
  NN -> "name"
  A1 -> "is"
  """)
 	
grammar3 = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
  """)

grammar6 = nltk.CFG.fromstring("""
  S -> NP VP
  PP -> P NP
  NP -> Det N | Det N PP
  VP -> V NP | VP PP
  Det -> 'DT'
  N -> 'NN'
  V -> 'VBZ'
  P -> 'PP'
  """)

# What's your name (you name what?) - (S (O (SP (PRP you) (NN name))) (VBP (QP (WP what))))
# I'm good (I good) - (S (O (SP (PRP i))) (JJ good))
# Where you from (you from where) -> (S (O (SP (PRP you) (IN from))) (VBP (QP (WRB where))))
# I live here -> (S (O (SP (PRP i))) (VBP (VB live) (RB here)))
# How are you (How you?) -> (S (QP (WRB how)) (SP (PRP you)))
# Where do you live? (you, live, where?) -> (S (O (SP (PRP you))) (VBP (VB live) (WRB where)))
# Are you okay? (you, okay) ->(S (O (SP (PRP you) (NN okay))))
# My name is (I-Me, name) -> (S (O (SP (PRP i) (NN name))))
# Go to the Office (Go, to, Office) -> (S (VBP (VB go) (TO to)) (SP (NN office)))
# Go to work (Go, to, work) -> (S (VBP (VB go) (TO to)) (SP (NN work)))

grammar1.productions()

text = "What you name"
sent = text.split()
# rd_parser = nltk.parse.shiftreduce.ShiftReduceParser(grammar1)
rd_parser = nltk.parse.recursivedescent.RecursiveDescentParser(grammar1)
for tree in rd_parser.parse(sent):
  print(tree)
# nltk.parse.shiftreduce.demo()
# nltk.parse.recursivedescent.demo()
=======
if __name__ == '__main__':
  text = 'egg i-me cook'
  text = naturalized_sentence(text)
  print(text)
  # print(word_list)
>>>>>>> 563668ae54441bf4c8c7c69a97400b814a1612f6
