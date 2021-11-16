import nltk
import re
from nltk import Tree
from nltk.util import flatten

data_list = ['name', 'egg','office','work','person','student','you','i-me','he-she','what','who','when','how','where','good','okay','from','to','here','live',
             'cook','go','study']


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
    JJ -> 'good' | 'okay'
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

  pnn_list = [word for word in sent if word not in data_list]
  pnn_list = '\'' + '\' | \''.join(pnn_list) + '\'' if len(pnn_list) else ''
  grammar = nltk.CFG.fromstring(update_grammar(word=pnn_list))

  # rd_parser = nltk.parse.shiftreduce.ShiftReduceParser(grammar6, trace=3)
  rd_parser = nltk.parse.recursivedescent.RecursiveDescentParser(grammar)
  for tree in rd_parser.parse(sent):
    tree = tree

  try: 
    tree = parse_root(tokenize(str(tree),' +|[A-Z a-z -]+|[()]'))
    pattern = show_children(parse_root(tokenize(str(tree), ' +|[A-Za-z-]+|[()]')), "")
    terminals = get_terminal(tree,[])
    sentence = gen_sentence(terminals, pattern.strip())
  except Exception as EXE:
    sentence ="unrecognized"

  return sentence

if __name__ == '__main__':

  text = 'i-me Joshua'
  text = naturalized_sentence(text)
  print(text)
