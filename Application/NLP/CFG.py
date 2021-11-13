import nltk

grammar1 = nltk.CFG.fromstring("""
  S -> QP | VP SP | QP SP | O SP | O VBP | WP | SP
  SP -> PRP NN | PRP IN | NN
  QP -> WP | WRB A2
  O -> PRP | PRP A1 | PRP A2 A3 | PRP
  A1 -> "is"
  A2 -> "are"
  A3 -> "ay"
  IN -> "from"
  WP -> "What" | "How" | "When"
  WDT -> "Which"
  WRB -> "Where"
  NN -> "name" | "estudyante"
  PRP -> "you" | "I"
  RB -> "here" 
  VBP -> "live"
  """)
# (S (QP (WP What)) (SP (PRP you) (NN name)))
grammar2 = nltk.CFG.fromstring("""
  S -> QP | VP SP | QP SP | O SP | O VBP
  SP -> "NN" | "NN PRP" 
  QP -> "WP" | "WP A1"
  O -> "PRP" | "PRP A3" | "PRP A2 A3"
  """)

grammar4 = nltk.CFG.fromstring("""
  S -> A
  A -> QP SP | 
  SP -> NN | PRP NN 
  QP -> WP 
  O -> A1 PRP | PRP A3 | PRP A2 A3
  WP -> "What"
  PRP -> "you"
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

# What's your name (What name?) - S -> QP SP -> QP -> WP A1 -> "Ano" -> "ang" -> SP -> NN PRP -> NN -> "pangalan" -> PRP -> "ikaw"
# Are you a student? (you student?) - S -> O SP -> O -> PRP A2 A3 -> PRP -> "ikaw" -> A2 -> "ba" -> A3 -> "ay" -> SP -> NN -> "estudyante"
# I'm good (I good) - S -> O VBP -> O -> PRP A2 -> PRP -> "ako" -> A2 -> "ay" -> VBP -> "mabuti"
# Where you from 
# I live here


grammar1.productions()

text = "What you name"
sent = text.split()
# rd_parser = nltk.parse.shiftreduce.ShiftReduceParser(grammar1)
rd_parser = nltk.parse.recursivedescent.RecursiveDescentParser(grammar1)
for tree in rd_parser.parse(sent):
  print(tree)
# nltk.parse.shiftreduce.demo()
# nltk.parse.recursivedescent.demo()