import nltk

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


import nltk

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

grammar7= nltk.CFG.fromstring("""
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

grammar8 = nltk.CFG.fromstring("""
  S -> QP | SP VP | SP JJ | SP PNN | VP PP
  QP -> SP WQ | SP PP WQ | SP VP WQ
  SP -> PRP NN | PRP | NN | NN SP
  VP -> VB RB | VB 
  PP -> IN DT SP | IN SP | IN
  WQ -> WP | WRB
  NN -> 'name' | 'eggs' | 'office' | 'work'
  PRP -> 'you' | 'i-me'
  WP -> 'what' | 'who' | 'when'
  WRB -> 'how' | 'where'
  JJ -> 'good' | 'okay'
  IN -> 'from' | 'to'
  RB -> 'here'
  VB -> 'live' | 'cook' | 'go'
  DT -> 'the' | 'an' | 'a'
  NNP -> letter NNP | letter
  letter -> 'J' | 'E' | 'R' | 'S'
  PNN -> 'jers'
""")

# What's your name (you name what?) - (S (O (SP (PRP you) (NN name))) (VBP (QP (WP what))))
# I'm good (I good) - (S (O (SP (PRP i))) (JJ good))
# Where you from (you from where) -> (S (O (SP (PRP you) (IN from))) (VBP (QP (WRB where))))
# Are you a student (You occupation study)
# I am cooking eggs (eggs, I-me, cook)
# Where do you live? (you, live, where?) -> (S (O (SP (PRP you))) (VBP (VB live) (WRB where)))
# Are you okay? (you, okay) ->(S (O (SP (PRP you) (NN okay))))
# My name is (I-Me, name) -> (S (O (SP (PRP i) (NN name))))
# Go to the Office (Go, to, Office) -> (S (VBP (VB go) (TO to)) (SP (NN office)))
# Go to work (Go, to, work) -> (S (VBP (VB go) (TO to)) (SP (NN work)))

grammar1.productions()
grammar8.productions()

text = 'go to work'
sent = text.split()
# rd_parser = nltk.parse.shiftreduce.ShiftReduceParser(grammar1)
rd_parser = nltk.parse.recursivedescent.RecursiveDescentParser(grammar8)
for tree in rd_parser.parse(sent):
  print(tree)
# nltk.parse.shiftreduce.demo()
# nltk.parse.recursivedescent.demo()