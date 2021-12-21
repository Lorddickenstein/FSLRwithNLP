#####################################################################
# Author: Jerson Destacamento, Joshua Cruzat, Rocella Legaspi       #
# Date: October-December                                            #
# Description: Contains the natural language processing using the   #
#              context free grammar. The cfg were designed by the   #
#              programmers and reads from the dictionary.txt file   #
#              to get the pos tag of each recognized word.          #
# General System Design: Sentence Generator, NLP Part               #
# Requirements: Camera (Hardware)                                   #
#####################################################################

import os
import nltk
import re
from Application.NLP.Utilities import read_file
from nltk import Tree
from nltk.util import flatten

# Dynamically reads the dictionary.txt from whatever working directory is using this program
cwd = os.getcwd()
cwd = cwd + '\\NLP' if '\\NLP' not in cwd else cwd
path = os.path.join(cwd, 'dictionary.txt')
word_list = read_file(path)
dictionary = read_file(path, dict_format=True)


def get_list(pos_tag):
    """ Takes the list of words from the dictionary to determine the production of the cfg. """
    temp_list = [word for word, tag in dictionary if tag == pos_tag]
    return '\'' + '\' | \''.join(temp_list) + '\''


# Acquire all the productions from the dictionary
nn_list = get_list('NN')
prp_list = get_list('PRP')
wp_list = get_list('WP')
wrb_list = get_list('WRB')
jj_list = get_list('JJ')
in_list = get_list('IN')
vb_list = get_list('VB')


def update_grammar(nn=nn_list, prp=prp_list, wp=wp_list, wrb=wrb_list, jj=jj_list, ins=in_list, vb=vb_list, nnp=''):
    """ Dynamically updates the production of the cfg to accommodate non-recognizable words or words
        that are not in the dictionary and label them as Proper Nouns.

        Args:
            All the productions lists

        Returns:
            grammar: The context free grammar that the program uses in recognizing the sentence.
    """
    grammar = f"""
    S -> QP | SP VP | SP JJ | SP NNP | VP PP | SP
    QP -> SP WQ | SP PP WQ | SP VP WQ
    SP -> PRP NN | PRP | NN | NN SP 
    VP -> VB RB | VB 
    PP -> IN SP | IN
    WQ -> WP | WRB
    NN -> {nn}
    PRP -> {prp}
    WP -> {wp}
    WRB -> {wrb}
    JJ -> {jj}
    IN -> {ins}
    RB -> 'here'
    VB -> {vb}
    NNP -> {nnp}
    """
    return grammar

class ParseError(Exception):
    pass


def tokenize(s, reg_ex):
    """ Tokenizes a string. Tokens yielded are of the form (type, string). Possible values for
        'type' are '(', ')' and 'WORD'
    """
    toks = re.compile(reg_ex)
    for match in toks.finditer(s):
        s = match.group(0)
        if s[0] == ' ':
            continue
        if s[0] in '()':
            yield (s, s)
        else:
            yield ('word', s)


def parse_inner(toks):
    """ Parse once we're inside an opening bracket."""
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
    """ Parses the grammar:
            ROOT ::= '(' INNER
            INNER ::= WORD ROOT* ')'
            WOD ::= [A-Za-z]+
    """
    ty, _ = next(toks)
    if ty != '(': raise ParseError
    return parse_inner(toks)


def show_children(tree, pattern):
    """ A recursion function that displays the children of a given tree in the form of ('tag' -> 'word').

        Args:
            tree: nltk Tree. The parse tree of the given grammar that results to the structure of the sentence.
            pattern: String. A state that keeps record of the pattern of the specific sentence structure. Once
                the program exits this function, a pattern that specifies the tracing of the sentence from the
                grammar.

        Returns:
            pattern
    """
    name, children = tree
    if not children: return ""

    pattern = "(%s -> %s) " % (name, ' '.join(child[0] for child in children))
    for child in children:
      pattern += show_children(child, pattern)

    return pattern


def get_terminal(tree, terminals):
    """ A recursion function that acquires all the terminals of a given tree. Useful in acquiring the
        actual words that are used by the user in the sentence structure that he/she
        produces after the recognized text.

        Args:
            tree: nltk Tree. The parse tree of where the terminals are going to be derived from.
            terminals: String List. A state that keeps record of all the leaf nodes or 'end nodes' which
                are the terminals of the grammar. The terminals are used to acquire the actual word used
                to derive a sentence based from those words.

        Returns:
            terminals: String List. A list of all the terminals found in the given parse tree that contains all the
                end nodes and their pos tag.
    """
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
    """ Generates a sentence by matching the patterns to the known sentence structure of the program.
        The program takes the terminals from the parse tree and then, transforms them into a natural language
        by providing the necessary improvements of that sentence in the english form. This function matches
        the patterns and formulates another pattern in the natural language form and try to create a parse
        of that form.

        Args:
            terminals: String List. A list of all the terminals found in the given parse tree that contains all the
                end nodes and their pos tag.
            pattern: String. The pattern which represents grammar of the recognized sentence. The pattern is the
                expansion of the parse tree.

        Returns:
            Sentence: String. A formalized string of words that represents the grammatically correct english sentence
                structure of the formerly barok fsl.
    """
    tree = ""
    string = "unrecognized"

    # you name what
    if pattern == '(S -> QP) (QP -> SP WQ) (SP -> PRP NN) (WQ -> WP)':
        prps, nn, wp = terminals
        prps = 'YOUR' if prps.split(' ')[1] == 'YOU' else 'MY' if prps.split(' ')[1] == 'I-ME' else 'HIS-HER'
        string = f'(S (QP (WQ (WP {wp.split()[1]}) (FWA IS)) (SP (PRPS {prps}) (NN {nn.split()[1]}))))'
    # you live where
    elif pattern == '(S -> QP) (QP -> SP VP WQ) (SP -> PRP) (VP -> VB) (WQ -> WRB)':
        prp, vb, wrb = terminals
        prp = prp.split(' ')[1]
        fwc = 'DOES' if prp == 'HE-SHE' else 'DO'
        prp = 'I' if prp == 'I-ME' else prp
        string = f'(S (QP (WQ (WRB {wrb.split()[1]}) (FWC {fwc})) (SP (PRP {prp})) (VP (VB {vb.split()[1]}))))'
    # you from where
    elif pattern == '(S -> QP) (QP -> SP PP WQ) (SP -> PRP) (PP -> IN) (WQ -> WRB)':
        prp, ins, wrb = terminals
        prp = prp.split(' ')[1]
        fwa = 'AM' if prp == 'I-ME' else 'ARE' if prp == 'YOU' else 'IS'
        prp = 'I' if prp == 'I-ME' else prp
        string = f'(S (QP (WQ (WRB {wrb.split()[1]}) (FWA {fwa})) (SP (PRP {prp})) (PP (IN {ins.split()[1]}))))'
    # i-me name +
    elif pattern == '(S -> SP NNP) (SP -> PRP NN)':
        prps, nn, nnp = terminals
        prps = 'YOUR' if prps.split(' ')[1] == 'YOU' else 'MY' if prps.split(' ')[1] == 'I-ME' else 'HIS-HER'
        string = f'(S (SP (PRPS {prps}) (NN {nn.split()[1]}) (FWA IS)) (NNP {nnp.split()[1]}))'
    # i-me good
    elif pattern == '(S -> SP JJ) (SP -> PRP)':
        prp, jj = terminals
        prp = prp.split(' ')[1]
        fwa = 'AM' if prp == 'I-ME' else 'ARE' if prp == 'YOU' else 'IS'
        prp = 'I' if prp == 'I-ME' else prp
        string = f'(S (SP (PRP {prp}) (FWA {fwa})) (JJ {jj.split()[1]}))'
    # egg i-me cook
    elif pattern == '(S -> SP VP) (SP -> NN SP) (SP -> PRP) (VP -> VB)':
        nn, prp, vb = terminals
        prp = prp.split(' ')[1]
        fwa = 'AM' if prp == 'I-ME' else 'ARE' if prp == 'YOU' else 'IS'
        vb = vb.split(' ')[1]
        vb = vb if prp == 'YOU' or prp == 'I-ME' else vb + 'S'
        prp = 'I' if prp == 'I-ME' else prp
        string = f'(S (SP (PRP {prp})) (VP (VB {vb}) (NN {nn.split()[1]})))'
    # go to office
    elif pattern == '(S -> VP PP) (VP -> VB) (PP -> IN SP) (SP -> NN)':
        vb, ins, nn = terminals
        string = f'(S (VP (VB {vb.split()[1]})) (PP (IN {ins.split()[1]}) (SP (NN {nn.split()[1]}))))'
    # you study person
    elif pattern == '(S -> SP) (SP -> PRP NN)':
        prp, nn = terminals
        fwa = 'ARE' if prp.split(' ')[1] == 'YOU' else 'IS' if prp.split(' ')[1] == 'HE-SHE' else 'AM'
        prp = 'I' if prp.split()[1] == 'I-ME' else prp.split()[1]
        string = f'(S (SP (PRP {prp}) (FWA {fwa}) (SP (NN {nn.split()[1]}))))'

    tree = Tree.fromstring(string)
    return ' '.join(flatten(tree))

def naturalized_sentence(tokens):
    """ This function takes a list of tokens to generate a sentence based on those tokens. Any unrecognized
        word will be taken as NNP which is Proper Noun. It will create the tree based on the sentence structure
        of the barok sentence.

        Args:
            tokens: List. A string list of all the tokens identified by the program after the capture of signs.
                The tokens have undergone annotations, and several transformation to get to the desired format.

        Returns:
            sentence: String. A formalized string of words that represents the grammatically correct english sentence
                structure of the formerly barok fsl. It returns 'Unrecognized' if the sentence structure in the barok
                sentence is not recognized by the program.

        Raises:
            Exception: if the word is not yet implemented in the sentence structure such as 'Yes'; if the tree is not
                yet created, for example, a sentence that may be formed using the words in the word pool but not yet
                recorded;
    """
    tree = None

    # Update the grammar for all unrecognized word to be labeled as proper nouns
    pnn_list = [word for word in tokens if word not in word_list]
    pnn_list = '\'' + '\' | \''.join(pnn_list) + '\'' if len(pnn_list) else ''
    grammar = nltk.CFG.fromstring(update_grammar(nnp=pnn_list))
    # Create the tree
    rd_parser = nltk.parse.recursivedescent.RecursiveDescentParser(grammar)
    for tree in rd_parser.parse(tokens):
        tree = tree

    try:
        tree = parse_root(tokenize(str(tree), ' +|[A-Z a-z -]+|[()]'))
        pattern = show_children(parse_root(tokenize(str(tree), ' +|[A-Za-z-]+|[()]')), "")
        terminals = get_terminal(tree, [])
        sentence = gen_sentence(terminals, pattern.strip())
    except Exception as EXE:
        sentence = "Unrecognized"

    return sentence


if __name__ == '__main__':
    text = 'YOU NAME WHAT'
    text = text.split()
    text = naturalized_sentence(text)
    print(text)
