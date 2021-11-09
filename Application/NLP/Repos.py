import os

def read_file(name='None'):
    if name:
        f = open(name, "r")
        dictionary = []
        for line in f:
            dictionary.append(line.strip('\n'))
    return dictionary

# dictionary = read_file('D:\Documents\Thesis\FSLRwithNLP\Application\\NLP\dictionary.txt')
# print(dictionary)