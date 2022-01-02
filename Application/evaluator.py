from Application.NLP import Generator
from Application.NLP.Utilities import read_file, write_file
import os

# read files from the directory
cwd = os.getcwd()
cwd = cwd + '\\NLP\\Evaluation' if '\\NLP\\Evaluation' not in cwd else cwd

input_path = os.path.join(cwd, 'sentence_inputs.txt')
inputs = read_file(input_path, 'input')

reference_path = os.path.join(cwd, 'reference_translations.txt')
reference = read_file(reference_path, 'reference')

candidate_path = os.path.join(cwd, 'candidate_translations.txt')

if __name__ == '__main__':
    translation = []
    for input in inputs:
        tokens = input.split()
        sentence = Generator.naturalized_sentence(tokens)
        translation.append(sentence)
    write_file(candidate_path, translation)
