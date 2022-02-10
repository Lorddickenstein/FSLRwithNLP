from Application.NLP import Generator
from Application.NLP.Utilities import read_file, write_file
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import warnings
# SUPPRESS WARNINGS
# warnings.filterwarnings("ignore")

# read files from the directory
cwd = os.getcwd()
cwd = cwd + '\\NLP\\Evaluation' if '\\NLP\\Evaluation' not in cwd else cwd

input_path = os.path.join(cwd, 'sentence_inputs.txt')
inputs = read_file(input_path, 'input')
reference_path = os.path.join(cwd, 'reference_translations.txt')
reference = read_file(reference_path, 'reference')
candidate_path = os.path.join(cwd, 'candidate_translations.txt')


def calculate_cummulative_ngram(reference, candidate):
    chencherry = SmoothingFunction()
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    return [sentence_bleu(reference, candidate, weights=weight, smoothing_function=chencherry.method4) for weight in weights]


def translate_inputs():
    translation = []
    for input in inputs:
        tokens = input.split()
        sentence = Generator.naturalized_sentence(tokens)
        translation.append(sentence)
    return translation


def write_translations(translation):
    write_file(candidate_path, translation)

if __name__ == '__main__':
    translation = [sentence.split() for sentence in translate_inputs()]
    reference = [[sentence.split() for sentence in sentences] for sentences in reference]

    # Uncomment if you want to write translations in a text file
    # write_translations(translation)

    for id, sentence in enumerate(translation):
        scores = calculate_cummulative_ngram(reference[id], sentence)
        print(f'{sentence}\n{reference[id]}')
        print(f'\tCumulative 1-gram(BLUE-1): {scores[0]:.2f}\n\tCumulative 2-gram(BLUE-3): {scores[1]:.2f}\n'
              f'\tCumulative 3-gram(BLUE-2): {scores[2]:.2f}\n\tCumulative 4-gram(BLUE-4): {scores[3]:.2f}\n')
