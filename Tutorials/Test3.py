from nltk.translate.bleu_score import sentence_bleu
import warnings
warnings.filterwarnings("ignore")
from Application.NLP.Utilities import read_file, write_file

reference = [['WHAT IS YOUR NAME'.split(), 'WHATS YOUR NAME'.split()],
             ['I AM GOOD'.split(), 'I AM DOING GOOD'.split()],
             ['WHERE ARE YOU FROM'.split(), 'WHERE DO YOU COME FROM'.split()],
             ['I AM COOKING AN EGG'.split(), 'I COOK AN EGG'.split()],
             ['WHERE DO YOU LIVE'.split()],
             ['YOU ARE OKAY'.split(), 'YOU ARE DOING OKAY'.split()],
             ['MY NAME IS JOSH'.split()],
             ['GO TO THE OFFICE'.split(), 'GO TO OFFICE'.split()],
             ['WHAT ARE YOU STUDYING'.split(), 'WHAT DO YOU STUDY'.split()],
             ['YOU ARE A STUDENT'.split()]
             ]

candidate = ['WHAT IS YOUR NAME'.split(),
             'I AM GOOD'.split(),
             'WHERE ARE YOU FROM'.split(),
             'I COOK EGG'.split(),
             'WHERE DO YOU LIVE'.split(),
             'YOU ARE OKAY'.split(),
             'MY NAME IS JOSH'.split(),
             'GO TO OFFICE'.split(),
             'WHAT DO YOU STUDY'.split(),
             'YOU ARE STUDENT'.split()
             ]

for id, sentence in enumerate(candidate):
    score = sentence_bleu(reference[id], sentence)
    score_1 = sentence_bleu(reference[id], sentence, weights=(1, 0, 0, 0))
    score_2 = sentence_bleu(reference[id], sentence, weights=(0.5, 0.5, 0, 0))
    score_3 = sentence_bleu(reference[id], sentence, weights=(0.33, 0.33, 0.33, 0))
    score_4 = sentence_bleu(reference[id], sentence, weights=(0.25, 0.25, 0.25, 0.25))
    print(score)
    print(f'{sentence}\n{reference[id]}')
    print(f'\tCumulative 1-gram: {score_1:.2f}\n\tCumulative 2-gram: {score_2:.2f}\n'
          f'\tCumulative 3-gram: {score_3:.2f}\n\tCumulative 4-gram: {score_4:.2f}\n')

# print(f'Individual 4-gram: {sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))}')
# print(f'Without indicating: {sentence_bleu(reference, candidate)}')