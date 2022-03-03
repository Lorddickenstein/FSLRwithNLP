#####################################################################
# Author: Jerson Destacamento, Joshua Cruzat, Rocella Legaspi       #
# Date: October-December                                            #
# Program Title: Utilities.py                                       #
# Description: Contains all the functions required in file handling #
# General System Design: Utility                                    #
# Data structures, Algorithms, Controls: Lists, Dictionaries,       #
#                 File Handling, Try Except                         #
# Requirements: None                                                #
#####################################################################

import os

def read_dictionary(name='None', dict_format=False) -> list:
    """ Returns a list containing all the recognizable words by the program or a
        dictionary if dict_format is True. Reads a file such as the dictionary.txt
        and puts all its content on a variable.

        Args:
            name: String or Directory. The name of the file or the directory of where the file
                is to be found. Defaults to None, which means an empty file or file is not existing.
            dict_format: Boolean. A boolean mode of reading from a file if the caller wants it to be
                a dictionary format or a list. Defaults to False.

        Returns:
            dictionary: Dictionary or list. In the form of [word, tag] based on what format the user chooses.

        Raises:
            FileNotFoundError: if the file is not found or when the current working directory is not on the
                location of the file.
    """
    dictionary = []
    word_list = []
    f = None
    if name:
        try:
            f = open(name, "r")
            for line in f:
                word, tag = line.split('\t\t')
                word = word.upper()
                dictionary.append([word, tag.strip()])
                word_list.append(word)
        except FileNotFoundError as exc:
            dictionary = None
            word_list = None
            print(exc)
        finally:
            if f is not None: f.close()

    return dictionary if dict_format else list(set(word_list))


def read_file(name=None, mode='input'):
    content = []
    f = None
    if name:
        try:
            f = open(name, 'r')
            for line in f:
                if mode == 'input':
                    content.append(line.strip())
                elif mode == 'reference':
                    line = line.strip()
                    line = line.split(', ')
                    content.append(line)
        except FileNotFoundError as exc:
            print(exc)
        finally:
            if f is not None: f.close()
        return content


def write_file(name=None, content=[]):
    f = None
    if name:
        f = open(name, 'w')
        for line in content:
            line = ' '.join(line)
            f.write(line + '\n')
        f.close()


if __name__ == '__main__':
    dictionary = read_dictionary('dictionary.txt', dict_format=True)
    print(dictionary)

    # persons = read_dictionary('persons.txt', dict_format=True)
    # print(persons)
