import os

def read_file(name='None', dict_format=False) -> list:
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


if __name__ == '__main__':
    dictionary = read_file('dictionary.txt', dict_format=True)
    print(dictionary)

    # persons = read_file('persons.txt', dict_format=True)
    # print(persons)
