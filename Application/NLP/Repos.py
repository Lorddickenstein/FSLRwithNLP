import os


def read_file(name='None') -> tuple:
    dictionary = ()
    if name:
        try:
            f = open(name, "r")
            for line in f:
                print(line.strip('\n'))
                word, tag = line.split('\t\t')
                dictionary.append((word, tag.strip('\n')))
        except FileNotFoundError as exc:
            dictionary = None
            print(exc)
        finally:
            if f is not None: f.close()

    return dictionary

if __name__ == '__main__':
    dictionary = read_file('dictionary.txt')