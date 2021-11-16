import os

def read_file(name='None') -> list:
    content = []
    f = None
    if name:
        try:
            f = open(name, "r")
            for line in f:
                word, tag = line.split('\t\t')
                content.append([word, tag.strip()])
        except FileNotFoundError as exc:
            content = None
            print(exc)
        finally:
            if f is not None: f.close()

    return content


if __name__ == '__main__':
    # dictionary = read_file('dictionary.txt')
    # print(dictionary)

    persons = read_file('persons.txt')
    print(persons)
