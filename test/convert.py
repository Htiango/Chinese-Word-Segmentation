"""
Test reading and converting.
"""

def convert(path):
    """
    In this convert function, I just ignore the UnicodeDecodeError
    :param path: the path of the input file
    :return:
    """
    encoding = 'big5hkscs'
    lines = []
    with open(path, encoding=encoding, errors='ignore') as f:
        for line in f:
            lines.append(line)
    print(len(lines))


if __name__ == '__main__':
    path = '../data/training.txt'
    convert(path)
