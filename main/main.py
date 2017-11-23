

def get_seg(path):
    lines = read_convert(path)
    corpus = generate_dict(lines)



def generate_dict(lines):
    """
    generate the whole corpus
    :param lines: input lines
    :return: the corpus, a dict, key is the 2-gram or 1-gram word, value is its index from 0-n
    """
    corpus = {}
    corpus_index = 0
    for line in lines:
        line_consecutive = line.replace(" ", "")
        for i in range(len(line_consecutive)):
            character = line_consecutive[i]
            word = line_consecutive[i:i + 2]
            if character not in corpus:
                corpus[character] = corpus_index
                corpus_index += 1
            if word not in corpus:
                corpus[word] = corpus_index
                corpus_index += 1
    print(len(corpus))
    return corpus



def read_convert(path):
    """
    In this convert function, I just ignore the UnicodeDecodeError
    :param path: the path of the input file
    :return: the list of lines
    """
    encoding = 'big5hkscs'
    lines = []
    with open(path, encoding=encoding, errors='ignore') as f:
        for line in f:
            lines.append(line)
    print(len(lines))
    return lines


if __name__ == '__main__':
    path = '../data/training.txt'
    get_seg(path)