

def get_seg(path):
    lines = read_convert(path)
    corpus_dict, index_dict = generate_dict(lines)
    for line in lines:
        space_index = get_space_index(line)



def get_space_index(s):
    ls = s.split("  ")
    result = [-1]
    for word in ls:
        index = result[-1] + len(word)
        result.append(index)
    result.pop(0)
    return result


def generate_dict(lines):
    """
    generate the whole corpus
    :param lines: input lines
    :return: 2 dicts: one whose key is the 2-gram or 1-gram word, value is its index from 0-n,
            the other whose ksy is the index and value is the word
    """
    corpus_dict = {}
    index_dict = {}
    corpus_index = 0
    for line in lines:
        line_consecutive = line.replace(" ", "")
        for i in range(len(line_consecutive)):
            character = line_consecutive[i]
            word = line_consecutive[i:i + 2]
            if character not in corpus_dict:
                corpus_dict[character] = corpus_index
                index_dict[corpus_index] = character
                corpus_index += 1
            if word not in corpus_dict:
                corpus_dict[word] = corpus_index
                index_dict[corpus_index] = word
                corpus_index += 1
    print(len(corpus_dict))
    return corpus_dict, index_dict



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
            line = line.strip('\n')
            lines.append(line)
    print(len(lines))
    return lines


if __name__ == '__main__':
    path = '../data/training.txt'
    get_seg(path)