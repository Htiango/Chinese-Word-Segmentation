import numpy as np
from scipy import sparse
import time

CON_NUM = 4


def get_seg(path):
    start = time.time()
    read_start = time.time()
    lines = read_convert(path)
    read_end = time.time()
    print("Reading time is " + str(read_end - read_start))

    corpus_start = time.time()
    corpus_dict, index_dict = generate_dict(lines)
    corpus_end = time.time()
    print("Generate Corpus time is " + str(corpus_end - corpus_start))
    X_train = None
    Y_train = None
    counter = 0
    feature_start = time.time()
    for line in lines:
        counter += 1
        space_index = get_space_index(line)
        X, Y = get_sparse_matrix(space_index, line, corpus_dict)
        if X is None and Y is None:
            continue
        if X_train is None and Y_train is None:
            X_train = X
            Y_train = Y
        else:
            X_train = sparse.vstack([X_train, X])
            Y_train = np.append(Y_train, Y)
        if counter % 1000 == 0:
            feature_end = time.time()
            print("Finish line-" + str(counter) + ", spend time: " + str(feature_end - feature_start))
            feature_start = time.time()

    print()
    np.save("../output/trainY", Y_train)
    save_sparse_csr("../output/trainX", X_train)
    end = time.time()
    print("Running time: " + str(end - start))


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def get_sparse_matrix(space_index, line, corpus_dict):
    """
    get the sparse matrix of each lines
    :param space_index:
    :param line:
    :param corpus_dict:
    :return:
    """
    line_consecutive = line.replace(" ", "")
    X = None
    Y = None
    for i in range(len(line_consecutive) - CON_NUM + 1):
        s = line_consecutive[i: i+4]
        label = 0
        if i+1 in space_index:
            label = 1
        label_arr = np.array([label])
        sparse_matrix =get_vector(s, corpus_dict)
        if X is None and Y is None:
            X = sparse_matrix
            Y = label_arr
        else:
            X = sparse.vstack([X, sparse_matrix])
            Y = np.append(Y, label_arr)
    return X, Y


def get_vector(s, corpus_dict):
    """
    get a sparse matrix
    :param s:
    :param corpus_dict:
    :return: a sparse matrix array-like
    """
    if len(s) != 4:
        print("Error in length!")
        return False
    v1 = s[0:2]
    index1 = corpus_dict[v1]
    v2 = s[1]
    index2 = corpus_dict[v2]
    v3 = s[1:3]
    index3 = corpus_dict[v3]
    v4 = s[2]
    index4 = corpus_dict[v4]
    v5 = s[2:4]
    index5 = corpus_dict[v5]
    row = np.array([0, 0, 0, 0, 0])
    col = np.array([index1, index2, index3, index4, index5])
    data = np.array([1, 1, 1, 1, 1])
    mtx = sparse.csr_matrix((data, (row, col)), shape=(1, len(corpus_dict)))
    return mtx


def get_space_index(s):
    """
    get the space index in each line
    :param s:
    :return:
    """
    ls = s.split("  ")
    result = [-1]
    for word in ls:
        index = result[-1] + len(word)
        result.append(index)
    result.pop(-1)
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
    print("corpus length is " + str(len(corpus_dict)))
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
    print("There are lines: " + str(len(lines)))
    return lines


if __name__ == '__main__':
    path = '../data/training.txt'
    get_seg(path)