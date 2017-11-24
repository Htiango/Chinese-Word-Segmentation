import numpy as np
from scipy import sparse
import time
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm

CON_NUM = 4
FEATURE_NUM = 5


def get_feature(path):
    """
    generate the corpus and the features
    :param path: the input training path
    :return:
    """
    start = time.time()
    read_start = time.time()
    lines = read_convert(path)
    read_end = time.time()
    print("Reading time is " + str(read_end - read_start))

    corpus_start = time.time()
    corpus_dict, index_dict = generate_dict(lines)
    corpus_end = time.time()
    print("Generate Corpus time is " + str(corpus_end - corpus_start))
    X_train_col = []
    X_train_row = []
    Y_train = []
    counter = 0
    feature_start = time.time()
    for line in lines:
        counter += 1
        space_index = get_space_index(line)
        X, Y = get_sparse_matrix(space_index, line, corpus_dict)
        if counter % 10000 == 0:
            feature_end = time.time()
            print("Finish line-" + str(counter) + ", spend time: " + str(feature_end - feature_start))
            feature_start = time.time()
        if X is None and Y is None:
            continue
        else:
            col_num = len(X)
            col_num_all = len(X_train_col)
            for i in range(col_num):
                X_train_row.append([i + col_num_all]*FEATURE_NUM)
            X_train_col.extend(X)
            Y_train.extend(Y)

    n_samples = len(X_train_col)
    n_features = FEATURE_NUM
    data = np.ones(n_samples * n_features, dtype=np.int64)

    row = np.reshape(np.array(X_train_row), n_samples * n_features)
    col = np.reshape(np.array(X_train_col), n_samples * n_features)

    X_train = sparse.csr_matrix((data, (row, col)), shape=(n_samples, len(corpus_dict)))
    Y_train = np.array(Y_train)

    print()
    np.save("../output/trainY", Y_train)
    save_sparse_csr("../output/trainX", X_train)
    save_obj("../output/corpus.pkl", corpus_dict)
    save_obj("../output/index.pkl", index_dict)
    end = time.time()
    print("Running time: " + str(end - start))


def save_obj(filename, items):
    with open(filename, 'wb') as f:
        pickle.dump(items, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                             shape=loader['shape'])


def get_sparse_matrix(space_index, line, corpus_dict):
    """
    get the sparse matrix's col and row of each lines
    :param space_index:
    :param line:
    :param corpus_dict:
    :return:
    """
    line_consecutive = line.replace(" ", "")
    X = None
    Y = None
    for i in range(len(line_consecutive) - CON_NUM + 1):
        s = line_consecutive[i: i + 4]
        label = 0
        if i + 1 in space_index:
            label = 1
        sparse_matrix_col = get_vector(s, corpus_dict)
        if X is None and Y is None:
            X = [sparse_matrix_col]
            Y = [label]
        else:
            X.append(sparse_matrix_col)
            Y.append(label)
    return X, Y


def get_vector(s, corpus_dict):
    """
    get a sparse matrix's col index
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
    # row = np.array([0, 0, 0, 0, 0])
    col = [index1, index2, index3, index4, index5]
    # data = np.array([1, 1, 1, 1, 1])
    # mtx = sparse.csr_matrix((data, (row, col)), shape=(1, len(corpus_dict)))
    return col


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
    return set(result)


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
            # line = line.strip('\n')
            line = '\t' + line
            lines.append(line)
    print("There are lines: " + str(len(lines)))
    return lines


def get_model(path):
    start = time.time()
    X_train = load_sparse_csr("../output/trainX.npz")
    Y_train = np.load("../output/trainY.npy")
    # corpus_dict = load_obj("../output/corpus.pkl")
    # index_dict = load_obj("../output/index.pkl")
    print("Load Successfully!")
    # read_start = time.time()
    # lines = read_convert(path)
    # read_end = time.time()

    # print("start training SVM Model.....")
    # train_start = time.time()
    # svmModel = svm.SVC()
    # svmModel.fit(X_train, Y_train)
    # train_end = time.time()
    # print("Get svm model, spend time: " + str(train_end - train_start))
    # save_obj("../output/svmModel.pkl", svmModel)

    print("start training LR Model.....")
    train_start = time.time()
    lrModel = LogisticRegression(penalty='l2')
    lrModel.fit(X_train, Y_train)
    train_end = time.time()
    print("Get LR model, spend time: " + str(train_end-train_start))
    save_obj("../output/LRModel.pkl", lrModel)

    end = time.time()
    print("Total time: " + str(end-start))


if __name__ == '__main__':
    path_train = '../data/training.txt'
    path_test = '../data/test.txt'
    # get_feature(path_train)
    get_model(path_test)
