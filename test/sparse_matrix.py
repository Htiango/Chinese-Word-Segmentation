from scipy import sparse
import numpy as np

# mtx = sparse.csr_matrix((3, 4), dtype=np.int8)
# mtx.todense()
# print(mtx)

row = np.array([0, 0, 0, 0, 0])
col = np.array([0, 10, 20, 30, 40])
data = np.array([1, 1, 1, 1, 1])
mtx = sparse.csr_matrix((data, (row, col)), shape=(1, 50))
mtx2 = np.append(mtx, mtx)


print(mtx.toarray())