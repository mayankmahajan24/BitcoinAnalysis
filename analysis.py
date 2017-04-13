import numpy as np
from scipy.sparse import csr_matrix

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def main():
	train = np.loadtxt("txTripletsCounts.txt", dtype=np.int)
	test = np.loadtxt("testTriplets.txt", dtype=np.int)
	M = int(max(train[:,1]))
	N = int(max(train[:,0]))

	counts = csr_matrix((N+1,M+1), dtype=np.int)
	for row in train:
		counts[row[0], row[1]] = row[2]

	save_sparse_csr("counts",counts)


if __name__ == "__main__":
	main()