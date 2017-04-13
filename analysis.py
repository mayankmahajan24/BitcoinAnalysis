import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, linalg

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'], dtype="F")

def main():
	'''
	train = np.loadtxt("txTripletsCounts.txt", dtype=np.int)
	M = int(max(train[:,1]))
	N = int(max(train[:,0]))

	counts = coo_matrix( (train[:,2], (train[:,0], train[:,1]) ))
	counts = counts.tocsr()
	save_sparse_csr("counts",counts)

	test = np.loadtxt("testTriplets.txt", dtype=np.int)
	'''

	counts = load_sparse_csr("counts.npz")
	U, s, VT = linalg.svds(counts, k=1000)
	sigma = np.diag(s)
	save_sparse_csr("U",csr_matrix(U))
	save_sparse_csr("sigma",csr_matrix(sigma))
	save_sparse_csr("VT",csr_matrix(VT))

if __name__ == "__main__":
	main()