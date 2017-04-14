import numpy as np

from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import csr_matrix, coo_matrix, linalg
import matplotlib.pyplot as plt

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'], dtype="F")

def run_nmf(n_components):
	model = NMF(n_components=110)
	model.fit(counts)
	return model

def gen_svd():
	U, s, VT = linalg.svds(counts, k=250)
	sigma = np.diag(s)
	save_sparse_csr("data/U",csr_matrix(U))
	save_sparse_csr("data/sigma",csr_matrix(sigma))
	save_sparse_csr("data/VT",csr_matrix(VT))

def gen_counts():
	train = np.loadtxt("txTripletsCounts.txt", dtype=np.int)
	M = int(max(train[:,1]))
	N = int(max(train[:,0]))

	counts = coo_matrix( (train[:,2], (train[:,0], train[:,1]) ))
	counts = counts.tocsr()
	save_sparse_csr("data/counts",counts)


def main():

	counts = load_sparse_csr("data/counts.npz")
	U = load_sparse_csr("data/U250.npz")
	sigma = load_sparse_csr("data/sigma250.npz")
	VT = load_sparse_csr("data/VT250.npz")

	test = np.loadtxt("data/testTriplets.txt", dtype=np.int)
	print "Reconstruct reduced rank matrix"
	recon =  U.dot(sigma.dot(VT))
	save_sparse_csr("data/counts250", recon)

	print "Plotting"
	for (i,row) in enumerate(test):
		n = recon[row[0], row[1]]
		color = 'ro' if row[2] == 0 else 'bo'
		plt.scatter(i, n, color)	

	plt.show()


if __name__ == "__main__":
	main()