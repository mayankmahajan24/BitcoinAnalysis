import numpy as np

from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import csr_matrix, coo_matrix, linalg
import matplotlib.pyplot as plt
import itertools


U = None
sigma = None
VT = None
counts = None
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'], dtype="F")


def gen_nmf(A, n_components):
    nmf_model = NMF(n_components=n_components)
    W = nmf_model.fit_transform(A);
    H = nmf_model.components_;
    save_sparse_csr("data/nmf_W_"+int(n_components), csr_matrix(W))
    save_sparse_csr("data/nmf_H_"+int(n_components), csr_matrix(H))


def save_nmf_recon_mat(n_components):
    W = load_sparse_csr("data/nmf_W_"+int(n_components)+".npz")
    H = load_sparse_csr("data/nmf_H_"+int(n_components)+".npz")
    recon = W.dot(H)
    save_sparse_csr("data/counts_nmf_"+int(n_components), recon)
    return recon


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


def save_svd_recon_mat():
    U = load_sparse_csr("data/U250.npz")
    sigma = load_sparse_csr("data/sigma250.npz")
    VT = load_sparse_csr("data/VT250.npz")
    recon = U.dot(sigma.dot(VT))
    save_sparse_csr("data/counts250", recon)
    return recon

def load_svd_components():
    global U, sigma, VT
    U = load_sparse_csr("data/U250.npz")
    sigma = load_sparse_csr("data/sigma250.npz")
    VT = load_sparse_csr("data/VT250.npz")

def plot_recon_values_scatter_plot(counts,test):
    global U, sigma, VT

    i = 0
    first = U[:1000,:].dot(sigma.dot(VT[:,:1000]))
    for send, rec in itertools.product(range(1000), range(1000)):
        color = 'b' if counts[send,rec] == 0 else 'r'
        print send, rec
        plt.scatter(i, first[send,rec], c=color)
        i+=1

    '''
	for (i,row) in enumerate(test):
        #To access any (x,y) coordinate, just do U[x,:] * sigma * VT[:,y]
		#val = U[row[0], :].dot (sigma.dot(VT[:,row[1]]))[0,0].real
		color = 'b' if row[2] == 0 else 'r'
        print i
		plt.scatter(i, val, c=color, mec=color)	
    '''
    plt.yscale('log')
    plt.show()


def main():
    counts = load_sparse_csr("data/counts.npz")
    test = np.loadtxt("data/testTriplets.txt", dtype=np.int)
    
    print "Reconstruct reduced rank matrix"
    load_svd_components()

    print "Plotting"
    plot_recon_values_scatter_plot(counts,test)
    


if __name__ == "__main__":
    main()