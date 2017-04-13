import numpy as np
from scipy.sparse import csr_matrix


def main():
	train = np.loadtxt("txTripletsCounts.txt")
	test = np.loadtxt("testTriplets.txt")
	M = max(train[:,1])
	N = max(train[:,0])

	full_counts = np.zeros((N,M), dtype=np.int)

	for row in train:
		full_counts[row[0], row[1]] = row[2]

	counts = csr_matrix(full_counts)

if __name__ == "__main__":
	main()