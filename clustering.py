import numpy as np

from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, coo_matrix, linalg
from scipy.stats import mode
import matplotlib.pyplot as plt
import itertools
import matplotlib

import random

import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score

ignore_colms = ['sender', 'receiver', 'transaction']
label_colm = 'transaction'


def load_features_file(convert_to_bin_trans, svd_k, nmf_n, unq_senders, unq_receivers):
    filename_postfix = '_bin{}_svd{}_nmf{}_snd{}_rcv{}.csv'.format(str(convert_to_bin_trans), str(svd_k), str(nmf_n), str(unq_senders), str(unq_receivers))
    ftrain = pd.read_csv('features/train' + filename_postfix, index_col=0)
    ftest = pd.read_csv('features/test' + filename_postfix, index_col=0)
    return ftrain, ftest


def get_train_test_matricies(traindf, testdf):
    train_X = traindf.drop(ignore_colms, axis=1).as_matrix()
    train_Y = traindf[label_colm].values
    
    test_X = testdf.drop(ignore_colms, axis=1).as_matrix()
    test_true_Y = testdf[label_colm].values
    return train_X, train_Y, test_X, test_true_Y

def get_guess(cluster, kmeans, train_Y):
	inds = np.where(kmeans.labels_ == cluster)[0]
	return mode(train_Y[inds])[0][0]

def plot_test_roc(pred, label, filename=None, threshold=None):
    
    fpr, tpr, thresholds = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc
    matplotlib.rcParams['figure.figsize'] = (10, 10)
    plt.plot(fpr, tpr, color='magenta', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if filename:
        plt.savefig(filename)
    plt.show()

def main():
	# Load DFs
	bin_trans = True
	svd_k = 50
	nmf_n = 12
	unq_senders = 250
	unq_receivers = 100

	traindf, testdf = load_features_file(bin_trans, svd_k, nmf_n, unq_senders, unq_receivers)
	train_X, train_Y, test_X, test_true_Y = get_train_test_matricies(traindf, testdf)

	for num_clusters in [1, 5, 50, 100, 250, 500, 1000, 5000, 10000]:
		#print "Training"
		kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(train_X)
		cluster_preds = kmeans.predict(test_X)
		#print "Computing guesses"
		guess_preds = np.array([get_guess(cl, kmeans, train_Y) for cl in cluster_preds])
		#print guess_preds
		#print len(guess_preds)
		print len(np.where(guess_preds == 1)[0])
		fpr, tpr, thresholds = roc_curve(test_true_Y, guess_preds)
		roc_auc = auc(fpr, tpr)
		print "Number of clusters: ", num_clusters, "Area: ", roc_auc
		#print len(np.where(test_true_Y == 1)[0])

	#plot_test_roc(guess_preds, test_true_Y)



if __name__ == "__main__":
	main()