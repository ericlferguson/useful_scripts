'''
Positional arguments -
 [1] path to confusion matrix
 [2] path to output
'''
import sys
import numpy as np

def compute_metrics(tp, fn, fp):
    recall =  tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

# load confusion matrix. assumes background is the [0]th row
conf_mat = np.genfromtxt(sys.argv[1], delimiter=',')
for n in range(conf_mat.shape[1]):
    mask = np.ones(conf_mat.shape[1])
    mask[n] = 0
    precision, recall, f1 = compute_metrics(tp=conf_mat[n, n],
                                            fn=np.sum(conf_mat[n] * mask),
                                            fp=np.sum(conf_mat[:, n] * mask))
    print("Class,%s\nPrecision,%.4f\nRecall,%.4f\nF1,%.4f\n" % (n, precision, recall, f1))
