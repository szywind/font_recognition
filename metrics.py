import numpy as np
from sklearn.metrics import f1_score

def compute_accuracy(y_true, y_pred, do_search_thresh=False):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    best_thresh = 0.5
    if do_search_thresh:
        best_f1 = 0
        for thresh in np.arange(0, 1, 0.01):
            pred = y_pred.ravel() > thresh
            f1 = f1_score(y_true, pred)
            print(thresh, f1, best_thresh)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

    pred = y_pred.ravel() > best_thresh
    # print(pred.shape, y_true.shape)
    # print(pred == y_true)

    # wrong_list = (pred != y_true).squeeze()

    return np.mean(pred == y_true), best_thresh # , wrong_list