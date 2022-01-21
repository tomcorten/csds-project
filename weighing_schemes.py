import numpy as np


def ce_weighting_scheme(R):

    R = R.to_numpy()
    # R = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1]]
    Xi, T = np.linalg.eig(R)
    eig = np.diag(Xi)
    eig_star = np.copy(eig)
    eig_star[np.where((eig_star > 1))] = 1
    eig_star = np.round(eig_star)
    R_star = T@eig_star@T.T
    return np.diag(R_star)/sum(np.diag(R_star))
