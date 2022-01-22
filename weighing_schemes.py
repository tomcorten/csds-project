import numpy as np


def ce_weighting_scheme(R):

    R = R.to_numpy()
    Xi, T = np.linalg.eig(R)
    eig = np.diag(Xi)
    eig_star = np.copy(eig)
    eig_star = np.round(eig_star)
    eig_star[np.where((eig_star > 1))] = 1
    R_star = T@eig_star@T.T

    return np.diag(R_star)/sum(np.diag(R_star))


def cs_weigting_scheme(R):

    D = np.identity(len(R))
    gamma, Q = np.linalg.eig(D@R@D)
    E = np.linalg.inv(D)@Q@gamma**(1/2)@Q.T
    D_1 = np.identity(len(R))@np.diag(E)

    return
