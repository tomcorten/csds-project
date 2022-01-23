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
    eps = 8*[1]
    toll = 1e-7

    while all(np.abs(x) > toll for x in eps):
        gamma, Q = np.linalg.eig(D@R@D)
        E = np.linalg.inv(D)@Q@np.diag(gamma)**(1/2)@Q.T
        D_1 = np.diag(np.diag(E))
        eps = np.diag(D_1-D)
        D = D_1

    d_star = np.diag(D)
    return (d_star**2/sum(d_star**2))
