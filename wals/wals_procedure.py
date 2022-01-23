import numpy as np
import statsmodels.api as sm
import scipy


def wals(X1, X2, y):

    n = len(y)
    k1 = X1.shape[1]
    k2 = X2.shape[1]
    k = k1 + k2
    q = 0.887630085544086
    alpha = 1-q
    c = np.log(2)

    """%% --- Step 2.a: Scaling X1 so that all diagonal elements of"""

    """%% --- Step 2.b: Scaling X2 so that all diagonal elements of"""

    # 2a
    d1 = np.diag(np.diag((X1.T@X1))**(-1/2))
    Z1 = X1@d1

    # 2b
    Z2d = X2.T@X2
    V1r = np.linalg.inv(Z1.T@Z1)
    VV12 = Z1.T@X2
    Z2d = Z2d - VV12.T@V1r@VV12
    d2 = np.diag(np.diag(Z2d)**(-1/2))
    Z2s = d2@Z2d@d2

    """%% --- Step 3: Semi-orthogonalization of Z2s"""

    Xi, T = np.linalg.eig(Z2s)
    Xi = np.diag(Xi)
    D2 = d2@T@np.diag(np.diag(Xi)**(-.5))
    Z2 = X2@D2

    """
    %% --- Step 4: OLS of unrestricted model"""

    Z = np.hstack((Z1, Z2))
    res = sm.OLS(y, Z).fit()
    resid = res.resid
    gamma_hat = res.params
    s2 = (resid.T@resid)/(n-k)
    s = s2**(1/2)
    gamma2_hat = gamma_hat[k1:]
    x = gamma2_hat/s

    m_post = np.zeros((k2, 1))
    v_post = np.zeros((k2, 1))
    delta = (1-alpha)/q

    for h in range(k2):
        xh = x[h]
        A0 = lambda gamma: (scipy.stats.norm.pdf(xh-gamma) + scipy.stats.norm.pdf(xh+gamma))*prior(gamma, alpha, c, delta, q)
        A1 = lambda gamma: ((xh - gamma)*scipy.stats.norm.pdf(xh-gamma) + (xh + gamma)*scipy.stats.norm.pdf(xh + gamma))*prior(gamma, alpha, c, delta, q)
        A2 = lambda gamma: (((xh - gamma)**2)*scipy.stats.norm.pdf(xh-gamma) + ((xh + gamma)**2)*scipy.stats.norm.pdf(xh + gamma))*prior(gamma, alpha, c, delta, q)
        int_A0, errA0 = scipy.integrate.quad(A0, 0, np.inf)
        int_A1, errA1 = scipy.integrate.quad(A1, 0, np.inf)
        int_A2, errA2 = scipy.integrate.quad(A2, 0, np.inf)
        psi1 = int_A1/int_A0
        psi2 = int_A2/int_A0
        m_post[h] = xh - psi1
        v_post[h] = psi2 - psi1**2

    # Large coefficients lead to integral of 0    
    """%% --- Step 6: WALS estimates """

    c2 = s*m_post
    c1 = V1r@Z1.T@(y.reshape((-1, 1))-Z2@c2)
    b1 = d1@c1
    b2 = D2@c2

    """%% --- Step 7: WALS precisions"""

    varc2 = s2*np.diag(v_post.T[0])
    varb2 = D2*varc2*D2
    Q = V1r@Z1.T@Z2
    varc1 = s2*V1r+Q@varc2@Q.T
    varb1 = d1*varc1*d1
    covc1c2 = -Q@varc2
    covb1b2 = d1@covc1c2@D2.T

    b = np.vstack((b1, b2))
    se = np.hstack((np.diag(varb1)**(.5), np.diag(varb2)**(.5)))
    V = np.hstack((varb1, covb1b2))
    return b, se, V


def prior(gamma, alpha, c, delta, q):

    return ((q*c**delta)/(2*np.exp(scipy.special.loggamma(delta))))*(np.abs(gamma)**(-alpha)*(np.exp(-c*(np.abs(gamma)**q))))
