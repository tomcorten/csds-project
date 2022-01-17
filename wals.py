import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
import scipy


def transformations(df):

    df = df.dropna()
    # df[['RM^2', 'NOX^2']] = df[['RM', 'NOX']]**2
    # df[['ln(DIS)', 'ln(RAD)', 'ln(LSTAT)', 'ln(INDUS)', 'ln(MEDV)']] = np.log(df[['DIS', 'RAD', 'LSTAT', 'INDUS', 'MEDV']])

    return df


def wals(X1, X2, y):

    n = len(y)
    k1 = X1.shape[1]
    k2 = X2.shape[1]
    k = k1 + k2

    q = 0.887630085544086
    alpha = 1-q
    c = np.log(2)

    """%% --- Step 2.a: Scaling X1 so that all diagonal elements of 
    %%     (X1*Delta1)'X1*Delta1 are all one
    d1 = diag(X1'*X1).^(-1/2);
    Delta1 = diag(d1);
    Z1 = X1 * Delta1;"""

    """%% --- Step 2.b: Scaling X2 so that all diagonal elements of
    %%     (X2*Delta2)'M1*X2*Delta2 are all one
    Z2d     = X2'*X2;
    V1r     = inv(Z1'*Z1);
    VV12    = Z1'*X2;
    Z2d     = Z2d - VV12' * V1r * VV12;
    d2      = diag(Z2d).^(-1/2);
    Delta2  = diag(d2);
    Z2s     = Delta2 * Z2d * Delta2;"""

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

    """%% --- Step 3: Semi-orthogonalization of Z2s
    [T, Xi]     = eig(Z2s);
    order       = max(size(Xi));
    eigv        = diag(Xi);
    tol         = eps; 
    rank        = sum(eigv>tol);
    if (rank ~= order) 
        exitflag = 4;
    end

    %% --- If Xi is positive definite, proceed
    if (exitflag ~= 4)

        %% --- Set up Z2 so that Z2'*M1*Z2 = I
        D2=Delta2 * T * diag(diag(Xi).^(-.5));
        Z2=X2 * D2;
        
        """

    T, Xi = np.linalg.eig(Z2s)
    order = max(Xi.shape)
    tol = sys.float_info.epsilon
    eigv = np.diag(Xi)
    rank = sum(eigv > tol)

    D2 = d2@T@np.diag(np.diag(Xi)**(-1/2))
    Z2 = X2*D2

    """
    %% --- Step 4: OLS of unrestricted model
    Z                      = [Z1 Z2];
    [gamma_hat, ci, resid] = regress(y, Z);
    if (sigknown==0) 
            s2             = resid' * resid / (n-k);
            s              = s2^0.5;
    else
            s              =sigknown;
            s2             =s^2;
    end
    gamma2_hat             = gamma_hat(k1+1:k);
    x                      = gamma2_hat / s;"""

    Z = np.hstack((Z1, Z2))

    # If Xi is not positive definite, what to do?
    # Z = Z[:, ~np.all(np.isnan(Z), axis=0)]

    res = sm.OLS(y, Z).fit()
    resid = res.resid
    ci = res.conf_int()
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
        pass

    def prior(gamma):
        return lambda gamma: ((q*c**delta)/(2*np.exp(scipy.special.loggamma(delta))))*(np.abs(gamma)**(-alpha)*(np.exp(-c*(np.abs(gamma)**q))))


    def A0(gamma, xh):
        return lambda gamma: scipy.stats.norm(xh-gamma) + scipy.stats.norm(xh + gamma)*prior(gamma)


    def A1(gamma, xh):
        return lambda gamma: (xh - gamma)*scipy.stats.norm(xh-gamma) + (xh + gamma )*scipy.stats.norm(xh + gamma)*prior(gamma)


    def A2(gamma, xh):
        return lambda gamma: (xh - gamma)**2*scipy.stats.norm(xh-gamma) + (xh + gamma )**2*scipy.stats.norm(xh + gamma)*prior(gamma)   

    """%% --- Step 5: Compute the mean and variance of the posterior 
                m_post  = zeros(k2,1);    
                v_post  = zeros(k2,1);
                delta=(1-alpha)./ q;
                Prior= @(gamma) ((q .* c.^delta) ./ (2 .* exp(gammaln(delta)))) .* abs(gamma).^(-alpha) .* (exp(-c.*(abs(gamma).^q)));    
                for h=1:k2
                    xh=x(h);
                    A0=@(gamma) (                 normpdf(xh-gamma) +                  normpdf(xh+gamma)).*Prior(gamma);
                    A1=@(gamma) ( (xh-gamma)    .*normpdf(xh-gamma) +  (xh+gamma).*    normpdf(xh+gamma)).*Prior(gamma);
                    A2=@(gamma) (((xh-gamma).^2).*normpdf(xh-gamma) + ((xh+gamma).^2).*normpdf(xh+gamma)).*Prior(gamma);
                    int_A0 = quadgk(A0,0,inf);
                    int_A1 = quadgk(A1,0,inf);
                    int_A2 = quadgk(A2,0,inf);
                    psi1 = int_A1/int_A0;
                    psi2 = int_A2/int_A0;
                    m_post(h) = xh - psi1;                                    
                    v_post(h) = psi2 - psi1^2;
                end
            end
        else
            pmdata  = xlsread([p.Results.postmoments]);
            xtab=pmdata(:,1);           
            mtab=pmdata(:,2);
            vtab=pmdata(:,3);
            for h=1:k2
                signxh=sign(x(h));
                xh=abs(x(h));
                if xh<100 
                    xhl1=floor(xh*100)/100;
                    whl1=1-(floor(xh*10000)/10000-xhl1).*100;
                    xhl1_ind=find(xtab==xhl1);
                    m_post(h) =signxh .* (whl1 .* mtab(xhl1_ind) + (1-whl1) .* mtab(xhl1_ind+1));
                    v_post(h) =whl1 .* vtab(xhl1_ind) + (1-whl1) .* vtab(xhl1_ind+1);
                else
                    m_post(h)=signxh .* mtab(10001);
                    v_post(h)=vtab(10001);
                end
            end
        end"""


def main():

    data = pd.read_csv('HousingData.csv')
    data = transformations(data)

    X = data.iloc[:, 0:6]
    y = data.iloc[:, -1]

    X1 = X.iloc[:, 0:4]
    X2 = X.iloc[:, 4:]
  
    wals(X1.to_numpy(), X2.to_numpy(), y.to_numpy())
    #print(sm.OLS(y, X).fit().summary())

if __name__ == "__main__":
    main()
