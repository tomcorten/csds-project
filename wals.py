from re import sub
import pandas as pd
import numpy as np


def transformations(df):

    df = df.dropna()
    #df[['RM^2', 'NOX^2']] = df[['RM', 'NOX']]**2
    #df[['ln(DIS)', 'ln(RAD)', 'ln(LSTAT)', 'ln(INDUS)', 'ln(MEDV)']] = np.log(df[['DIS', 'RAD', 'LSTAT', 'INDUS', 'MEDV']])

    return df


def step2(X1, X2, y):

    
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

    #2a
    d1 = np.diag(np.diag(np.dot(X1.T, X2))**(-1/2))
    Z1 = np.dot(X1, d1)

    #2b
    Z2d = np.dot(X2.T, X2)
    V1r = np.linalg.inv(np.dot(Z1.T, Z1))
    VV12 = np.dot(Z1.T, X2)
    Z2d = Z2d - np.dot(VV12.T, np.dot(V1r, VV12))
    d2 = np.diag(np.diag(Z2d)**(-1/2))
    Z2d = np.dot(d2, np.dot(Z2d, d2))
    print(Z2d)


def main():

    data = pd.read_csv('HousingData.csv')
    data = transformations(data)

    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]

    X1 = X.iloc[:,0:4]
    X2 = X.iloc[:,4:]

    step2(X1.to_numpy(), X2.to_numpy(), y.to_numpy())

if __name__ == "__main__":
    main()
