import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def transformations(df):

    df[['RM^2', 'NOX^2']] = df[['RM', 'NOX']]**2
    df[['ln(DIS)', 'ln(RAD)', 'ln(LSTAT)', 'ln(INDUS)', 'ln(MEDV)']] = np.log(df[['DIS', 'RAD', 'LSTAT', 'INDUS', 'MEDV']])

    return df


def main():

    data = pd.read_csv('HousingData.csv')

    data = transformations(data)
    print(data)

    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    y = np.log(y)

    #reg1 = LinearRegression().fit(X, y)
    #reg2 = LinearRegression().fit(X.drop(['B', 'C'], axis=1), y)



if __name__ == "__main__":
    main()
