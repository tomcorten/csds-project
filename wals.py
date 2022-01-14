from re import sub
import pandas as pd
import numpy as np

def main():

    data = pd.read_csv('HousingData.csv')
    data = transformations(data)

    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]

    pred1 = sub_regression(X, y, to_drop=['RM', 'DIS', 'RAD', 'ln(INDUS)', 'NOX', 'MEDV'])
    #pred2 = sub_regression(X, y, to_drop=['RM', 'DIS', 'RAD', 'ln(INDUS)', 'NOX^2'])
    pred6 = sub_regression(X, y, to_drop=['RM', 'RM^2', 'AGE', 'ln(DIS)', 'RAD', 'ln(RAD)', 'TAX', 'PTRATIO', 'B', 'ln(LSTAT)', 'ZN', 'ln(INDUS)', 'CHAS', 'NOX', 'NOX^2', 'MEDV'])
    print(np.corrcoef(pred1, pred6))


if __name__ == "__main__":
    main()
