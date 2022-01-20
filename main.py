import pandas as pd
import numpy as np
# import statsmodels.api as sm
# from scipy.stats.stats import pearsonr
from wals_estimator import WALSestimator
import statsmodels.api as sm
import seaborn as sb
import matplotlib.pyplot as mp


MODEL1_INDEP = ['const', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'TAX', 'PTRATIO', 'B', 'RM^2', 'NOX^2', 'ln(DIS)', 'ln(RAD)', 'ln(LSTAT)']
MODEL2_INDEP = ['const', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'AGE', 'TAX', 'PTRATIO', 'B', 'RM^2', 'ln(DIS)', 'ln(RAD)', 'ln(LSTAT)']
MODEL3_INDEP = ['const', 'ln(INDUS)', 'NOX', 'ln(DIS)', 'RAD', 'ln(LSTAT)']
MODEL4_INDEP = ['const', 'RM^2', 'TAX', 'PTRATIO', 'ln(LSTAT)', 'NOX^2']
MODEL5_INDEP = ['const', 'RM^2', 'TAX', 'PTRATIO', 'ln(LSTAT)']
MODEL6_INDEP = ['const', 'DIS', 'CRIM', 'INDUS']
MODEL7_INDEP = ['const', 'RM', 'PTRATIO', 'B', 'CRIM']
MODEL8_INDEP = ['const', 'CRIM', 'ZN', 'INDUS', 'NOX', 'AGE', 'TAX', 'PTRATIO', 'B', 'RM^2', 'NOX','NOX^2', 'ln(DIS)', 'ln(RAD)', 'ln(LSTAT)']

MODELS = [MODEL1_INDEP, MODEL2_INDEP, MODEL3_INDEP, MODEL4_INDEP, MODEL5_INDEP, MODEL6_INDEP, MODEL7_INDEP, MODEL8_INDEP]


def transformations(df):

    df = df.dropna()
    df[['RM^2', 'NOX^2']] = df.loc[:, ['RM', 'NOX']]**2
    df[['ln(DIS)', 'ln(RAD)', 'ln(LSTAT)', 'ln(INDUS)', 'ln(MEDV)']] = np.log(df.loc[:, ['DIS', 'RAD', 'LSTAT', 'INDUS', 'MEDV']])
    return df


def sub_regression(X, y):

    pred = sm.OLS(y, X).fit().predict()
    return pred


def prediction_matrix(X, y):

    prediction_matrix = []
    for model in MODELS:
        indep = X[model]
        prediction_matrix.append(sub_regression(indep, y))
    prediction_df = pd.DataFrame(np.transpose(prediction_matrix), columns=['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'])

    return prediction_df


def plot_heatmap(pred_df):

    mask = np.triu(np.ones_like(pred_df.corr()))

    # plotting a triangle correlation heatmap
    dataplot = sb.heatmap(pred_df.corr(), cmap="YlGnBu", annot=True, mask=mask, fmt='g')
    # displaying heatmap
    mp.show()


def main():

    data = pd.read_csv('HousingData.csv')
    data = transformations(data)

    X = data.iloc[:, 0:-1]
    X = sm.add_constant(X)
    y = data.iloc[:, -1]
    X1 = X.iloc[:, 0:4]
    X2 = X.iloc[:, 4:]

    #model = WALSestimator(y, X1, X2)
    #model.fit()
    #pred = model.predict(X)
    #print(model.b)
    #print(sm.OLS(y, X).fit().summary())

    prediction_df = prediction_matrix(X, y)
    plot_heatmap(prediction_df)


if __name__ == "__main__":
    main()
