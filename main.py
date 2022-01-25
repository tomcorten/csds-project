import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sb
import matplotlib.pyplot as mp
from wals.wals_estimator import WALSestimator
import weighing_schemes


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


def return_prediction(X, y):

    pred = sm.OLS(y, X).fit().predict()
    return pred


def prediction_matrix(X, y, dev=False):

    prediction_matrix = []
    for model in MODELS:
        indep = X[model]
        prediction_matrix.append(return_prediction(indep, y))

    prediction_df = pd.DataFrame(np.transpose(prediction_matrix), columns=['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'])
    if dev:
        prediction_df = prediction_df.sub(prediction_df.mean(axis=1), axis=0)
    return prediction_df


def plot_heatmap(pred_df):

    mask = np.triu(np.ones_like(pred_df.corr()))
    # plotting a triangle correlation heatmap
    sb.heatmap(pred_df.corr(), cmap="YlGnBu", annot=True, mask=mask, fmt='g')
    # displaying heatmap
    mp.show()


def to_output(data, model):

    estimates = dict(zip(data.columns.values[:-1], model.b.T[0]))
    df = pd.DataFrame(data=estimates, index=[0])
    df = (df.T)
    df.columns = ['Coefficient value']


def main():

    data = pd.read_csv('HousingData.csv')
    data = transformations(data)
    data = sm.add_constant(data)

    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    x1 = X.iloc[:, 0:4]
    x2 = X.iloc[:, 4:]

    model = WALSestimator(y, x1, x2)
    model.fit()
    to_output(data, model)

    # prediction_df = prediction_matrix(X, y)

    avg_prediction_df = prediction_matrix(X, y, dev=True)
    # plot_heatmap(data)
    R = avg_prediction_df.corr()
    ce = (weighing_schemes.ce_weighting_scheme(R))
    cs = (weighing_schemes.cs_weigting_scheme(R))

    r_df = pd.DataFrame(data=np.vstack((ce, cs)).T, columns=['Cap-Eigenvalue', 'Cos-Square'])
    print(r_df)


if __name__ == "__main__":
    main()
