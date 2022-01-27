import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sb
import matplotlib.pyplot as mp


def transformations(df):

    df = df.dropna()
    df = sm.add_constant(df)
    df[['RM^2', 'NOX^2']] = df.loc[:, ['RM', 'NOX']]**2
    df[['ln(DIS)', 'ln(RAD)', 'ln(LSTAT)', 'ln(INDUS)', 'ln(MEDV)']] = np.log(df.loc[:, ['DIS', 'RAD', 'LSTAT', 'INDUS', 'MEDV']])

    return df


def return_prediction(X, y):

    pred = sm.OLS(y, X).fit().predict()
    return pred


def prediction_matrix(X, y, MODELS, dev=False):

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
    print(df)


def vif_scores(X):

    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    print(vif_data)