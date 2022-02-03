import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from wals.wals_estimator import WALSestimator
import weighing_schemes
import utils


def wals_evaluation(X_train, y_train, X_test, y_test, k1=1):

    x1 = X_train[['const', 'ln(LSTAT)']]
    x2 = X_train.drop(x1.columns, axis=1)
    model = WALSestimator(y_train, x1, x2)
    model.fit()
    model.predict(X_test[['const', 'ln(LSTAT)']], X_test.drop(x1.columns, axis=1))
    return model.mse(y_test)


def gw_evaluation(X_train, y_train, X_test, y_test, method, dev):

    R, params = utils.prediction_matrix(X_train, y_train)
    prediction_df = R.copy()
    if dev:
        R = utils.get_deviations(prediction_df)
    R_corr = R.corr()
    R_corr.to_csv('schemes.csv')
    if method == 'ce':
        scheme = (weighing_schemes.ce_weighting_scheme(R_corr))
    else:
        scheme = (weighing_schemes.cs_weigting_scheme(R_corr))
    predictions = [X_test[p.index]@p for p in params]
    test_predictions = pd.DataFrame(np.transpose(predictions), columns=prediction_df.columns+'_pred')
    weighted_test_predictions = test_predictions@scheme
    mse = metrics.mean_squared_error(y_test, weighted_test_predictions)
    return mse


def evaluation_pipeline(X, y, method, dev):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if method == 'wals':
        mse = wals_evaluation(X_train, y_train, X_test, y_test)
    else:
        mse = gw_evaluation(X_train, y_train, X_test, y_test, method, dev)
    print(f'mean-squared-error for method {method}: {round(mse,3)} Deviations instead of predictions: {dev}')


def main():

    data = pd.read_csv('HousingData.csv')
    X, y = utils.preprocess(data)

    evaluation_pipeline(X, y, method='wals', dev=True)
    evaluation_pipeline(X, y, method='ce', dev=False)
    evaluation_pipeline(X, y, method='ce', dev=True)
    evaluation_pipeline(X, y, method='cs', dev=False)
    evaluation_pipeline(X, y, method='cs', dev=True)


if __name__ == "__main__":
    main()
