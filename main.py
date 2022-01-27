import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from wals.wals_estimator import WALSestimator
import weighing_schemes
import utils


MODEL1_INDEP = ['const', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'TAX', 'PTRATIO', 'B', 'RM^2', 'NOX^2', 'ln(DIS)', 'ln(RAD)', 'ln(LSTAT)']
MODEL2_INDEP = ['const', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'AGE', 'TAX', 'PTRATIO', 'B', 'RM^2', 'ln(DIS)', 'ln(RAD)', 'ln(LSTAT)']
MODEL3_INDEP = ['const', 'ln(INDUS)', 'NOX', 'ln(DIS)', 'RAD', 'ln(LSTAT)']
MODEL4_INDEP = ['const', 'RM^2', 'TAX', 'PTRATIO', 'ln(LSTAT)', 'NOX^2']
MODEL5_INDEP = ['const', 'RM^2', 'TAX', 'PTRATIO', 'ln(LSTAT)']
MODEL6_INDEP = ['const', 'DIS', 'CRIM', 'INDUS']
MODEL7_INDEP = ['const', 'RM', 'PTRATIO', 'B', 'CRIM']
MODEL8_INDEP = ['const', 'CRIM', 'ZN', 'INDUS', 'NOX', 'AGE', 'TAX', 'PTRATIO', 'B', 'RM^2', 'NOX','NOX^2', 'ln(DIS)', 'ln(RAD)', 'ln(LSTAT)']
MODELS = [MODEL1_INDEP, MODEL2_INDEP, MODEL3_INDEP, MODEL4_INDEP, MODEL5_INDEP, MODEL6_INDEP, MODEL7_INDEP, MODEL8_INDEP]


def wals_evaluation(X_train, y_train, X_test, y_test, k1=4):

    x1 = X_train.iloc[:, 0:k1]
    x2 = X_train.iloc[:, k1:]
    model = WALSestimator(y_train, x1, x2)
    model.fit()
    model.predict(X_test)

    return model.rmse(y_test)


def evaluation_pipeline(X, y, method='wals'):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    if method == 'wals':
        mse = wals_evaluation(X_train, y_train, X_test, y_test)
        print(mse)


def main():

    data = pd.read_csv('HousingData.csv')
    data = utils.transformations(data)

    X = data.iloc[:, 0:-1]

    # vif_scores(X)
    y = data.iloc[:, -1]
    evaluation_pipeline(X, y)

    """# prediction_df = prediction_matrix(X, y)

    prediction_df = utils.prediction_matrix(X, y, MODELS, dev=False)
    # plot_heatmap(data)
    R = prediction_df.corr()
    ce = (weighing_schemes.ce_weighting_scheme(R))
    cs = (weighing_schemes.cs_weigting_scheme(R))

    #r_df = pd.DataFrame(data=np.vstack((ce, cs)).T, columns=['Cap-Eigenvalue', 'Cos-Square'])
    #print(r_df)

    print(ce)
    print(prediction_df)"""


if __name__ == "__main__":
    main()
