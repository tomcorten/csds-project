import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats.stats import pearsonr
from wals_procedure import wals


"""%% Example.m
%
% version: 1.0
% created: 18 December 2013
% authors: J.R. Magnus, G. De Luca
%
% Description: This Matlab code calls the 'wals.m' code which perform 
% weighted average least squares (WALS) estimation of linear regression 
% models with sperical errors. 
%
% Your output is written to the folder: "results".
% Our results, which should be the same, are given in the folder: "our-output".
%
% Comments and suggestions are welcome and should be addressed to 
% 'magnus@uvt.nl' and 'giuseppe.deluca@unipa.it' 
%% -------------------------------------------------------------
"""


def main():

    data = pd.read_excel('MATLAB\data\M1.xls')
    data = data.dropna()

    k = data.shape[1]
    k1 = 6
    k2 = k - k1
    y = data.iloc[:, 1]
    X1 = data.iloc[:, 2:k1+2]
    X2 = data.iloc[:, k1+2:k+2]
    wals(X1.to_numpy(), X2.to_numpy(), y.to_numpy())
    
if __name__ == "__main__":
    main()