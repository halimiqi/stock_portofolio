import numpy as np
import scipy
import pandas as pd

def data_process():
    dfs = pd.read_excel("data/stock_price_return(1).xlsx", sheetname="Stock Historical")
    return dfs

def calculate_mean_std(dfs):
    # choose the data
    dfs = dfs[0:192]
    dfs = dfs.iloc[:,1:22]
    mean = dfs.mean()
    var = dfs.var()
    cov = dfs.cov()
    return mean , var, cov

def calculate_potofolio():

    return

if __name__ == "__main__":
    df = data_process()
    calculate_mean_std(df)