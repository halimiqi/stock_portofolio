import numpy as np
import scipy
import pandas as pd
from scipy import optimize
def data_process():
    dfs = pd.read_excel("data/stock_price_return(1).xlsx", sheet_name="Stock Historical")
    return dfs

def calculate_mean_std(df):
    # choose the data
    dfs = df[1:192]
    #dfs = dfs.iloc[:,1:22]
    mean = dfs.mean()
    var = dfs.var()
    cov = dfs.cov()
    return mean, var, cov

def mini_fun(x, cov):
    results = 0.0
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            results += x[i]*x[j]*cov.iloc[i,j]
    return (1/2)*results

def constrain_fun_eq(x, mean, target_r):
    return sum(x*mean) - target_r

def calculate_potofolio(target_r, mean_r, cov):
    cons =({"type":"eq", "fun":constrain_fun_eq, "args":(mean_r,target_r)},
           {"type": "eq", "fun": lambda x:sum(x) -1}
           )
    bond = []
    for idx in range(0,len(mean_r)):
        bond.append((0,None))
    x_0 = (1/len(mean_r))*np.ones(len(mean_r))
    res = optimize.minimize(mini_fun,x_0, args = (cov),method = "SLSQP", constraints=cons, bounds=bond)
    return res

def calculate_prdict(x, y,df):
    rr_x = 1
    rr_y = 1
    array_x = []
    array_y = []
    for i in range(0,df.shape[0]):
        temp = df.iloc[i]
        rr_x = rr_x*sum((1+temp) * x)
        rr_y = rr_y*sum((1+temp) * y)
        array_x.append(sum(x*temp))
        array_y.append(sum(y*temp))
    return rr_x, rr_y, array_x, array_y

def getc1c2(m,n,df,line_index):
    b_1 = 0.5
    b_2 = 0.5
    score_list = []
    for i in range(0,21):
        R_1 = np.log(df.iloc[line_index-1,i+1] / df.iloc[line_index-n, i +1])
        R_2 =np.log((df.iloc[line_index-m,i+1] / df.iloc[line_index-1, i +1]))
        score = b_1*R_1 + b_2*R_2
        score_list.append(score)
    #scores = pd.Series(score_list,index=df.columns)
    sorted = np.array(score_list).argsort()
    return sorted[0:10]

def get_new_df(df, col_idx):
    new_df = df.iloc[:,col_idx]
    return
if __name__ == "__main__":
    df = data_process()
    dfs = df.iloc[:, 22:]
    mean, var, cov = calculate_mean_std(dfs)
    predict_df = df.iloc[192:,22:]
    final_mean_array_x = []
    final_mean_array_y = []
    final_var_array_x = []
    final_var_array_y = []
    final_target_list = np.arange(0,max(mean),0.005)
    for I in final_target_list:
        target_r = I
        res = calculate_potofolio(target_r,mean,cov)
        y = (1/21)*np.ones(21)
        rr_x, rr_y, array_x, array_y = calculate_prdict(res.x,y,predict_df)
        final_mean_array_x.append(np.mean(array_x))
        final_var_array_x.append(np.var(array_x))
        final_mean_array_y.append(np.mean(array_y))
        final_var_array_y.append(np.var(array_y))
        print("return %f"%(I))
    print("potofolio x:%f" % (rr_x))
    print("portofolio y:%f" % (rr_y))


    # for the question 2
    final_c_mean_array_x = []
    final_c_mean_array_y = []
    final_c_var_array_x = []
    final_c_var_array_y = []
    final_mn_set = []
    target_r = np.mean(mean)
    for m in range(3,10):
        for n in range(1,7):
            sorted = getc1c2(m,n,df,191)
            temp = sorted +1
            new_df = df.iloc[:, sorted+22]
            mean, var, cov = calculate_mean_std(new_df)
            predict_df = df.iloc[192:,sorted+22]
            #target_r = np.mean(mean)
            res = calculate_potofolio(target_r, mean, cov)
            y = (1 / 10) * np.ones(10)
            rr_x, rr_y, array_x, array_y = calculate_prdict(res.x, y, predict_df)
            final_c_mean_array_x.append(np.mean(array_x))
            final_c_var_array_x.append(np.var(array_x))
            final_c_mean_array_y.append(np.mean(array_y))
            final_c_var_array_y.append(np.var(array_y))
            final_mn_set.append([m,n])
            print("potofolio x:%f" % (rr_x))
            print("portofolio y:%f" % (rr_y))
