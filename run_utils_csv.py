import random
import sys, os, yaml
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from scipy import interpolate
from pathlib import Path
from econml.dr import LinearDRLearner, ForestDRLearner

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBRegressor, XGBClassifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_project_path():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    return str(path.parent.absolute())


def load_yaml(path_relative):
    return yaml.safe_load(open(get_project_path() + path_relative + ".yaml", 'r'))

def load_all_yaml_from_directory(path_relative):
    path = get_project_path() + path_relative
    files = os.listdir(path)
    configs = []
    for file in files:
        if file.endswith(".yaml"):
            configs.append(load_yaml(path_relative + file[:-5]))
    return configs

def save_yaml(path_relative, file):
    with open(get_project_path() + path_relative + ".yaml", 'w') as outfile:
        yaml.dump(file, outfile, default_flow_style=False)

def estimate_e(X, A, model_e=None):
    '''
        Estimate propensity score using a regularized model_e
    '''
    if model_e is None:
        model_e = XGBClassifier(
            n_estimators=100,
            max_depth=2,  
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
    )
    e = model_e.fit(X, A).predict_proba(X)[:, 1]
    return e.reshape(-1, 1)


def estimate_mu(X, A, y, model_y=None):
    '''
        Estimate response function using a regularized model_y
    '''
    train_data = np.concatenate((X, A), axis=1)
    if model_y is None:
        model_y = XGBRegressor(
            n_estimators=200,
            max_depth=3, 
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
    )
    mu = model_y.fit(train_data, y.reshape(-1, 1))
    
    test_0 = np.concatenate((X, np.zeros_like(A)), axis=1)
    test_1 = np.concatenate((X, np.ones_like(A)), axis=1)
    mu0 = mu.predict(test_0)
    mu1 = mu.predict(test_1)

    return mu0, mu1


def get_estimates(dataset_train, dataset_val, delta, significance_level = 0.05):
    '''Param setting'''
    alpha = significance_level
    z_alpha = norm.ppf(1 - alpha/2)
    Y_train = np.array(dataset_train['y']).reshape(-1, 1)
    A_train = np.array(dataset_train['A']).reshape(-1, 1)
    X_train = np.array(dataset_train['X']).reshape(-1, 1)
        
    '''IPW Implementation test'''
    e = estimate_e(X_train, A_train.flatten())
    mu0, mu1 = estimate_mu(X_train, A_train, Y_train)
    
    aipw = (A_train * Y_train / e - (1 - A_train) * Y_train / (1 - e)) - \
            ((A_train - e) / e * (1 - e)) * ((1-e) * mu1 + e * mu0)
    ate_est_aipw = np.mean(aipw)
    ate_ci_aipw = (ate_est_aipw - z_alpha * np.sqrt(np.var(aipw)/n), ate_est_aipw + z_alpha * np.sqrt(np.var(aipw)/n))

    print("aipw", aipw.shape) 
    print("var_aipw", np.var(aipw))
    print("ate_est_aipw", ate_est_aipw)
    print("ate_ci_aipw", ate_ci_aipw)

    '''PPI Implementation'''
    N = np.array(dataset_val['y']).shape[0]
    N_train = int(N/2)
    N_eval = N - N_train

    Y_N_train = np.array(dataset_val['y']).reshape(-1, 1)[:N_train]
    T_N_train = np.array(dataset_val['A']).reshape(-1, 1)[:N_train]
    X_N_train = np.array(dataset_val['X']).reshape(-1, 1)[:N_train]
    
    X_N_eval = np.array(dataset_val['X']).reshape(-1, 1)[N_train:]
    T_N_eval = np.array(dataset_val['A']).reshape(-1, 1)[N_train:]
    Y_N_eval = np.array(dataset_val['y']).reshape(-1, 1)[N_train:]

    est_2 = ForestDRLearner(
        model_regression=XGBRegressor(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.005,
            subsample=0.7,
            colsample_bytree=0.8,
            min_child_weight=3,        
            reg_alpha=0.1,            
            reg_lambda=1.0,            
            random_state=42,
            n_jobs=-1   
        ),
        model_propensity=XGBClassifier(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.005,
            subsample=0.7,
            colsample_bytree=0.8,
            min_child_weight=3,        
            reg_alpha=0.1,            
            reg_lambda=1.0,            
            random_state=42,
            n_jobs=-1 
        ),
        min_samples_leaf=30,
        n_estimators=300,
        random_state=42
    )
    
    y_N_train = Y_N_train.reshape(N_train)
    est_2.fit(y_N_train, T_N_train, X=X_N_train)

    cate_N = est_2.effect(X_N_eval)
    ate_N = np.mean(cate_N)
    var_N = np.var(cate_N)
    pred_n = est_2.effect(X_train).reshape(-1, 1)

    print("aipw", aipw.shape)
    print("pred_n", pred_n.shape)
    mean_rectifier = np.mean(aipw - pred_n)
    var_rectifier = np.var(aipw - pred_n)
    
    ate_est_ppi = ate_N + mean_rectifier
    ate_ci_norm_ppi = (ate_est_ppi - z_alpha * np.sqrt(var_rectifier/n + var_N/N_eval),
                       ate_est_ppi + z_alpha * np.sqrt(var_rectifier/n + var_N/N_eval))
    print("var_ppi", var_rectifier)
    print("ate_est_ppi", ate_est_ppi)
    print("ate_ci_ppi", ate_ci_norm_ppi)

    '''Normal/Asymptotic setting: Observational data only'''
    ate_est_obs = ate_N
    ate_ci_obs = (ate_est_obs - z_alpha * np.sqrt(var_N/N_eval), \
                  ate_est_obs + z_alpha * np.sqrt(var_N/N_eval))
    print("ate_ci_obs", ate_ci_obs)

    return [ate_est_aipw, ate_est_ppi, ate_est_obs], \
           [ate_ci_aipw, ate_ci_norm_ppi, ate_ci_obs]


def sim_cases(seed, df_rct, df_obs, significance_level, delta):
   
   print(f"RCT data first 5 rows:\n{df_rct.head()}")
   print(f"Obs data first 5 rows:\n{df_obs.head()}")
   
   set_seed(seed)
   
   ate_estimates, ate_ci = get_estimates(df_rct, df_obs, delta, significance_level)
   return ate_estimates, ate_ci 