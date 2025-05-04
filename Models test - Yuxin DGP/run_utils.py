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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBRegressor, XGBClassifier
from rct_data.synthetic_data_generation import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

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


def rbf_linear_kernel(X1, X2, length_scales=np.array([0.1,0.1]), alpha=np.array([0.1,0.1]), var=5):  # works with 2D covariates only
    distances = np.linalg.norm((X1[:, None, :] - X2[None, :, :]) / length_scales, axis=2)
    rbf_term = var * np.exp(-0.5 * distances**2)
    linear_term = np.dot(np.dot(X1, np.diag(alpha)), X2.T)
    return rbf_term + linear_term

def data_generation(gp_params, all_covs, big_n_rct, big_n_obs, n_MC, X_range, U_range, pasx, seed):
    gp_funcs = {}

    gp_funcs["om_A0"] = sample_outcome_model_gp(X_range, U_range, gp_params["om_A0_par"], seed + 0)  # GP - outcome model under treatment A=0
    gp_funcs["om_A1"] = sample_outcome_model_gp(X_range, U_range, gp_params["om_A1_par"], seed + 1)  # GP - outcome model under treatment A=1
    gp_funcs["w_sel"] = sample_outcome_model_gp(X_range, U_range, gp_params["w_sel_par"], seed + 2)  # GP - selection score model P(S=1|X)
    gp_funcs["w_trt"] = sample_outcome_model_gp(X_range, U_range, gp_params["w_trt_par"], seed + 3)  # GP - propensity score in OBS study P(A=1|X, S=2)

    SyntheticData = SyntheticDataModule(big_n_rct, big_n_obs, n_MC, gp_funcs, all_covs, X_range, U_range, pasx, seed + 4)
    mean_trail, _ = SyntheticData.get_true_mean()
    df_comp_big, df_obs = SyntheticData.get_df() 

    return mean_trail, df_comp_big, df_obs

def tune_model(model, param_grid, X, y, scoring, cv=3, verbose=1, n_iter=10):
  
    random_search = RandomizedSearchCV(
        model, param_grid, scoring=scoring, cv=cv, verbose=verbose, 
        n_jobs=-1, n_iter=n_iter, random_state=42
    )
    random_search.fit(X, y)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def sample_outcome_model_gp(X, U, param, seed):  # works with 2D covariates only

    np.random.seed(seed)
    XX, UU = np.meshgrid(X, U)
    XU_flat = np.c_[XX.ravel(), UU.ravel()]

    mean = np.zeros(len(XU_flat))

    if param["kernel"] == "rbf":
        K = rbf_linear_kernel(XU_flat, XU_flat, np.array(param["ls"]), np.array(param["alpha"]))

    f_sample = np.random.multivariate_normal(mean, K)
    Y = f_sample.reshape(XX.shape)

    # gp_func = interpolate.interp2d(X, U, Y, kind="linear")
    gp_func = interpolate.RectBivariateSpline(X, U, Y.T)

    return gp_func

def estimate_e(X, A, model_e=None):
    '''
        Estimate propensity score using a model_e
    '''
    if model_e is None:
        class_weight = float(np.sum(A == 0) / np.sum(A == 1))
        print(f"Class imbalance ratio (control/treatment): {class_weight:.4f}")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.03, 0.05],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        base_model = XGBClassifier(
            gamma=0.1,
            reg_alpha=0.2,
            reg_lambda=1.0,
            scale_pos_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        model_e = tune_model(
            base_model, 
            param_grid, 
            X, 
            A.ravel(), 
            scoring='roc_auc'
        )
        
        # Apply calibration to ensure well-calibrated probabilities
        model_e = CalibratedClassifierCV(
            model_e,
            method='sigmoid',
            cv=3
        )

    e = model_e.fit(X, A).predict_proba(X)[:, 1]
    return e.reshape(-1, 1)

def estimate_mu(X, A, y, model_y=None):
    '''
        Estimate response function using a model_y
    '''
    train_data = np.concatenate((X, A), axis=1)

    if model_y is None:
        # Default outcome model with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        base_model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        model_y = tune_model(
            base_model,
            param_grid,
            train_data,
            y.ravel(),
            scoring='neg_root_mean_squared_error'
        )
    
    mu = model_y.fit(train_data, y.reshape(-1, 1))
    test_0 = np.concatenate((X, np.zeros_like(A)), axis=1)
    test_1 = np.concatenate((X, np.ones_like(A)), axis=1)
    mu0 = mu.predict(test_0)
    mu1 = mu.predict(test_1)

    return mu0, mu1


def get_estimates(dataset_train, dataset_val, delta, significance_level = 0.05):

    '''Param settinig'''
    alpha = significance_level
    z_alpha = norm.ppf(1 - alpha/2)
    Y_train = np.stack(np.array(dataset_train['y']))
    A_train = np.array(dataset_train['A']).reshape(-1, 1)
    X_train = np.array(dataset_train['X']).reshape(-1, 1)
    n = Y_train.shape[0]
    d = X_train.shape[1]

    '''IPW Implementation test'''
    e = estimate_e(X_train, A_train)
    mu0, mu1 = estimate_mu(X_train, A_train, Y_train)
    A_flat = A_train.flatten()
    Y_flat = Y_train.flatten()
    e_flat = e.flatten()
    mu0_flat = mu0
    mu1_flat = mu1
    
    aipw_term1 = (A_flat * Y_flat / e_flat) - ((1 - A_flat) * Y_flat / (1 - e_flat))
    aipw_term2 = ((A_flat - e_flat) / e_flat * (1 - e_flat)) * ((1-e_flat) * mu1_flat + e_flat * mu0_flat)
    aipw = (aipw_term1 - aipw_term2).reshape(-1, 1)
    
    print(f"Shape of A_train: {A_train.shape}")
    print(f"Shape of Y_train: {Y_train.shape}")
    print(f"Shape of e: {e.shape}")
    print(f"Shape of mu0: {np.array(mu0).shape}")
    print(f"Shape of mu1: {np.array(mu1).shape}")
    print(f"Corrected shape of AIPW: {aipw.shape}")

    ate_est_aipw = np.mean(aipw)
    ate_ci_aipw = (ate_est_aipw - z_alpha * np.sqrt(np.var(aipw)/n), ate_est_aipw + z_alpha * np.sqrt(np.var(aipw)/n))
    print("var_aipw", np.var(aipw))
    print("ate_est_aipw", ate_est_aipw)
    print("ate_ci_aipw", ate_ci_aipw)

    '''Normal/Asymptotic setting + PPI '''
    N = np.stack(np.array(dataset_val['y'])).shape[0]
    N_train = int(N/2)
    N_eval = N - N_train
    Y_N_train = np.stack(np.array(dataset_val['y']))[:N_train, :]
    T_N_train = np.array(dataset_val['A']).reshape(-1, 1)[:N_train, :]
    X_N_train = np.array(dataset_val['X']).reshape(-1, 1)[:N_train, :]
    X_N_eval = np.array(dataset_val['X']).reshape(-1, 1)[N_train:, :]
    T_N_eval = np.array(dataset_val['A']).reshape(-1, 1)[N_train:, :]
    Y_N_eval = np.stack(np.array(dataset_val['y']))[N_train:, :]

    X_train_fit, X_test_fit, Y_train_fit, Y_test_fit, T_train_fit, T_test_fit = train_test_split(
        X_N_train, Y_N_train, T_N_train, test_size=0.2, random_state=42
    )

    outcome_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.03, 0.05],
        'min_child_weight': [1, 3, 5]
    }
    
    base_regressor = XGBRegressor(
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,          
        reg_alpha=0.2,      
        reg_lambda=1.0,     
        random_state=42,
        n_jobs=-1,           
        objective='reg:squarederror'
    )
    
    X_T_train_fit = np.concatenate((X_train_fit, T_train_fit), axis=1)
    
    regressor = tune_model(
        base_regressor,
        outcome_param_grid,
        X_T_train_fit,
        Y_train_fit.ravel(),
        scoring='neg_root_mean_squared_error'
    )
    
    class_weight = float(np.sum(T_N_train == 0) / np.sum(T_N_train == 1))
 
    propensity_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.03, 0.05]
    }
    
    base_propensity = XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=class_weight,
    )
    
    tuned_propensity = tune_model(
        base_propensity,
        propensity_param_grid,
        X_N_train,
        T_N_train.ravel(),
        scoring='roc_auc',
        verbose=0
    )
    
    propensity = CalibratedClassifierCV(
        tuned_propensity,
        method='sigmoid',
        cv=3
    )

    est_2 = ForestDRLearner(
        model_regression=regressor,
        model_propensity=propensity,
        min_samples_leaf=10,
        n_estimators=252,
        max_depth=6,
        random_state=42
    )

    y_N_train = Y_N_train.reshape(N_train)
    est_2.fit(y_N_train, T_N_train, X=X_N_train)

    cate_N = est_2.effect(X_N_eval)
    ate_N = np.mean(cate_N)
    var_N = np.var(cate_N)

    pred_n = est_2.effect(X_train).reshape(-1, 1)  

    print(f"Shape of AIPW: {aipw.shape}")
    print(f"Shape of pred_n: {pred_n.shape}")

    mean_rectifier = np.mean(aipw - pred_n)
    var_rectifier = np.var(aipw - pred_n)

    ate_est_ppi = ate_N + mean_rectifier
    ate_ci_norm_ppi = (ate_est_ppi - z_alpha * np.sqrt(var_rectifier/n + var_N/N_eval), \
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
    print(f"RCT data shape: {df_rct.shape}")
    print(f"Obs data shape: {df_obs.shape}")
    print(f"RCT data first 5 rows:\n{df_rct.head()}")
    print(f"Obs data first 5 rows:\n{df_obs.head()}")
    
    ate_estimates, ate_ci = get_estimates(df_rct, df_obs, delta, significance_level)

    return ate_estimates, ate_ci