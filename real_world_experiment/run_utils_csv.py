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

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
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

def tune_model(model, param_grid, X, y, scoring, cv=5, verbose=1, n_iter=20):
  
    random_search = RandomizedSearchCV(
        model, param_grid, scoring=scoring, cv=cv, verbose=verbose, 
        n_jobs=-1, n_iter=n_iter, random_state=42
    )
    random_search.fit(X, y)
    
    print(f"Best parameters: {random_search.best_params_}")
    
    return random_search.best_estimator_

def estimate_e(X, A, model_e=None):
    '''
        Estimate propensity score using a regularized model_e
    '''
    print(f"X shape: {X.shape}, A shape: {A.shape}")
    
    if model_e is None:
        class_weight = float(np.sum(A == 0) / np.sum(A == 1))
        
        param_grid = {
            'n_estimators': [100, 200, 300, 500, 700, 1000],
            'max_depth': [2, 3, 4, 5, 6, 7, 8],
            'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            'min_child_weight': [1, 2, 3, 5, 7],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.5],
            'reg_alpha': [0, 0.1, 0.2, 0.5, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0]
        }
        
        base_model = XGBClassifier(
            scale_pos_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        model_e = tune_model(
            base_model, 
            param_grid, 
            X, 
            A.ravel(), 
            scoring='roc_auc',
            cv=5,
            n_iter=30
        )
        
        model_e = CalibratedClassifierCV(
            model_e,
            method='sigmoid',
            cv=5
        )

    e = model_e.fit(X, A.ravel()).predict_proba(X)[:, 1]
    
    auc = roc_auc_score(A, e)
    
    print(f"  - AUC-ROC: {auc:.4f}")
    
    print(f"  - Min: {np.min(e):.4f}, Max: {np.max(e):.4f}")
    print(f"  - Mean: {np.mean(e):.4f}, Std: {np.std(e):.4f}")
    
    extreme_props = np.sum((e < 0.1) | (e > 0.9)) / len(e)
    print(f"  - Proportion of extreme propensity scores (<0.1 or >0.9): {extreme_props:.4f}")
    
    return e.reshape(-1, 1)


def estimate_mu(X, A, y, model_y=None):
    '''
    Estimate response function using a regularized and tuned model
    '''
    train_data = np.concatenate((X, A), axis=1)
    
    if model_y is None:
        # hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300, 500, 700, 1000],
            'max_depth': [2, 3, 4, 5, 6, 7, 8, 10],
            'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3, 5, 7, 10],
            'gamma': [0, 0.1, 0.2, 0.5],
            'reg_alpha': [0, 0.1, 0.2, 0.5, 1.0, 2.0],
            'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
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
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_iter=30
        )
    
    mu = model_y.fit(train_data, y.reshape(-1, 1))
    
    y_pred = mu.predict(train_data)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"  - R² score: {r2:.4f} ")
    print(f"  - RMSE: {rmse:.4f}")
        
    test_0 = np.concatenate((X, np.zeros_like(A)), axis=1)
    test_1 = np.concatenate((X, np.ones_like(A)), axis=1)
    mu0 = mu.predict(test_0)
    mu1 = mu.predict(test_1)
        
    return mu0, mu1


def get_estimates(dataset_train, dataset_val, delta, significance_level = 0.05):
    '''Param setting'''
    print(f"Dataset train shape: {dataset_train.shape}")
    print(f"Dataset validation shape: {dataset_val.shape}")
    
    alpha = significance_level
    z_alpha = norm.ppf(1 - alpha/2)
    Y_train = np.array(dataset_train['y']).reshape(-1, 1)
    A_train = np.array(dataset_train['A']).reshape(-1, 1)
    
    if 'X' in dataset_train.columns:
        X_train = np.array(dataset_train['X']).reshape(-1, 1)
    else:
        covariate_cols = [col for col in dataset_train.columns 
                          if col not in ['y', 'A', 'Y0', 'Y1']]
        X_train = np.array(dataset_train[covariate_cols])
    
    n = Y_train.shape[0]
    print(f"Sample size n: {n}")
    
    '''Normal/Asymptotic setting: AIPW'''
    e = estimate_e(X_train, A_train)
    mu0, mu1 = estimate_mu(X_train, A_train, Y_train)
    
    A_flat = A_train.flatten()
    Y_flat = Y_train.flatten()
    e_flat = e.flatten()
    
    aipw_term1 = (A_flat * Y_flat / e_flat) - ((1 - A_flat) * Y_flat / (1 - e_flat))
    aipw_term2 = ((A_flat - e_flat) / e_flat * (1 - e_flat)) * ((1-e_flat) * mu1 + e_flat * mu0)
    aipw = (aipw_term1 - aipw_term2).reshape(-1, 1)

    print(f"Shape of A_train: {A_train.shape}")
    print(f"Shape of Y_train: {Y_train.shape}")
    print(f"Shape of e: {e.shape}")
    print(f"Corrected shape of AIPW: {aipw.shape}")
    
    ate_est_aipw = np.mean(aipw)
    ate_ci_aipw = (ate_est_aipw - z_alpha * np.sqrt(np.var(aipw)/n), ate_est_aipw + z_alpha * np.sqrt(np.var(aipw)/n))
    
    print(f"AIPW variance: {np.var(aipw):.4f}")
    print(f"AIPW estimate: {ate_est_aipw:.4f}")
    print(f"AIPW CI: {ate_ci_aipw}")
    print(f"AIPW CI width: {ate_ci_aipw[1] - ate_ci_aipw[0]:.4f}")

    '''PPI Implementation'''
    N = np.array(dataset_val['y']).shape[0]
    N_train = int(N/2)
    N_eval = N - N_train

    Y_N_train = np.array(dataset_val['y']).reshape(-1, 1)[:N_train]
    T_N_train = np.array(dataset_val['A']).reshape(-1, 1)[:N_train]
    
    if 'X' in dataset_val.columns:
        X_N_train = np.array(dataset_val['X']).reshape(-1, 1)[:N_train]
        X_N_eval = np.array(dataset_val['X']).reshape(-1, 1)[N_train:]
    else:
        covariate_cols = [col for col in dataset_val.columns 
                         if col not in ['y', 'A', 'Y0', 'Y1']]
        X_N_train = np.array(dataset_val[covariate_cols])[:N_train]
        X_N_eval = np.array(dataset_val[covariate_cols])[N_train:]
    
    T_N_eval = np.array(dataset_val['A']).reshape(-1, 1)[N_train:]
    Y_N_eval = np.array(dataset_val['y']).reshape(-1, 1)[N_train:]
        
    X_train_fit, X_test_fit, Y_train_fit, Y_test_fit, T_train_fit, T_test_fit = train_test_split(
        X_N_train, Y_N_train, T_N_train, test_size=0.2, random_state=42
    )
    
    outcome_param_grid = {
        'n_estimators': [100, 200, 300, 500, 700, 1000],
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 10],
        'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        'min_child_weight': [1, 2, 3, 5, 7, 10],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.1, 0.2, 0.5, 1.0, 2.0],
        'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
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
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_iter=30
    )
    
    # Tune propensity model
    class_weight = float(np.sum(T_N_train == 0) / np.sum(T_N_train == 1))
    
    propensity_param_grid = {
        'n_estimators': [100, 200, 300, 500, 700, 1000],
        'max_depth': [2, 3, 4, 5, 6, 7, 8],
        'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        'min_child_weight': [1, 2, 3, 5, 7],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.5],
        'reg_alpha': [0, 0.1, 0.2, 0.5, 1.0],
        'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0]
    }
    
    base_propensity = XGBClassifier(
        scale_pos_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    
    tuned_propensity = tune_model(
        base_propensity,
        propensity_param_grid,
        X_N_train,
        T_N_train.ravel(),
        scoring='roc_auc',
        cv=5,
        n_iter=30,
        verbose=0
    )
    
    propensity = CalibratedClassifierCV(
        tuned_propensity,
        method='sigmoid',
        cv=5
    )

    est_2 = ForestDRLearner(
        model_regression=regressor,
        model_propensity=propensity,
        min_samples_leaf=5,  
        n_estimators=500,    
        max_depth=15,        
        random_state=42
    )
    
    y_N_train = Y_N_train.reshape(N_train)
    est_2.fit(y_N_train, T_N_train, X=X_N_train)
    print(f"X_N_train shape: {X_N_train.shape}")
    print(f"T_N_train shape: {T_N_train.shape}")
    print(f"Y_N_train shape: {Y_N_train.shape}")
   
    
    cate_N = est_2.effect(X_N_eval)
    print(f"cate_N shape: {cate_N.shape}")

    ate_N = np.mean(cate_N)
    var_N = np.var(cate_N)    
    print(f"ATE from obs data: {ate_N:.4f}")
    print(f"Variance of CATE estimates: {var_N:.4f}")
    
    pred_n = est_2.effect(X_train).reshape(-1, 1)
    print(f"Shape of pred_n: {pred_n.shape}")
    
    mean_rectifier = np.mean(aipw - pred_n)
    var_rectifier = np.var(aipw - pred_n)
    
    print(f"Mean rectifier: {mean_rectifier:.4f}")
    print(f"Var rectifier: {var_rectifier:.4f}")
    
    ate_est_ppi = ate_N + mean_rectifier
    ate_ci_norm_ppi = (ate_est_ppi - z_alpha * np.sqrt(var_rectifier/n + var_N/N_eval),
                       ate_est_ppi + z_alpha * np.sqrt(var_rectifier/n + var_N/N_eval))
    
    print(f"PPI var: {var_rectifier:.4f}")
    print(f"PPI estimate: {ate_est_ppi:.4f}")
    print(f"PPI CI: {ate_ci_norm_ppi}")
    print(f"PPI CI width: {ate_ci_norm_ppi[1] - ate_ci_norm_ppi[0]:.4f}")

    '''Normal/Asymptotic setting: Observational data only'''
    ate_est_obs = ate_N
    ate_ci_obs = (ate_est_obs - z_alpha * np.sqrt(var_N/N_eval), 
                 ate_est_obs + z_alpha * np.sqrt(var_N/N_eval))
    
    print(f"Obs-only estimate: {ate_est_obs:.4f}")
    print(f"Obs-only CI: {ate_ci_obs}")
    print(f"Obs-only CI width: {ate_ci_obs[1] - ate_ci_obs[0]:.4f}")

    return [ate_est_aipw, ate_est_ppi, ate_est_obs], \
           [ate_ci_aipw, ate_ci_norm_ppi, ate_ci_obs]


def sim_cases(seed, df_rct, df_obs, significance_level, delta):
   
   print(f"RCT data shape: {df_rct.shape}")
   print(f"Obs data shape: {df_obs.shape}")
   print(f"RCT data first 5 rows:\n{df_rct.head()}")
   print(f"Obs data first 5 rows:\n{df_obs.head()}")
   
   set_seed(seed)
   
   ate_estimates, ate_ci = get_estimates(df_rct, df_obs, delta, significance_level)
   
   # Summary of results
   methods = ["AIPW", "PPI", "Observational-only"]
   for i, method in enumerate(methods):
       print(f"{method}: Estimate = {ate_estimates[i]:.4f}, CI = {ate_ci[i]}, Width = {ate_ci[i][1] - ate_ci[i][0]:.4f}")
       
   return ate_estimates, ate_ci 