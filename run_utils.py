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
from synthetic_data_generation import *
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, r2_score, mean_squared_error


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

def data_generation(all_covs, n_rct, n_MC, X_range, pasx, seed, df_obs=None):
    SyntheticData = SyntheticDataModule(
        n_rct=n_rct,
        n_MC=n_MC,
        covs=all_covs,
        X_range=X_range,
        pasx=pasx,
        df_obs=df_obs,
        seed=seed + 4
    )

    mean_trail, _ = SyntheticData.get_true_mean()
    df_comp_big, df_obs_out = SyntheticData.get_df()

    return mean_trail, df_comp_big, df_obs_out

def tune_model(model, param_grid, X, y, scoring, cv=3, verbose=1):
    """
    Tune model hyperparameters using GridSearchCV.
    
    Parameters:
    -----------
    model : estimator object
        The model to tune
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter values
    X : array-like
        Training data
    y : array-like
        Target values
    scoring : string
        Scoring method for model evaluation
    cv : int, default=3
        Number of cross-validation folds
    verbose : int, default=1
        Verbosity level
        
    Returns:
    --------
    best_model : estimator object
        The best model found by GridSearchCV
    """
    grid_search = GridSearchCV(
        model, param_grid, scoring=scoring, cv=cv, verbose=verbose, n_jobs=-1
    )
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def estimate_e(X, A, model_e=None):
    '''
    Estimate propensity score using a regularized and tuned model
    '''
    print(f"\n------ PROPENSITY MODEL DEBUG ------")
    print(f"X shape: {X.shape}, A shape: {A.shape}")
    
    if model_e is None or isinstance(model_e, LogisticRegression):
        # Default propensity model with hyperparameter tuning
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],  # using only l2 for compatibility with all solvers
            'solver': ['lbfgs', 'newton-cg'],
            'class_weight': [None, 'balanced']
        }
        base_model = LogisticRegression(max_iter=1000, random_state=42)
        model_e = tune_model(
            base_model, 
            param_grid, 
            X, 
            A.ravel(), 
            scoring='roc_auc'
        )
    
    e = model_e.fit(X, A.ravel()).predict_proba(X)[:, 1]
    
    # Add these debug statements to check propensity model fit
    y_pred = model_e.predict(X)
    auc = roc_auc_score(A, e)
    brier = brier_score_loss(A, e)
    logloss = log_loss(A, e)
    
    print(f"Propensity model fit metrics:")
    print(f"  - AUC-ROC: {auc:.4f} (higher is better, > 0.7 is reasonable)")
    print(f"  - Brier score: {brier:.4f} (lower is better, < 0.25 is reasonable)")
    print(f"  - Log loss: {logloss:.4f} (lower is better)")
    
    # Diagnostic for propensity distribution
    print(f"Propensity distribution summary:")
    print(f"  - Min: {np.min(e):.4f}, Max: {np.max(e):.4f}")
    print(f"  - Mean: {np.mean(e):.4f}, Std: {np.std(e):.4f}")
    print(f"  - Quantiles (10%, 25%, 50%, 75%, 90%): {np.quantile(e, [0.1, 0.25, 0.5, 0.75, 0.9])}")
    
    # Check for extreme propensity scores (potential positivity violations)
    extreme_props = np.sum((e < 0.1) | (e > 0.9)) / len(e)
    print(f"  - Proportion of extreme propensity scores (<0.1 or >0.9): {extreme_props:.4f}")
    
    return e.reshape(-1, 1)

def estimate_mu(X, A, y, model_y=None):
    '''
    Estimate response function using a regularized and tuned model
    '''
    train_data = np.concatenate((X, A), axis=1)
    
    if model_y is None or isinstance(model_y, LinearRegression):
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
    
    y_pred = mu.predict(train_data)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"\n------ OUTCOME MODEL DEBUG ------")
    print(f"Outcome model fit metrics:")
    print(f"  - R² score: {r2:.4f} (higher is better, > 0.5 is reasonable)")
    print(f"  - RMSE: {rmse:.4f} (lower is better, depends on outcome scale)")
    
    # Calculate and print residuals summary
    residuals = y - y_pred
    print(f"Residuals summary:")
    print(f"  - Mean: {np.mean(residuals):.4f} (should be close to 0)")
    print(f"  - Std: {np.std(residuals):.4f}")
    print(f"  - Quantiles (10%, 25%, 50%, 75%, 90%): {np.quantile(residuals, [0.1, 0.25, 0.5, 0.75, 0.9])}")
    
    # Simple check for heteroskedasticity by treatment group
    res_treated = residuals[A.flatten() == 1]
    res_control = residuals[A.flatten() == 0]
    print(f"  - Residual std by group: Control={np.std(res_control):.4f}, Treated={np.std(res_treated):.4f}")
    
    test_0 = np.concatenate((X, np.zeros_like(A)), axis=1)
    test_1 = np.concatenate((X, np.ones_like(A)), axis=1)
    mu0 = mu.predict(test_0)
    mu1 = mu.predict(test_1)
        
    return mu0, mu1

def get_estimates(dataset_train, dataset_val, delta, significance_level=0.05):
    '''Param setting'''
    print("\n====== STARTING ESTIMATION PROCEDURE ======")
    print(f"Dataset train shape: {dataset_train.shape}")
    print(f"Dataset validation shape: {dataset_val.shape}")
    
    alpha = significance_level
    z_alpha = norm.ppf(1 - alpha/2)
    
    Y_train = np.array(dataset_train['y']).reshape(-1, 1)
    A_train = np.array(dataset_train['A']).reshape(-1, 1)
    X_train = np.array(dataset_train['X']).reshape(-1, 1)
    n = Y_train.shape[0]
    print(f"Sample size n: {n}")
    
    '''Normal/Asymptotic setting: AIPW'''
    e = estimate_e(X_train, A_train)
    mu0, mu1 = estimate_mu(X_train, A_train, Y_train)
    
    # Fix the shape issue by ensuring all components are properly shaped
    # Convert to flattened arrays where needed
    A_flat = A_train.flatten()
    Y_flat = Y_train.flatten()
    e_flat = e.flatten()
    mu0_flat = np.array(mu0).flatten()
    mu1_flat = np.array(mu1).flatten()
    
    # Calculate AIPW with explicit control of dimensions
    aipw_term1 = (A_flat * Y_flat / e_flat) - ((1 - A_flat) * Y_flat / (1 - e_flat))
    aipw_term2 = ((A_flat - e_flat) / e_flat * (1 - e_flat)) * ((1-e_flat) * mu1_flat + e_flat * mu0_flat)
    aipw = (aipw_term1 - aipw_term2).reshape(-1, 1)
    
    # Print shape information for debugging
    print(f"Shape of A_train: {A_train.shape}")
    print(f"Shape of Y_train: {Y_train.shape}")
    print(f"Shape of e: {e.shape}")
    print(f"Shape of mu0: {np.array(mu0).shape}")
    print(f"Shape of mu1: {np.array(mu1).shape}")
    print(f"Corrected shape of AIPW: {aipw.shape}")
    
    ate_est_aipw = np.mean(aipw)
    ate_ci_aipw = (ate_est_aipw - z_alpha * np.sqrt(np.var(aipw)/n), ate_est_aipw + z_alpha * np.sqrt(np.var(aipw)/n))
    
    print(f"AIPW variance: {np.var(aipw):.4f}")
    print(f"AIPW estimate: {ate_est_aipw:.4f}")
    print(f"AIPW CI: {ate_ci_aipw}")
    print(f"AIPW CI width: {ate_ci_aipw[1] - ate_ci_aipw[0]:.4f}")

    '''PPI Implementation'''
    print("\n------ PPI ESTIMATION ------")
    N = dataset_val.shape[0]
    N_train = int(N/2)
    N_eval = N - N_train
    
    Y_N_train = np.array(dataset_val['y']).reshape(-1, 1)[:N_train]
    T_N_train = np.array(dataset_val['A']).reshape(-1, 1)[:N_train]
    X_N_train = np.array(dataset_val['X']).reshape(-1, 1)[:N_train]
    X_N_eval = np.array(dataset_val['X']).reshape(-1, 1)[N_train:]
    T_N_eval = np.array(dataset_val['A']).reshape(-1, 1)[N_train:]
    Y_N_eval = np.array(dataset_val['y']).reshape(-1, 1)[N_train:]
    
    print(f"Treatment proportion in obs train data: {np.mean(T_N_train):.4f}")
    
    # Create a validation set for early stopping
    X_train_fit, Y_train_fit, T_train_fit = train_test_split(
        X_N_train, Y_N_train, T_N_train, test_size=0.2, random_state=42
    )
    
    # Tune outcome regression model
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
    
    # Prepare combined features (X and treatment)
    X_T_train_fit = np.concatenate((X_train_fit, T_train_fit), axis=1)
    
    regressor = tune_model(
        base_regressor,
        outcome_param_grid,
        X_T_train_fit,
        Y_train_fit.ravel(),
        scoring='neg_root_mean_squared_error'
    )
    
    # Tune propensity model
    class_weight = float(np.sum(T_N_train == 0) / np.sum(T_N_train == 1))
    print(f"Class imbalance ratio (control/treatment): {class_weight:.4f}")
    
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
    
    # Use calibration for better probability estimates
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
    print(f"ATE from obs data: {ate_N:.4f}")
    print(f"Variance of CATE estimates: {var_N:.4f}")
    
    pred_n = est_2.effect(X_train).reshape(-1, 1)
    print(f"Shape of AIPW: {aipw.shape}")
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
    print("\n====== ESTIMATION COMPLETE ======")

    return [ate_est_aipw, ate_est_ppi, ate_est_obs], \
           [ate_ci_aipw, ate_ci_norm_ppi, ate_ci_obs]

def sim_cases(seed, df_rct, df_obs, significance_level, delta):
    print(f"RCT data shape: {df_rct.shape}")
    print(f"Obs data shape: {df_obs.shape}")
    print(f"RCT data first 5 rows:\n{df_rct.head()}")
    print(f"Obs data first 5 rows:\n{df_obs.head()}")

    ate_estimates, ate_ci = get_estimates(df_rct, df_obs, delta, significance_level)
    
    # Summary of results
    methods = ["AIPW", "PPI", "Observational-only"]
    for i, method in enumerate(methods):
        print(f"{method}: Estimate = {ate_estimates[i]:.4f}, CI = {ate_ci[i]}, Width = {ate_ci[i][1] - ate_ci[i][0]:.4f}")

    return ate_estimates, ate_ci