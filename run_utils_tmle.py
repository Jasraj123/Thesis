import random
import sys, os, yaml
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from scipy.stats import norm
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import pandas as pd

# You'll need to install zEpid for TMLE implementation
# pip install zepid
from zepid.causal.doublyrobust import TMLE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_project_path():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    return str(path.parent.absolute())

def load_yaml(path_relative):
    return yaml.safe_load(open(get_project_path() + path_relative + ".yaml", 'r'))

def save_yaml(path_relative, file):
    with open(get_project_path() + path_relative + ".yaml", 'w') as outfile:
        yaml.dump(file, outfile, default_flow_style=False)

def get_tmle_estimates(dataset_train, dataset_val, delta, significance_level=0.05):
    '''
    Estimate ATE using TMLE and perform prediction-powered inference
    '''
    alpha = significance_level
    z_alpha = norm.ppf(1 - alpha/2)
    
    # Training data (RCT)
    Y_train = np.array(dataset_train['y']).reshape(-1)
    A_train = np.array(dataset_train['A']).reshape(-1)
    X_train = np.array(dataset_train['X']).reshape(-1)
    n = Y_train.shape[0]
    
    # Validation data (Observational)
    N = np.array(dataset_val['y']).shape[0]
    N_train = int(N/2)
    N_eval = N - N_train
    
    Y_N_train = np.array(dataset_val['y']).reshape(-1)[:N_train]
    T_N_train = np.array(dataset_val['A']).reshape(-1)[:N_train]
    X_N_train = np.array(dataset_val['X']).reshape(-1)[:N_train]
    
    X_N_eval = np.array(dataset_val['X']).reshape(-1)[N_train:]
    T_N_eval = np.array(dataset_val['A']).reshape(-1)[N_train:]
    Y_N_eval = np.array(dataset_val['y']).reshape(-1)[N_train:]
    
    # Prepare data frames for TMLE
    rct_df = pd.DataFrame({
        'X': X_train,
        'A': A_train,
        'Y': Y_train
    })
    
    obs_train_df = pd.DataFrame({
        'X': X_N_train,
        'A': T_N_train,
        'Y': Y_N_train
    })
    
    obs_eval_df = pd.DataFrame({
        'X': X_N_eval,
        'A': T_N_eval,
        'Y': Y_N_eval
    })
    
    # 1. TMLE on RCT data
    tmle_rct = TMLE(rct_df, exposure='A', outcome='Y')
    
    # Adding covariates
    tmle_rct.add_covariate(['X'])
    
    # Specifying estimation methods for Q (outcome) and g (propensity score)
    tmle_rct.estimate_risk_differences(
        g_estimator=XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        q_estimator=XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    )
    
    # Extract ATE estimate and confidence interval
    ate_est_tmle = tmle_rct.risk_difference
    ate_ci_tmle = (tmle_rct.risk_difference_lower, tmle_rct.risk_difference_upper)
    
    print("ate_est_tmle (RCT)", ate_est_tmle)
    print("ate_ci_tmle (RCT)", ate_ci_tmle)
    
    # 2. TMLE on observational training data
    tmle_obs = TMLE(obs_train_df, exposure='A', outcome='Y')
    tmle_obs.add_covariate(['X'])
    tmle_obs.estimate_risk_differences(
        g_estimator=XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        ),
        q_estimator=XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
    )
    
    # 3. TMLE on observational evaluation data
    tmle_eval = TMLE(obs_eval_df, exposure='A', outcome='Y')
    tmle_eval.add_covariate(['X'])
    tmle_eval.estimate_risk_differences(
        g_estimator=XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        ),
        q_estimator=XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
    )
    
    # Extract ATE estimates for observational data
    ate_N = tmle_eval.risk_difference
    se_N = (tmle_eval.risk_difference_upper - tmle_eval.risk_difference_lower) / (2 * z_alpha)
    var_N = se_N ** 2
    
    # Calculate individual treatment effects difference
    # Note: TMLE doesn't provide individual effects directly like some other methods,
    # so we'll use a surrogate approach for PPI
    
    # Get conditional expectations for both treatments
    # We need to implement custom individual predictions since zEpid doesn't provide this directly
    g_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    g_model.fit(rct_df[['X']], rct_df['A'])
    
    q1_model = XGBRegressor(
        n_estimators=100, 
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    q1_model.fit(rct_df[rct_df['A'] == 1][['X']], rct_df[rct_df['A'] == 1]['Y'])
    
    q0_model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    q0_model.fit(rct_df[rct_df['A'] == 0][['X']], rct_df[rct_df['A'] == 0]['Y'])
    
    # Predict individual effects for RCT data
    ite_rct = q1_model.predict(rct_df[['X']]) - q0_model.predict(rct_df[['X']])
    
    # Fit models on observational data for prediction
    g_model_obs = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    g_model_obs.fit(obs_train_df[['X']], obs_train_df['A'])
    
    q1_model_obs = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    q1_model_obs.fit(obs_train_df[obs_train_df['A'] == 1][['X']], 
                     obs_train_df[obs_train_df['A'] == 1]['Y'])
    
    q0_model_obs = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    q0_model_obs.fit(obs_train_df[obs_train_df['A'] == 0][['X']], 
                     obs_train_df[obs_train_df['A'] == 0]['Y'])
    
    # Predict individual effects for RCT data using observational models
    pred_n = q1_model_obs.predict(rct_df[['X']]) - q0_model_obs.predict(rct_df[['X']])
    
    # Calculate mean and variance of the difference
    mean_rectifier = np.mean(ite_rct - pred_n)
    var_rectifier = np.var(ite_rct - pred_n)
    
    # PPI estimate using TMLE
    ate_est_ppi_tmle = ate_N + mean_rectifier
    ate_ci_ppi_tmle = (ate_est_ppi_tmle - z_alpha * np.sqrt(var_rectifier/n + var_N),
                      ate_est_ppi_tmle + z_alpha * np.sqrt(var_rectifier/n + var_N))
    
    print("var_ppi_tmle", var_rectifier)
    print("ate_est_ppi_tmle", ate_est_ppi_tmle)
    print("ate_ci_ppi_tmle", ate_ci_ppi_tmle)
    
    # Observational estimate only
    ate_est_obs_tmle = ate_N
    ate_ci_obs_tmle = (ate_est_obs_tmle - z_alpha * np.sqrt(var_N),
                      ate_est_obs_tmle + z_alpha * np.sqrt(var_N))
    
    print("ate_ci_obs_tmle", ate_ci_obs_tmle)
    
    return [ate_est_tmle, ate_est_ppi_tmle, ate_est_obs_tmle], \
           [ate_ci_tmle, ate_ci_ppi_tmle, ate_ci_obs_tmle]

def sim_cases(seed, df_rct, df_obs, significance_level, delta):
    """
    Run simulation cases with TMLE methods on provided RCT and observational datasets.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    df_rct : pandas.DataFrame
        RCT dataset
    df_obs : pandas.DataFrame
        Observational dataset
    significance_level : float
        Statistical significance level (alpha)
    delta : float
        Parameter for confidence interval calculations
    
    Returns:
    --------
    ate_estimates : list
        List of ATE estimates from different methods
    ate_ci : list
        List of confidence intervals for ATE estimates
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Get estimates
    ate_estimates, ate_ci = get_tmle_estimates(df_rct, df_obs, delta, significance_level)

    return ate_estimates, ate_ci 