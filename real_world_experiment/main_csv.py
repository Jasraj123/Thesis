import numpy as np
import pandas as pd
import os, json, sys
import yaml

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from run_utils_csv import sim_cases  # Now using run_utils_csv instead
from csv_data_handler import CSVDataHandler

def run_experiment(config_run, seed_list, methods_list, alpha):
    alpha = config_run['data']['alpha']
    delta = config_run['data']['delta']
    
    all_covs = config_run['data']['covariate_name']
    print(f"Using covariates: {all_covs}")
    
    # Get sample sizes
    n_rct = config_run['data']['n_rct']
    n_obs = config_run['data']['n_obs']
    
    # Get CSV file paths 
    rct_csv_path = config_run['data']['rct_csv_path']
    obs_csv_path = config_run['data']['obs_csv_path']
    
    # Initialize data handler
    data_handler = CSVDataHandler(
        rct_csv_path=rct_csv_path,
        obs_csv_path=obs_csv_path,
        covs=all_covs,
        seed=config_run['data']['data_seed']
    )
    
    mean_trail, _ = data_handler.compute_estimated_ate()
    
    save_dir = config_run['relative_path']
    
    for seed_index, seed in enumerate(seed_list):
        df = pd.DataFrame()
        
        # Get data slices for this seed
        df_rct, df_obs = data_handler.get_df(n_rct=n_rct, n_obs=n_obs, seed_index=seed_index)
        
        print(f"NaN counts in RCT data:\n{df_rct.isna().sum()}")
        print(f"NaN counts in OBS data:\n{df_obs.isna().sum()}")
        
        df_rct = df_rct.dropna()
        df_obs = df_obs.dropna()
        
        if len(df_rct) < 10 or len(df_obs) < 10:  
            print(f"Warning: Too few samples after removing NaN values. RCT: {len(df_rct)}, OBS: {len(df_obs)}")
            continue
        
        print(f"RCT data shape: {df_rct.shape}")
        print(f"OBS data shape: {df_obs.shape}")
        print(f"RCT columns: {df_rct.columns.tolist()}")
        print(f"OBS columns: {df_obs.columns.tolist()}")
            
        try:
            ate_est, ate_ci = sim_cases(seed, df_rct, df_obs, alpha, delta)
        except Exception as e:
            print(f"Error processing seed {seed}: {str(e)}")
            continue
        
        # Store results
        df["true_ate"] = [mean_trail for i in range(len(methods_list))]
        df["ate_est"] = ate_est
        df["ate_ci_width"] = [0.5 * (ate_ci[i][1] - ate_ci[i][0]) for i in range(len(methods_list))]
        
        os.makedirs(f"{save_dir}/exp_results/alpha_{alpha}/unconfounding_0", exist_ok=True)
        df.to_csv(f"{save_dir}/exp_results/alpha_{alpha}/unconfounding_0/estimates_{seed}.csv")

if __name__ == "__main__":
    seed_list = [
        list(range(0, 18, 2)),  # Seeds for alpha=0.05 
        list(range(1, 18, 2))   # Seeds for alpha=0.1
    ]    

    # Get the current directory where main_csv.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config_csv.yaml")

    # Load the config file directly from the current directory
    config_run = yaml.safe_load(open(config_path, 'r'))
    
    # Use the alpha value from config
    alpha = config_run['data']['alpha']
    alpha_index = 0 if alpha == 0.05 else 1  # Determine which seed list to use
    
    methods_list = ["normal_aipw", "normal_ppi", "normal_obs"]
    
    # Run the experiment with the alpha from config
    run_experiment(config_run, seed_list[alpha_index], methods_list, alpha=alpha) 