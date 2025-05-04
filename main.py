import numpy as np
import pandas as pd
import os, json, itertools, sys
import yaml

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from run_utils import *


def run_experiment(config_run, seed_list, methods_list, alpha, llm_obs_path):
    #alpha = config_run['data']['alpha']
    delta = config_run['data']['delta']
    data_seed = config_run['data']['data_seed']

    all_covs = config_run['data']['covariate_name']
    X_param = config_run['data']['X_range']
    X_range = np.linspace(X_param[0], X_param[1], X_param[2])
    
    n_rct = config_run['data']['n_rct']
    n_obs = config_run['data']['n_obs']
    
    # Generate a much larger RCT dataset to randomly sample from
    big_n_rct = max(len(seed_list) * n_rct * 5, 10000)  # Generate a large pool of RCT data
    n_MC = config_run['data']['n_MC']
    pasx = config_run['pasx']
    
    print(f"Loading observational data from {llm_obs_path}")
    big_df_obs = pd.read_csv(llm_obs_path)

    print(f"Generating a large RCT data pool with {big_n_rct} samples...")
    mean_trail, big_df_rct, _ = data_generation(
        all_covs=all_covs,
        n_rct=big_n_rct,
        n_MC=n_MC,
        X_range=X_range,
        pasx=pasx,
        seed=data_seed,
        df_obs=big_df_obs  
    )

    print(f"Loaded observational data shape: {big_df_obs.shape}")
    print(f"Generated RCT data pool shape: {big_df_rct.shape}")
        
    save_dir = config_run['relative_path']
    
    os.makedirs(f"{save_dir}/exp_results/alpha_{alpha}/unconfounding_0", exist_ok=True)
    print(f"Results will be saved to {save_dir}/exp_results/alpha_{alpha}/unconfounding_0/")
  
    for seed_index in range(len(seed_list)):
        df = pd.DataFrame()
        current_seed = seed_list[seed_index]
        
        # Set seed for random sampling
        np.random.seed(current_seed)
        
        df_rct = big_df_rct.iloc[seed_index * n_rct: (seed_index + 1) * n_rct]
        
        # Random sampling for observational data
        obs_indices = np.random.choice(len(big_df_obs), size=n_obs, replace=False)
        df_obs = big_df_obs.iloc[obs_indices]
        
        print(f"Randomly sampled RCT data shape: {df_rct.shape}")
        print(f"Randomly sampled Obs data shape: {df_obs.shape}")
        
        ate_est, ate_ci = sim_cases(current_seed, df_rct, df_obs, alpha, delta)

        df["true_ate"] = [mean_trail for i in range(len(methods_list))]
        df["ate_est"] = ate_est
        df["ate_ci_width"] = [0.5 * (ate_ci[i][1] - ate_ci[i][0]) for i in range(len(methods_list))]
        
        print(f"Saving results to estimates_{current_seed}.csv")
        df.to_csv(f"{save_dir}/exp_results/alpha_{alpha}/unconfounding_0/estimates_{current_seed}.csv")


if __name__ == "__main__":
    seed_list = [
        list(range(0, 20, 2)),  # Seeds for alpha=0.05 
        list(range(1, 20, 2))   # Seeds for alpha=0.1
    ]    

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")

    config_run = yaml.safe_load(open(config_path, 'r'))
    # Use the alpha value from config
    alpha = config_run['data']['alpha']
    alpha_index = 0 if alpha == 0.05 else 1  # Determine which seed list to use
    
    
    methods_list = ["normal_aipw", "normal_ppi", "normal_obs"]
    llm_obs_path = os.path.join(current_dir, "observational_data.csv")
    
    run_experiment(config_run, seed_list[alpha_index], methods_list, alpha=alpha,
                  llm_obs_path=llm_obs_path)
