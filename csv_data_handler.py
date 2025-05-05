import pandas as pd
import numpy as np

class CSVDataHandler:
    def __init__(self, 
                 rct_csv_path, 
                 obs_csv_path,
                 covs=["AGE", "RSBP", "ST1PE"],
                 seed=42):
      
        self.rct_csv_path = rct_csv_path
        self.obs_csv_path = obs_csv_path
        self.covs = covs
        self.seed = seed
        
        np.random.seed(self.seed)
        
        self.full_rct_data = pd.read_csv(self.rct_csv_path)
        self.full_obs_data = pd.read_csv(self.obs_csv_path)
        
        print(f"Loaded RCT data with {len(self.full_rct_data)} samples")
        print(f"Loaded observational data with {len(self.full_obs_data)} samples")
        
    def get_df(self, n_rct=None, n_obs=None, seed_index=0):
        random_state = np.random.RandomState(self.seed + seed_index)
        
        # Random sampling for RCT data
        if n_rct is not None and n_rct < len(self.full_rct_data):
            rct_indices = random_state.choice(len(self.full_rct_data), size=n_rct, replace=False)
            df_rct = self.full_rct_data.iloc[rct_indices].reset_index(drop=True)
            print(f"Randomly sampled {n_rct} points from {len(self.full_rct_data)} RCT data points")
        else:
            df_rct = self.full_rct_data.copy()
            print(f"Using all {len(df_rct)} RCT data points")
        
        # Random sampling for observational data
        if n_obs is not None and n_obs < len(self.full_obs_data):
            obs_indices = random_state.choice(len(self.full_obs_data), size=n_obs, replace=False)
            df_obs = self.full_obs_data.iloc[obs_indices].reset_index(drop=True)
            print(f"Randomly sampled {n_obs} points from {len(self.full_obs_data)} observational data points")
        else:
            df_obs = self.full_obs_data.copy()
            print(f"Using all {len(df_obs)} observational data points")
        
        for cov in self.covs:
            if cov not in df_rct.columns:
                print(f"Warning: Covariate '{cov}' not found in RCT data!")
            if cov not in df_obs.columns:
                print(f"Warning: Covariate '{cov}' not found in observational data!")
            
        return df_rct, df_obs
    
    def compute_estimated_ate(self, df_rct=None):
        if df_rct is None:
            df_rct = self.full_rct_data
        
        if 'Y0' in df_rct.columns and 'Y1' in df_rct.columns:
            treatment_effect = df_rct['Y1'] - df_rct['Y0']
            true_ate = np.mean(treatment_effect)
            std_ate = np.std(treatment_effect) / np.sqrt(len(df_rct))
        else:
            treated = df_rct[df_rct['A'] == 1]['y']
            control = df_rct[df_rct['A'] == 0]['y']
            true_ate = np.mean(treated) - np.mean(control)
            std_ate = np.sqrt(np.var(treated)/len(treated) + np.var(control)/len(control))
        
        return true_ate, std_ate 