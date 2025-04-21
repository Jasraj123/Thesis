import pandas as pd
import numpy as np

class CSVDataHandler:
    def __init__(self, 
                 rct_csv_path, 
                 obs_csv_path,
                 covs=["X"],
                 seed=42):
        """
        Initialize the CSV data handler to load both RCT and observational data from files.
        
        Parameters:
        -----------
        rct_csv_path : str
            Path to the RCT data CSV file
        obs_csv_path : str
            Path to the observational data CSV file
        covs : list
            List of covariate names
        seed : int
            Random seed for reproducibility
        """
        self.rct_csv_path = rct_csv_path
        self.obs_csv_path = obs_csv_path
        self.covs = covs
        self.seed = seed
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        # Load the full datasets
        self.full_rct_data = pd.read_csv(self.rct_csv_path)
        self.full_obs_data = pd.read_csv(self.obs_csv_path)
        
        print(f"Loaded RCT data with {len(self.full_rct_data)} samples")
        print(f"Loaded observational data with {len(self.full_obs_data)} samples")
        
    def get_df(self, n_rct=None, n_obs=None, seed_index=0):
        """
        Get both RCT and observational datasets with random sampling.
        
        Parameters:
        -----------
        n_rct : int or None
            Number of RCT samples to use. If None, use all available data.
        n_obs : int or None
            Number of observational samples to use. If None, use all available data.
        seed_index : int
            Used to create different random samples for different experiment iterations
            
        Returns:
        --------
        df_rct : pandas.DataFrame
            Randomly sampled RCT dataset
        df_obs : pandas.DataFrame
            Randomly sampled observational dataset
        """
        # Create a new random seed based on the base seed and seed_index
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
        
        # Verify that all required covariates are present
        for cov in self.covs:
            if cov not in df_rct.columns:
                print(f"Warning: Covariate '{cov}' not found in RCT data!")
            if cov not in df_obs.columns:
                print(f"Warning: Covariate '{cov}' not found in observational data!")
            
        return df_rct, df_obs
    
    def compute_true_ate(self, df_rct=None):
        if df_rct is None:
            df_rct = self.full_rct_data
        
        # If the RCT data has potential outcomes Y0 and Y1, use those
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