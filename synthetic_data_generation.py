import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import expit
from scipy.stats import norm

import os, json, itertools


class SyntheticDataModule:
    def __init__(self,
                n_rct=200,
                n_MC=100000,
                covs=["X"],  # Keep X as the primary covariate for compatibility
                X_range=np.linspace(-1, 1, 51),
                pasx={"lb": 0.1, "ub": 0.9, "trial": 0.5},
                seed=42,
                df_obs=None):
        
        self.df_obs = df_obs
        self.n_rct = n_rct
        self.n_MC = n_MC
        self.covs = covs
        self.X_range = X_range
        self.seed = seed
        self.d = len(covs)
        self.prop_clip_lb = pasx["lb"]
        self.prop_clip_ub = pasx["ub"]
        self.pas1 = pasx["trial"]  # probability of treatment assignment in the trial
        self.sbl, self.sbu = 0.5, 1
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Complex function parameters
        self.freq_params = np.random.uniform(0.5, 3.0, size=3)  # Still use 3 features internally
        self.phase_params = np.random.uniform(0, 2*np.pi, size=3)
        self.amplitude_params = np.random.uniform(0.5, 2.5, size=3)
        self.interaction_weights = np.random.uniform(-1, 1, size=(3, 3))

    def _rbf_kernel_like(self, X1, X2=None, length_scale=1.0):
        """Compute an RBF-like function between points"""
        if X2 is None:
            X2 = X1
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        return np.exp(-0.5 * np.sum(((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) / length_scale) ** 2, axis=2))
    
    def _generate_complex_nonlinear_features(self, X):
        """Generate complex nonlinear features from input covariates"""
        n_samples = len(X)
        
        # Create synthetic additional covariates from X
        X1 = X.copy()
        X2 = np.sin(2.5 * X) + 0.3 * np.random.normal(size=n_samples)
        X3 = np.cos(1.7 * X) + 0.4 * np.random.normal(size=n_samples)
        
        X_matrix = np.column_stack([X1, X2, X3])
        
        # Basic nonlinear transformations
        features = np.zeros((n_samples, 5))
        
        # Sinusoidal components with varying frequencies and phases
        for i in range(3):  # Use 3 components
            features[:, 0] += self.amplitude_params[i] * np.sin(self.freq_params[i] * X_matrix[:, i] + self.phase_params[i])
            features[:, 1] += self.amplitude_params[i] * np.cos(self.freq_params[i] * X_matrix[:, i] * 1.5)
        
        # Exponential and polynomial components
        features[:, 2] = np.exp(-0.5 * np.sum(X_matrix[:, :3]**2, axis=1))
        features[:, 3] = np.sum(X_matrix[:, :3]**3, axis=1) - np.sum(X_matrix[:, :3]**2, axis=1)
        
        # Interaction terms
        for i in range(3):
            for j in range(i+1, 3):
                features[:, 4] += self.interaction_weights[i, j] * X_matrix[:, i] * X_matrix[:, j]
        
        return features

    def _outcome_model(self, X, treatment):
        """
        Complex nonlinear outcome model with treatment interaction effects
        """
        n = len(X)
        
        # Generate complex features
        features = self._generate_complex_nonlinear_features(X)
        
        # Treatment-specific effects with complex heterogeneity
        treatment_effect = 15.0 + 3.0 * features[:, 0] + 2.5 * features[:, 3]
        
        # Control outcome (complex baseline)
        y0 = 5.0 * features[:, 0] + 3.0 * features[:, 1] + 7.0 * features[:, 2] + 2.0 * features[:, 4]
        
        # Treatment outcome (baseline + effect)
        y1 = y0 + treatment_effect
        
        # Heteroskedastic noise
        noise_scale = 0.5 + 0.5 * np.abs(features[:, 2])
        noise = np.random.normal(0, noise_scale, size=n)
        
        # Select outcomes based on treatment
        if np.isscalar(treatment):
            if treatment == 0:
                return y0 + noise
            else:
                return y1 + noise
        else:
            return np.where(treatment.reshape(-1) == 1, y1, y0) + noise

    def _generate_data_rct(self):
        df = pd.DataFrame(index=np.arange(self.n_rct))
        np.random.seed(self.seed + 1)

        # Generate primary covariate
        df["X"] = np.random.uniform(-1.5, 1.5, size=self.n_rct)
        
        # For RCT, treatment is random (not influenced by any confounders)
        df["A"] = np.array(self.pas1 > np.random.uniform(size=self.n_rct), dtype=int)

        # Generate potential outcomes (without U for true RCT)
        print("Generating potential outcomes...")
        Y0 = self._outcome_model(df["X"].values, treatment=0)
        Y1 = self._outcome_model(df["X"].values, treatment=1)
        
        df['Y0'] = Y0
        df['Y1'] = Y1

        # Observed outcome
        df["y"] = df["Y1"] * df["A"] + df["Y0"] * (1 - df["A"])

        # Return only the observed variables (X, A, y)
        return df[["X", "A", "y"]]
        
    def get_df(self):
        """Get both RCT and observational datasets"""
        df_rct = self._generate_data_rct()

        return df_rct, self.df_obs

    def get_true_mean(self):
        """Calculate true treatment effect using Monte Carlo"""
        np.random.seed(self.seed)
        
        # Generate covariate for MC samples
        X_mc = np.random.uniform(-1.5, 1.5, size=self.n_MC)
        
        # Calculate potential outcomes for all MC samples 
        print("Calculating potential outcomes...")
        Y1 = self._outcome_model(X_mc, treatment=1)  # Outcomes if everyone treated
        Y0 = self._outcome_model(X_mc, treatment=0)  # Outcomes if no one treated

        # Calculate individual treatment effects and their mean
        treatment_effect = Y1 - Y0
        true_ate = np.mean(treatment_effect)
        std_ate = np.std(treatment_effect) / np.sqrt(self.n_MC)

        return true_ate, std_ate


# def generate_large_dataset(n_samples=250000, seed=42, output_file="optimized_generated_data.csv"):
#     np.random.seed(seed)
    
#     large_data_generator = SyntheticDataModule(n_rct=n_samples, seed=seed)
    
#     # Generate outcomes influenced by X, A, and U
#     print(f"Generating data for {n_samples} samples...")
#     large_df = large_data_generator._generate_data_rct()
    
#     large_df.to_csv(output_file, index=False)
#     print(f"Saved large dataset to {output_file}")
    
#     return large_df

# if __name__ == "__main__":
#     generate_large_dataset()
#     print("Done")

