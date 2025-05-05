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
                n_rct=700,
                n_MC=100000,
                covs=["X"],  
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
        
        np.random.seed(self.seed)
        random.seed(self.seed)


    def _outcome_model(self, X, treatment):
        """
        Advanced nonlinear outcome model with interactions between covariates.
        """
        X = np.asarray(X)
        n = len(X)

        X1 = X + 0.3 * np.random.normal(size=n)  
        X2 = X**2 + 0.5 * np.random.normal(size=n)  #
        X3 = np.sin(X * 2) + 0.2 * np.random.normal(size=n)  

        interaction_1 = X * X1  # interaction between X and X1
        interaction_2 = X2 * X3  # interaction between X2 and X3

        y0 = 3 + 2 * X + 1.5 * X1 - 2 * X2 + 1.8 * np.cos(3*X**2) + 0.5 * interaction_1 + 0.2 * interaction_2

        treatment_effect = 3 + 1.5 * X + 2 * X2 + 0.7 * np.sin(1.5 * X) - 1 * X3

        y1 = y0 + treatment_effect

        # Add noise
        noise_scale = 1.0 + 0.5 * np.abs(X)
        noise = np.random.normal(0, noise_scale, size=n)

        if np.isscalar(treatment):
            return y0 + noise if treatment == 0 else y1 + noise
        else:
            return np.where(treatment.reshape(-1) == 1, y1, y0) + noise

    def _generate_data_rct(self):
        df = pd.DataFrame(index=np.arange(self.n_rct))
        np.random.seed(self.seed + 1)

        df["X"] = np.random.uniform(-1.5, 1.5, size=self.n_rct)
        
        df["A"] = np.array(self.pas1 > np.random.uniform(size=self.n_rct), dtype=int)

        print("Generating potential outcomes...")
        Y0 = self._outcome_model(df["X"].values, treatment=0)
        Y1 = self._outcome_model(df["X"].values, treatment=1)
        
        df["Y0"] = Y0
        df["Y1"] = Y1

        df["y"] = df["Y1"] * df["A"] + df["Y0"] * (1 - df["A"])

        return df[["X", "A", "y"]]      
      
    def get_df(self):
        """Get both RCT and observational datasets"""
        df_rct = self._generate_data_rct()

        return df_rct, self.df_obs

    def get_true_mean(self):
        """Calculate true treatment effect using Monte Carlo"""
        np.random.seed(self.seed)
        
        X_mc = np.random.uniform(-1.5, 1.5, size=self.n_MC)
        
        print("Calculating potential outcomes...")
        Y1 = self._outcome_model(X_mc, treatment=1)  # Outcomes if everyone treated
        Y0 = self._outcome_model(X_mc, treatment=0)  # Outcomes if no one treated

        treatment_effect = Y1 - Y0
        true_ate = np.mean(treatment_effect)
        std_ate = np.std(treatment_effect) / np.sqrt(self.n_MC)

        return true_ate, std_ate
