import numpy as np
import pandas as pd
import random
import os 
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import expit

import os, json, itertools


class SyntheticDataModule:
    def __init__(self,  
                n_rct=200,
                n_obs=10000,
                n_MC=100000,
                gp_funcs=None,
                covs=["X", "U"],
                X_range=np.linspace(-1,1,51),
                U_range=np.linspace(-1,1,51),
                pasx={"lb":0.1, "ub":0.9, "trial":0.5},
                seed=42,
                ):
        
        self.n_rct = n_rct
        # self.n_tar = n_tar
        self.n_obs = 2*n_obs
        self.n = n_rct
        self.n_MC = n_MC  # monte-carlo sample size to calculate "true mean" in the target population
        self.covs = covs
        self.X = X_range
        self.U = U_range
        self.seed = seed
        self.d = len(covs)  # covariates dimensionality (integer)

        self.om_A0 = gp_funcs["om_A0"]   # GP for the outcome model under treatment A=0
        self.om_A1 = gp_funcs["om_A1"]   # GP for the outcome model under treatment A=1
        self.w_sel = gp_funcs["w_sel"]   # GP for the selection score model P(S=1 | X)
        self.w_trt = gp_funcs["w_trt"]   # GP for the propensity score in OBS study P(A=1 | X, S=2)

        self.XX, self.UU = np.meshgrid(self.X, self.U)
        self.XU_flat = np.c_[self.XX.ravel(), self.UU.ravel()]

        self.prop_clip_lb = pasx["lb"]  #  exclude patients whose probability of treatment is < 0.1 
        self.prop_clip_ub = pasx["ub"]  #  exclude patients whose probability of treatment is > 0.9
        self.pas1 = pasx["trial"]  # probability of treatment assignment in the trial
        self.sbl, self.sbu = 0.5, 1


    def _generate_data(self):  
        # complete = False
        complete = True

        while not complete:
            try:
                df = pd.DataFrame(index=np.arange(self.n))
                df[self.covs] = 2 * np.random.rand(self.n, self.d) - 1  # Uniform[-1,1]

                df["P(S=0|X)"] = df.apply(lambda row: \
                            int(self.sbl < row["X"] < self.sbu) * np.clip(expit(self.w_sel(row["X"], row["U"])[0]), self.prop_clip_lb, self.prop_clip_ub), axis=1)
                df["S"] = np.array(df["P(S=0|X)"] < np.random.uniform(size=self.n), dtype=int)  # selection into trial via sampling from Bernoulli(P(S=1|X))

                rct_idx = random.sample(df.index[df["S"] == 1].tolist(), self.n_rct)
                tar_idx = random.sample(df.index[df["S"] == 0].tolist(), self.n_tar)

                complete=True
            except:
                print("Please wait patiently as we generate your synthetic nested trial data...")
                self.n = 2 * self.n

                
        np.random.seed(self.seed + 1)
        df = pd.DataFrame(index=np.arange(self.n))
        df[self.covs[0]] = 2 * np.random.rand(self.n, 1) - 1  # Uniform[-1,1]
        df[self.covs[1]] = np.zeros(self.n).reshape(-1, 1)

        df['Y0'] = df.apply(lambda row: self.om_A0(row["X"], row["U"])[0], axis=1)
        df['Y1'] = df.apply(lambda row: self.om_A1(row["X"], row["U"])[0], axis=1)
        df["A"] = np.array(self.pas1 > np.random.uniform(size=self.n_rct), dtype=int)  # random sampling 
        df['y'] = df['Y1'] * df['A'] + df['Y0'] * (1 - df['A'])

        return df
    

    def _generate_data_obs(self):  

        df = pd.DataFrame(index=np.arange(self.n_obs))
        np.random.seed(self.seed + 1)
        df[self.covs] = 2 * np.random.rand(self.n_obs, self.d) - 1

        df["P(A=1|X)"] = df.apply(lambda row: np.clip(expit(self.w_trt(row["X"], row["U"])[0]), self.prop_clip_lb, self.prop_clip_ub), axis=1)
        df["A"] = np.array(df["P(A=1|X)"] > np.random.uniform(size=self.n_obs), dtype=int) 

        df['Y0'] = df.apply(lambda row: self.om_A0(row["X"], row["U"])[0], axis=1)
        df['Y1'] = df.apply(lambda row: self.om_A1(row["X"], row["U"])[0], axis=1)

        df["y"] = df["Y1"] * df["A"] + df["Y0"] * (1 - df["A"])


        return df


    def get_df(self):
        np.random.seed(self.seed)
        self.df  = self._generate_data()
        self.df_obs = self._generate_data_obs()
        
        return self.df.copy(), self.df_obs.copy()
    

    def get_true_mean(self, print_res=True):  

        np.random.seed(self.seed + 1)
        df = pd.DataFrame(index=np.arange(self.n_MC))
        df[self.covs[0]] = 2 * np.random.rand(self.n_MC, 1) - 1 
        df[self.covs[1]] = np.zeros(self.n_MC).reshape(-1, 1)


        df['Y0'] = df.apply(lambda row: self.om_A0(row["X"], row["U"])[0], axis=1)
        df['Y1'] = df.apply(lambda row: self.om_A1(row["X"], row["U"])[0], axis=1)
        y1 = np.stack(np.array(df["Y1"]))
        y0 = np.stack(np.array(df["Y0"]))
        treatment_effect = y1 - y0

        true_ate, std_ate = treatment_effect.mean(),treatment_effect.std() / self.n_MC


        if print_res:
            print(f"MC sample sizes: n_MC = {self.n_MC}")
            # print(f"Target pop. mean: {mean_S0Y1:.3f} +- {std_S0Y1:.3f}")
            print(f"Trial pop. mean: {true_ate:.3f} +- {std_ate:.3f}")

        return true_ate, std_ate
    

    def get_some_stats(self):  
        A1_mean = self.df_obs["A"].mean()
        U_mean = self.df_obs[self.df_obs.A == 1]["U"].mean()
        X_mean = self.df_obs[self.df_obs.A == 1]["X"].mean()

        print(f"A mean (OBS study): {A1_mean:.2f}")
        print(f"U mean (OBS study): {U_mean:.2f}")
        print(f"X mean (OBS study): {X_mean:.2f}")
    
    
    def plot_om(self, save_dir):

        matplotlib.rcParams['pdf.fonttype'] = 42  # no type-3
        matplotlib.rcParams['ps.fonttype'] = 42

        Yp = self.om_A1(self.X, self.U)
        breakpoint()
        psx = np.clip(expit(self.w_sel(self.X, self.U)), self.prop_clip_lb, self.prop_clip_ub)
        pax = np.clip(expit(self.w_trt(self.X, self.U)), self.prop_clip_lb, self.prop_clip_ub)

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 15))

        contour = ax1.contourf(self.XX, self.UU, Yp, cmap='RdBu')
        ax1.set_xlabel('X')
        ax1.set_ylabel('U')
        plt.colorbar(contour, ax=ax1)

        ax2 = fig.add_subplot(322, projection='3d')
        ax2.plot_surface(self.XX, self.UU, Yp, cmap='RdBu')
        ax2.set_xlabel('X')
        ax2.set_ylabel('U')
        ax2.set_zlabel(f'E[Y1 | X, U]')

        avg_val_x = np.mean(Yp, axis=0)
        ax3.plot(self.X, avg_val_x)
        ax3.set_xlabel('X')
        ax3.set_ylabel(f'E[Y1 | X]')

        avg_val_u = np.mean(Yp, axis=1)
        ax4.plot(self.U, avg_val_u)
        ax4.set_xlabel('U')
        ax4.set_ylabel(f'E[Y1 | U]')

        psx_avg = np.mean(psx, axis=0)
        ax5.plot(self.X, psx_avg)
        ax5.set_xlabel('X')
        ax5.set_ylabel('P(S=1 | X)')

        ax6.plot(self.U, np.mean(pax, axis=1))
        ax6.set_xlabel('U')
        ax6.set_ylabel('P(A=1 | U, S=2 (OBS))')

        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, "true_dgp.jpg")
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()

        return avg_val_x, psx_avg

