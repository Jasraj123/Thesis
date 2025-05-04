import numpy as np
import pandas as pd
import os, json, itertools, sys

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from rct_data.run_utils import *

def run_experiment(config_run, seed_list, methods_list, alpha):

    # alpha = config_run['data']['alpha']
    delta = config_run['data']['delta']
    data_seed = config_run['data']['data_seed']

    all_covs = config_run['data']['covairate_name']
    X_param = config_run['data']['X_range']
    X_range = np.linspace(X_param[0], X_param[1], X_param[2])
    U_param = config_run['data']['U_range']
    U_range = np.linspace(U_param[0], U_param[1], U_param[2])

    n_rct = config_run['data']['n_rct']
    n_obs = config_run['data']['n_obs']
    big_n_rct = len(seed_list) * n_rct
    big_n_obs = len(seed_list) * n_obs
    n_MC = config_run['data']['n_MC']

    om_A0_par_list = [config_run['data']['om_A0_par']]
    om_A1_par_list = [config_run['data']['om_A1_par']]
    w_sel_par_list = [config_run['data']['w_sel_par']]
    w_trt_par_list = [config_run['data']['w_trt_par'][i] for i in range(len(config_run['data']['w_trt_par']))]

    pasx = config_run['pasx']

    for ns, (om_A1_par, om_A0_par, w_sel_par, w_trt_par) in \
        enumerate(itertools.product(om_A1_par_list, om_A0_par_list, w_sel_par_list,  w_trt_par_list)):

        gp_params = {"om_A0_par": om_A0_par, "om_A1_par": om_A1_par, "w_sel_par": w_sel_par, "w_trt_par": w_trt_par}
        save_dir = config_run['relative_path']

        mean_trail, big_df_rct, big_df_obs = data_generation(gp_params, all_covs, big_n_rct, big_n_obs, n_MC, X_range, U_range, pasx, data_seed)

        for seed_index in range(len(seed_list)):
            df = pd.DataFrame()

            df_rct = big_df_rct.iloc[seed_index * n_rct : (seed_index + 1) * n_rct]
            df_obs = big_df_obs.iloc[seed_index * n_obs : (seed_index + 1) * n_obs]
            ate_est, ate_ci = sim_cases(seed_list[seed_index], df_rct, df_obs, alpha, delta)
        
            df["true_ate"] = [mean_trail for i in range(len(methods_list))]
            df["ate_est"] = ate_est
            df["ate_ci_width"] = [0.5 * (ate_ci[i][1] - ate_ci[i][0]) for i in range(len(methods_list))]

            if not os.path.exists(f"{save_dir}/exp_results/alpha_{alpha}/unconfounding_{ns}"):
                os.makedirs(f"{save_dir}/exp_results/alpha_{alpha}/unconfounding_{ns}")
            
            df.to_csv(f"{save_dir}/exp_results/alpha_{alpha}/unconfounding_{ns}/estimates_{seed_list[seed_index]}.csv")
   

if __name__ == "__main__":
    seed_list = [list(range(0, 10, 2)), list(range(1, 11, 2))]
    alpha_list = [0.05, 0.1]
    methods_list = ["normal_aipw", "normal_ppi", "normal_obs"]
    config_run = load_yaml("/rct_data/experiments_u/config")

    for alpha_index in [1]:
        run_experiment(config_run, seed_list[alpha_index], methods_list, alpha=alpha_list[alpha_index])
