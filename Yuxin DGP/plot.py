import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys, os, itertools

path = os.path.dirname(os.getcwd())
sys.path.append(path)

font_size = 15

label_dic = {
             "normal_aipw": r"$\hat{\tau}^{ AIPW}$" + r" ($\mathcal{D}^1$" + " only)", 
             "normal_ppi": r"$\hat{\tau}^{ PP}$"+" (Ours)", 
             "normal_obs": r"$\hat{\tau}^{ AIPW}$" + r" ($\mathcal{D}^2$" + " only)" ,
             'True ATE': r"$\tau$" + " (True in " + r"$\mathcal{D}^1$" + ")"}

color_dic = {"normal_ppi": "navy", "normal_aipw": "gold", "normal_obs": "gray"}
tick_dic = {"normal_ppi": 0, "normal_aipw": 0.4, "normal_obs": 0.8}

name_list = ["normal_ppi", "normal_aipw", "normal_obs"]
# Fix the indices to match the CSV file row order
# CSV order appears to be: normal_aipw, normal_ppi, normal_obs
seq_dic = {"normal_aipw": 0, "normal_ppi": 1, "normal_obs": 2}

# Define title dictionary
title_dic = {0: "Experiment Results"}

alpha_list = [0.05, 0.1]
seed_list_dic = {
                    "rct_data": [list(range(0, 10, 2)), list(range(1, 11, 2))],
                }


exp_name = "rct_data" 
fig, ax = plt.subplots(2, 1, figsize=(3, 5))

for alpha_index in [0, 1]:
    subplot_index = 0
    read_dir = f"/Users/RajNeeti/Desktop/Thesis/Yuxin DGP/exp_results/alpha_{alpha_list[alpha_index]}"
    true_ate = []
    seed_list = seed_list_dic[exp_name][alpha_index]
    print(exp_name, alpha_list[alpha_index], seed_list)
    for seed in seed_list:
        try:
            df = pd.read_csv(f"{read_dir}/unconfounding_{subplot_index}/estimates_{seed}.csv")
            true_ate.append(df["true_ate"].to_list()[0])

            tick_list = []
            label_list = []
            for i in range(len(name_list)):
                name = name_list[i]
                ate_est = df["ate_est"].to_list()[seq_dic[name]]
                ate_width = df["ate_ci_width"].to_list()[seq_dic[name]]
                # Update to use 1D axis array instead of 2D
                ax[alpha_index].errorbar(ate_est, tick_dic[name]+0.05*seed_list.index(seed), xerr=ate_width, capsize=4, color=color_dic[name], label = label_dic[name], marker='o',markersize=5.)

                tick_list.append(tick_dic[name])
                label_list.append(label_dic[name])

            ax[alpha_index].set_yticks([])
            ax[alpha_index].spines['top'].set_color('none')
            ax[alpha_index].spines['right'].set_color('none')
            ax[alpha_index].spines['left'].set_color('none')
        except Exception as e:
            print(f"Error processing file for seed {seed}: {e}")

    if true_ate:
        ax[alpha_index].axvline(np.mean(np.array(true_ate)), color='r', linestyle= 'dashed', label = label_dic['True ATE'])

    if alpha_index == 0:
        ax[alpha_index].set_title(title_dic[subplot_index], fontsize=font_size)
    
    ax[alpha_index].set_ylabel(r"$\alpha = $" + str(alpha_list[alpha_index]), fontsize=font_size)

    handles, labels = ax[alpha_index].get_legend_handles_labels()

unique = dict(zip(labels, handles))
legend_dic = {}
for name in name_list:
    if label_dic[name] in unique:
        legend_dic[label_dic[name]] = unique[label_dic[name]]
    
if label_dic['True ATE'] in unique:
    legend_dic[label_dic['True ATE']] = unique[label_dic['True ATE']]

fig.legend(legend_dic.values(), legend_dic.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=font_size-2)
plt.tight_layout()

# Ensure directories exist before saving
os.makedirs(f"{path}/figs/results/{exp_name}", exist_ok=True)
plt.savefig(f"/Users/RajNeeti/Desktop/Thesis/Yuxin DGP/experiments_results.pdf", bbox_inches="tight")
print(f"Plot saved to /Users/RajNeeti/Desktop/Thesis/Yuxin DGP/experiments_results.pdf")
