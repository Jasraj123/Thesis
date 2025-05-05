import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = os.getcwd()

font_size = 15

label_dic = {
    "normal_aipw": r"$\hat{\tau}^{ AIPW}$" + r" ($\mathcal{D}^1$" + " only)",
    "normal_ppi": r"$\hat{\tau}^{ PP}$"+" (Ours)",
    "normal_obs": r"$\hat{\tau}^{ AIPW}$" + r" ($\mathcal{D}^2$" + " only)",
    'True ATE': r"$\tau$" + " (True in " + r"$\mathcal{D}^1$" + ")"
}

color_dic = {
    "normal_ppi": "navy",
    "normal_aipw": "gold",
    "normal_obs": "gray"
}
tick_dic = {
    "normal_ppi": 0,
    "normal_aipw": 0.4,
    "normal_obs": 0.8
}

name_list = ["normal_ppi", "normal_aipw", "normal_obs"]
seq_dic = {"normal_aipw": 0, "normal_ppi": 1, "normal_obs": 2}

alpha_list = [0.05, 0.1]
seed_list_dic = {
    "rct_data": [
        list(range(20)),  # Many more seeds for alpha = 0.05
        list(range(20))   # Many more seeds for alpha = 0.1
    ]
}

exp_name = "rct_data"
fig, ax = plt.subplots(2, 1, figsize=(3, 5))

for alpha_index in range(len(alpha_list)):
    read_dir = os.path.join(path, "code", exp_name, "experiments_u", "exp_results", f"alpha_{alpha_list[alpha_index]}", "unconfounding_0")
    
    true_ate = []
    seed_list = seed_list_dic[exp_name][alpha_index]
    
    for seed in seed_list:
        file_path = os.path.join(read_dir, f"estimates_{seed}.csv")
        if not os.path.exists(file_path):
            print(f"Warning: Missing file {file_path}")
            continue

        df = pd.read_csv(file_path)
        true_ate.append(df["true_ate"].iloc[0])

        tick_list = []
        label_list = []
        for i in range(len(name_list)):
            name = name_list[i]
            ate_est = df["ate_est"].to_list()[seq_dic[name]]
            ate_width = df["ate_ci_width"].to_list()[seq_dic[name]]
            ax[alpha_index].errorbar(ate_est, 
                                   tick_dic[name] + 0.02*seed_list.index(seed), 
                                   xerr=ate_width, 
                                   capsize=4, 
                                   color=color_dic[name], 
                                   label=label_dic[name], 
                                   marker='o',
                                   markersize=3.)

            tick_list.append(tick_dic[name])
            label_list.append(label_dic[name])

        ax[alpha_index].set_yticks([])
        ax[alpha_index].spines['top'].set_color('none')
        ax[alpha_index].spines['right'].set_color('none')
        ax[alpha_index].spines['left'].set_color('none')

    ax[alpha_index].axvline(np.mean(np.array(true_ate)), 
                           color='r', 
                           linestyle='dashed', 
                           label=label_dic['True ATE'])
    
    ax[alpha_index].set_ylabel(r"$\alpha = $" + str(alpha_list[alpha_index]), 
                              fontsize=font_size)

    handles, labels = ax[alpha_index].get_legend_handles_labels()

unique = dict(zip(labels, handles))
legend_dic = {}
for name in name_list:
    legend_dic[label_dic[name]] = unique[label_dic[name]]
legend_dic[label_dic['True ATE']] = unique[label_dic['True ATE']]

fig.legend(legend_dic.values(), legend_dic.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=6, fontsize=font_size-2)
plt.tight_layout()
plt.savefig(f"{path}/figs/results/{exp_name}/experiments_u.pdf", bbox_inches="tight")
