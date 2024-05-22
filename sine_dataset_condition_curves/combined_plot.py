import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams.update({'font.size': 14})

legend_border=True

df = pd.read_csv("gamma_condition_numbers.csv", index_col=0)
gamma_labels = ["\infty", "P^\\ell_{\mathrm{i}}","P^\ell_{\mathrm{i}}/10", "P^\ell_{\mathrm{i}}/100"]

import matplotlib.pyplot as plt

"""gamma plot"""
fig, axs = plt.subplots(1, 3, figsize=(14, 5))
ax = axs[0]
for gamma_index in range(len(gamma_labels)):
    ax.plot(df.iloc[gamma_index], label=f"$\\gamma={gamma_labels[gamma_index]}$")
ax.set_xlabel("Epoch")
ax.set_ylabel("Condition Number")
ax.set_yscale("log")
ax.set_xticks([1,len(df.columns)//2, len(df.columns)-1])
ax.set_xlim(1,len(df.columns))
ax.set_ylim(1e3, 1e23)
ax.legend(prop={'size': 11}, frameon=legend_border, loc='lower right')

"""dof plot"""
df = pd.read_csv("dof_with_taylor_condition_numbers.csv", index_col=0)
dof_labels = ["0", "10^{-15}", "10^{-12}", "10^{-9}", "10^{-6}", "10^{-3}", "10^0", "10^{3}", "10^{6}"]
ax = axs[1]
for dof_index in range(len(dof_labels)):
    xs = df.iloc[dof_index].tolist()
    ax.plot(xs, label=f"$\\nu={dof_labels[dof_index]}$")
ax.set_xlabel("Epoch")
ax.set_yscale("log")
ax.set_xticks([1,len(df.columns)//2, len(df.columns)-1])
ax.set_xlim(1,len(df.columns))
ax.set_ylim(1e3, 1e23)
plt.setp(ax.get_yticklabels(), visible=False)
ax.legend(prop={'size': 11}, frameon=legend_border, loc='center right', bbox_to_anchor=(1.0, 0.5), ncols=2)

df = pd.read_csv("dof_without_taylor_condition_numbers.csv", index_col=0)
ax = axs[2]
for dof_index in range(len(dof_labels)):
    xs = df.iloc[dof_index].tolist()
    ax.plot(xs, label=f"$\\nu={dof_labels[dof_index]}$")
ax.set_xlabel("Epoch")
ax.set_yscale("log")
ax.set_xticks([1,len(df.columns)//2, len(df.columns)-1])
ax.set_xlim(1,len(df.columns))
ax.set_ylim(1e3, 1e23)
plt.setp(ax.get_yticklabels(), visible=False)
ax.legend(prop={'size': 11}, frameon=legend_border, loc='center right', bbox_to_anchor=(1.0, 0.4), ncols=2)
plt.tight_layout()
plt.subplots_adjust(wspace=0.07)
plt.savefig("condition_numbers.pdf", bbox_inches='tight', pad_inches=0.05)
plt.clf()


"""appendix 10k epochs"""
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df = pd.read_csv("gammas_condition_numbers_10000epochs.csv", index_col=0)
for gamma_index in range(len(gamma_labels)):
    ax.plot(df.iloc[gamma_index], label=f"$\\gamma={gamma_labels[gamma_index]}$")
ax.set_xlabel("Epoch")
ax.set_ylabel("Condition Number")
ax.set_yscale("log")
ax.set_xticks([1,len(df.columns)//2, len(df.columns)-1])
ax.set_xlim(1,len(df.columns))
ax.set_ylim(1e3, 1e23)
ax.legend(prop={'size': 11}, frameon=legend_border, loc='lower right')
plt.tight_layout()
plt.savefig("condition_numbers_10kepochs.pdf", bbox_inches='tight', pad_inches=0.05)
plt.clf()