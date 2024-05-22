import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams.update({'font.size': 14})  

df = pd.read_csv("gamma_condition_numbers.csv", index_col=0)
gamma_labels = ["\infty", "P^\\ell_{\mathrm{i}}","P^\ell_{\mathrm{i}}/10", "P^\ell_{\mathrm{i}}/100"]

fig, ax = plt.subplots(figsize=(5, 4))

#plot the condition numbers, using df
for gamma_index in range(len(gamma_labels)):
    ax.plot(df.iloc[gamma_index], label=f"$\gamma={gamma_labels[gamma_index]}$")
ax.set_xlabel("Epoch")
ax.set_ylabel("Condition Number")
ax.set_yscale("log")
ax.set_xticks([1,len(df.columns)//2, len(df.columns)-1])
ax.set_xlim(1,len(df.columns))
ax.set_ylim(df.min().min(), df.max().max())
ax.legend()

plt.tight_layout()
plt.show()

df = pd.read_csv("dof_without_taylor_condition_numbers.csv", index_col=0)
dof_labels = ["0", "10^{-15}", "10^{-12}", "10^{-9}", "10^{-6}", "10^{-3}", "1", "10^{3}", "10^{6}"]

fig, ax = plt.subplots(figsize=(5, 4))

#plot the condition numbers, using df
for dof_index in range(len(dof_labels)):
    ax.plot(df.iloc[dof_index], label=f"$nu={dof_labels[dof_index]}$")
ax.set_xlabel("Epoch")
ax.set_ylabel("Condition Number")
ax.set_yscale("log")
ax.set_xticks([1,len(df.columns)//2, len(df.columns)-1])
ax.set_xlim(1,len(df.columns))
ax.set_ylim(df.min().min(), df.max().max())
ax.legend()

plt.tight_layout()
plt.show()

df = pd.read_csv("dof_with_taylor_condition_numbers.csv", index_col=0)
dof_labels = ["0", "10^{-15}", "10^{-12}", "10^{-9}", "10^{-6}", "10^{-3}", "1", "10^{3}", "10^{6}"]

fig, ax = plt.subplots(figsize=(5, 4))

#plot the condition numbers, using df
for dof_index in range(len(dof_labels)):
    ax.plot(df.iloc[dof_index], label=f"$nu={dof_labels[dof_index]}$")
ax.set_xlabel("Epoch")
ax.set_ylabel("Condition Number")
ax.set_yscale("log")
ax.set_xticks([1,len(df.columns)//2, len(df.columns)-1])
ax.set_xlim(1,len(df.columns))
ax.set_ylim(df.min().min(), df.max().max())
ax.legend()

plt.tight_layout()
plt.show()