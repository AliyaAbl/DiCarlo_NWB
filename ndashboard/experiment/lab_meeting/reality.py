import utilz

utilz.set_my_matplotlib_defaults()

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
vec = np.random.randn(20)
vec_model = vec + np.random.randn(len(vec)) * 0.5
vmin = np.min(vec)
vmax = np.max(vec)

nmissing = int(0.5 * len(vec))
i_miss = np.random.choice(len(vec), nmissing, replace=False)
i_keep = np.array([i for i in range(len(vec)) if i not in i_miss])

fig, ax = plt.subplots(1, 3, figsize=(4, 1))
plt.sca(ax[0])
plt.imshow(vec[:, None], cmap='coolwarm', vmin=vmin, vmax=vmax)
utilz.remove_ticks(plt.gca())
utilz.remove_spines(plt.gca())

plt.sca(ax[1])

exp_mat = np.random.randn(len(vec), 10) * 0.5 + vec[:, None]


mu = np.mean(exp_mat, axis=1)
zeros = np.zeros_like(exp_mat) * np.nan
stacked= np.hstack([exp_mat, zeros, mu[:, None]])
stacked[i_miss, :] = np.nan
plt.imshow(stacked, cmap='coolwarm', alpha=1, vmin=vmin, vmax=vmax)
utilz.remove_ticks(plt.gca())
utilz.remove_spines(plt.gca())

plt.sca(ax[2])
vec_model[i_miss] = np.nan
plt.imshow(vec_model[:, None], cmap='coolwarm', vmin=vmin, vmax=vmax)
utilz.remove_ticks(plt.gca())
utilz.remove_spines(plt.gca())
utilz.savefig('reality.pdf', dpi=300)

plt.show()

# %%
np.random.seed(0)
vec = np.random.randn(20)
vec_model = vec + np.random.randn(len(vec)) * 0.5

plt.imshow(vec_model[:, None], cmap='coolwarm', vmin=vmin, vmax=vmax)
utilz.remove_ticks(plt.gca())
utilz.remove_spines(plt.gca())

plt.show()

# %%
np.random.seed(0)
vec = np.random.randn(20)
vec_model = vec + np.random.randn(len(vec)) * 0.5
vmin = np.min(vec)
vmax = np.max(vec)


fig, ax = plt.subplots(1, 3, figsize=(4, 1))
plt.sca(ax[0])
plt.imshow(vec[:, None], cmap='coolwarm', vmin=vmin, vmax=vmax)
utilz.remove_ticks(plt.gca())
utilz.remove_spines(plt.gca())

plt.sca(ax[1])

exp_mat = np.random.randn(len(vec), 10) * 0.5 + vec[:, None]
nmissing = int(0.5 * len(vec))

mu = np.mean(exp_mat, axis=1)
zeros = np.zeros_like(exp_mat) * np.nan
stacked= np.hstack([exp_mat, zeros, mu[:, None]])
i_miss = np.random.choice(len(vec), nmissing, replace=False)
stacked[i_miss, :] = np.nan
plt.imshow(stacked, cmap='coolwarm', alpha=1, vmin=vmin, vmax=vmax)
utilz.remove_ticks(plt.gca())
utilz.remove_spines(plt.gca())

plt.sca(ax[2])
vec_model[i_miss] = np.nan
plt.imshow(vec_model[:, None], cmap='coolwarm', vmin=vmin, vmax=vmax)
utilz.remove_ticks(plt.gca())
utilz.remove_spines(plt.gca())
utilz.savefig('reality_check.pdf', dpi=300)

plt.show()
