import numpy as np
import collections

# %%
np.random.seed(0)
image_distribution = np.random.randn(5)
image_sigmas = np.random.randn(len(image_distribution))

b_gt = np.mean(image_distribution)

# %%

nimage_samples = 10
nreps_per = np.random.randint(2, 10, size=nimage_samples)

# %%

niter = 10000
from tqdm import trange

d = collections.defaultdict(list)
for _ in trange(niter):
    i_image_samples = np.random.choice(len(image_distribution), size=nimage_samples, replace=True)
    image_sigmas_cur = image_sigmas[i_image_samples]
    image_mu_cur = image_distribution[i_image_samples]

    mu_hats = np.zeros(nimage_samples)
    mu_hat_vars = np.zeros(nimage_samples)
    for i, (mu, sigma, reps) in enumerate(zip(image_mu_cur, image_sigmas_cur, nreps_per)):
        samps = np.random.randn(reps) * sigma + mu
        mu_hats[i] = samps.mean()
        mu_hat_vars[i] = np.var(samps, ddof=1) / reps

    b_hat = np.mean(mu_hats)
    b_var1 = np.var(mu_hats, ddof=1) / nimage_samples

    signal_var = np.var(mu_hats, ddof = 1) - np.mean(mu_hat_vars)
    b_var2 = signal_var / nimage_samples + 1 / (nimage_samples ** 2) * np.sum(mu_hat_vars)


    d['bhat'].append(b_hat)
    d['signal_var'].append(signal_var)
    d['var1_bhat'].append(b_var1)  # Bessel's formula for SVM
    d['var2_bhat'].append(b_var2)  # Bessel's formula for SVM

import xarray as xr

ds = xr.Dataset(d)
# %%
import matplotlib.pyplot as plt
import utilz

utilz.set_my_matplotlib_defaults()
plt.hist(ds.bhat, bins=100)
utilz.axvline(b_gt, color='red')
plt.show()

# %%
plt.hist(ds.var1_bhat, bins=100)
gt_estimator_var = np.var(ds.bhat, ddof=1)
var1_mean = ds.var1_bhat.mean()
utilz.axvline(gt_estimator_var, color='red')
utilz.axvline(var1_mean, color='blue', label='Bessel')
print('gt', gt_estimator_var)
print('Bessel', var1_mean)
plt.show()


# %% Signal variance
plt.hist(ds.signal_var, bins=100)

utilz.axvline(np.mean(ds.signal_var), color = 'red')
utilz.axvline(np.var(image_distribution), color = 'blue', label = 'gt')
plt.show()


#%%
np.random.seed(0)
x = np.random.randn(100)
y = np.random.randn(100)
mux = x.mean()
muy = y.mean()
cov1 = (x * y).mean() - mux * muy
cov2 = ((x - mux) * (y - muy)).mean()
cov3 = ((x - mux) * (y - muy)).sum() / (len(x) - 1)