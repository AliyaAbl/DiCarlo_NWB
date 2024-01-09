# %% Verify signal covariance formulas


import numpy as np
import matplotlib.pyplot as plt

from nquality.statistics import get_within_session_metrics, get_cross_session_metrics

np.random.seed(0)
import collections

import utilz
import scipy.stats as ss


utilz.set_my_matplotlib_defaults()

# %% Set up ground truth parameters

baseline_x = 10
baseline_y = 50
x_gt = np.square(np.random.randn(100)) * 20 + baseline_x
y_gt = np.array(x_gt) + np.random.randn(len(x_gt)) * np.maximum(x_gt / 2, 4) + baseline_y
y_gt += baseline_y
y_gt = np.abs(y_gt)

plt.title('Ground truth')
plt.plot(x_gt, y_gt, '.')
utilz.unity(0, 100)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


def get_ground_truth(mu):
    gt = dict(
        noise_variance=mu.mean(),  # Poisson
        signal_variance=np.var(mu, ddof=0),
        baseline_spike_rate=mu.mean(),
        baseline_power=np.square(mu.mean()),

    )
    gt['SNR'] = gt['signal_variance'] / gt['noise_variance']
    return gt


def get_ground_truth_cross(mu_x, mu_y):
    gtx = get_ground_truth(mu_x)
    gty = get_ground_truth(mu_y)

    gt_cross = dict(
        squared_baseline_error=np.square(mu_x.mean() - mu_y.mean()),
        squared_signal_error=np.square((mu_x - mu_x.mean()) - (mu_y - mu_y.mean())).mean(),
        signal_covariance=((mu_x - mu_x.mean()) * (mu_y - mu_y.mean())).mean(),
        signal_correlation=ss.pearsonr(mu_x, mu_y)[0],
        noise_error=gtx['noise_variance'] + gty['noise_variance'],
    )

    niter = 10000
    single_rep_mse = np.zeros(niter)

    for i in range(niter):
        icur = np.random.choice(len(mu_x))
        x = np.random.poisson(mu_x[icur])
        y = np.random.poisson(mu_y[icur])
        single_rep_mse[i] = np.square(x - y)
    gt_cross['single_rep_mse'] = single_rep_mse.mean()
    return gt_cross


gt_cross = get_ground_truth_cross(x_gt, y_gt)
gt_x = get_ground_truth(x_gt)
gt_y = get_ground_truth(y_gt)


# %% Statistics


# %% Perform experiments
def simulate_data(mu: np.ndarray, nreps: int):
    dat = np.random.poisson(mu[:, None], size=(len(mu), nreps))

    mu_x = dat.mean(1)
    var_x = dat.var(1, ddof=1)

    return mu_x, var_x


n_image_samples = 10
X_nreps_per = 2
Y_nreps_per = X_nreps_per

niter = 1000
from tqdm import trange
import pandas as pd

dx = collections.defaultdict(list)
dy = collections.defaultdict(list)
dcross = collections.defaultdict(list)

for _ in trange(niter):
    i_samples = np.random.choice(len(x_gt), size=n_image_samples, replace=True)

    mu_x_gt_cur = x_gt[i_samples]
    mu_y_gt_cur = y_gt[i_samples]

    signal_mse_gt = np.square(mu_x_gt_cur - mu_y_gt_cur).mean()

    mu_x, var_x = simulate_data(mu_x_gt_cur, X_nreps_per)
    mu_y, var_y = simulate_data(mu_y_gt_cur, Y_nreps_per)

    xstats = get_within_session_metrics(mu_x, var_x, np.ones(len(mu_x), dtype=int) * X_nreps_per)
    ystats = get_within_session_metrics(mu_y, var_y, np.ones(len(mu_x), dtype=int) * Y_nreps_per)

    for k, v in xstats.items():
        dx[f'{k}'].append(v)
    for k, v in ystats.items():
        dy[f'{k}'].append(v)

    # % % Get cross stats
    cross_stats = get_cross_session_metrics(mu_x, var_x, np.ones(len(mu_x), dtype=int) * X_nreps_per,
                                            mu_y, var_y, np.ones(len(mu_x), dtype=int) * Y_nreps_per)
    for k, v in cross_stats.items():
        dcross[k].append(v)
dfx = pd.DataFrame(dx)
dfy = pd.DataFrame(dy)

dfcross = pd.DataFrame(dcross)

# %%

for k in dfcross.columns:
    plt.figure()
    plt.title(k)
    plt.hist(dfcross[k], bins=50, alpha=0.5, label='x')
    utilz.axvline(dfcross[k].mean(), color='k')
    utilz.axvline(gt_cross[k], color='b', lw=4, ls='-', alpha=0.4)
    plt.xlim([0, None])
    plt.show()
plt.close('all')

# %%

for k in dfx.columns:
    plt.hist(dfx[k], bins=50, alpha=0.5, label='x')

    plt.title(k)
    utilz.axvline(get_ground_truth(x_gt)[k], color='b', lw=4, ls='-', alpha=0.4)
    utilz.axvline(dfx[k].mean(), color='k')
    plt.xlim([0, None])
    plt.show()
plt.close('all')
