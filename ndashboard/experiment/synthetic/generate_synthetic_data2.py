# %% Verify signal covariance formulas


import numpy as np
import matplotlib.pyplot as plt

from nquality.statistics import get_within_session_metrics, get_cross_session_metrics

np.random.seed(0)
import collections

import utilz
import scipy.stats as ss
import xarray as xr

utilz.set_my_matplotlib_defaults()

# %% Set up ground truth seed neuronal parameters
gt_seed = 0
RS_gt = np.random.RandomState(gt_seed)
num_neurons = 1
nimages = 200

_signal_scales = np.abs(np.random.randn(num_neurons) * 0.5)
_b_gt = np.abs(RS_gt.randn(num_neurons) * 10 + 30)
z = np.square(np.random.randn(nimages, num_neurons))
_mu_gt = _signal_scales * np.sqrt(_b_gt) * z + _b_gt[None, :]

mu_seed = xr.DataArray(_mu_gt, dims = ['stimulus_id', 'neuroid'], )




# %% Setup cross-day drift parameters
nsessions = 10

SIGNAL_DRIFT = 0.2
ADDITIVE_NOISE = 0.1
BASELINE_DRIFT = 0.1
# %%
mu_cur = mu_seed

dlist = []
RS_gt = np.random.RandomState(gt_seed)

for i_session in range(nsessions):

    # Signal drift
    mu_drift = np.sqrt(mu_cur) * np.abs(RS_gt.randn(*mu_cur.shape) * SIGNAL_DRIFT)
    mu_cur = np.abs(mu_drift + mu_cur)

    additive_noise_today = np.sqrt(mu_cur) * RS_gt.randn(*mu_cur.shape) * ADDITIVE_NOISE

    baseline_drift = np.sqrt(mu_cur.mean('stimulus_id')) * RS_gt.randn(len(mu_cur.neuroid)) * BASELINE_DRIFT

    mu_cur = mu_cur + additive_noise_today + baseline_drift

    ds_session = xr.Dataset()
    ds_session['mu'] = mu_cur
    ds_session['noise_variance'] = ds_session.mu.mean('stimulus_id')  # Poisson noise model
    ds_session['signal_variance'] = ds_session.mu.var('stimulus_id', ddof=1)
    ds_session['baseline_spike_rate'] = ds_session.mu.mean('stimulus_id')
    ds_session['baseline_power'] = ds_session.baseline_spike_rate ** 2
    ds_session['SNR'] = ds_session.signal_variance / ds_session.noise_variance
    dlist.append(ds_session.assign_coords(session = i_session))

ds_gt = xr.concat(dlist, dim = 'session')

# Get cross metrics


# %%

def get_ground_truth_cross(ds_gt ):
    ilist =[]
    for i in range(ds_gt.session.size):
        jlist = []
        for j in range(ds_gt.session.size):
            mu_i = ds_gt.mu.isel(session = i)
            mu_j = ds_gt.mu.isel(session = j)

            b_i = mu_i.mean('stimulus_id')
            b_j = mu_j.mean('stimulus_id')

            noise_i = mu_i.mean('stimulus_id') # Poisson noise model
            noise_j = mu_j.mean('stimulus_id')

            squared_signal_error=np.square((mu_i - b_i) - (mu_j - b_j)).mean('stimulus_id')
            squared_baseline_error = np.square(b_i - b_j)
            gt_ij = xr.Dataset(
                data_vars = dict(
                squared_baseline_error=squared_baseline_error,
                squared_signal_error=squared_signal_error,
                signal_covariance=((mu_i - b_i) * (mu_j - b_j)).mean(),
                signal_correlation=utilz.calc_correlation('stimulus_id', mu_i, mu_j),
                noise_error=noise_i + noise_j,
                    single_rep_mse= noise_i + noise_j + squared_signal_error + squared_baseline_error,
            )
            )


            gt_ij = gt_ij.assign_coords(session_j = j)

            jlist.append(gt_ij)
        ilist.append(xr.concat(jlist, 'session_j').assign_coords(session_i = i))

    ds_cross = xr.concat(ilist, 'session_i')


    return ds_cross


gt_cross = get_ground_truth_cross(ds_gt)


# %% Statistics
neuroid = 0

plt.imshow(gt_cross.isel(neuroid = neuroid).squared_baseline_error, vmin = 0)
plt.colorbar()
plt.show()

# %% Perform experiments
def simulate_data(mu: xr.DataArray, nreps: int):
    """

    :param mu: (stimulus, neurons)
    :param nreps:
    :return:
    """
    dat = np.random.poisson(mu[..., None], size=mu.shape + (nreps,))
    mu_x = dat.mean(-1)
    var_x = dat.var(-1, ddof=1)
    return mu_x, var_x

n_image_samples = 86
n_reps = 20


niter = 1000
from tqdm import trange
import pandas as pd
import itertools

dwithin = collections.defaultdict(list)
dcross = []

for _ in trange(niter):
    i_samples = np.random.choice(len(ds_gt.stimulus_id), size=n_image_samples, replace=True)
    mu_gt_simulation = ds_gt.mu.isel(stimulus_id = i_samples)
    mu_gt_simulation = mu_gt_simulation.transpose('stimulus_id','session', 'neuroid', )

    mu_x, var_x = simulate_data(mu_gt_simulation.values, n_reps)

    # mu_x: 'stimulus_id','session', 'neuroid',
    within_day_metrics = get_within_session_metrics(mu_x, var_x, np.ones(mu_x.shape, dtype=int) * n_reps)
    for k in within_day_metrics:
        dwithin[k].append(within_day_metrics[k])

    # % % Get cross stats
    cross_day_outcomes = collections.defaultdict(list)
    for i_day, j_day in itertools.combinations_with_replacement(range(nsessions), r = 2):
        mu_i, var_i = mu_x[:, i_day], var_x[:, i_day]
        mu_j, var_j = mu_x[:, j_day], var_x[:, j_day]
        cross_stats = get_cross_session_metrics(mu_i, var_i, np.ones(mu_i.shape, dtype=int) * n_reps,
                                                mu_j, var_j, np.ones(mu_i.shape, dtype=int) * n_reps)
        if i_day == j_day:
            for k in cross_stats:
                cross_stats[k]*=np.nan
        cross_day_outcomes[(i_day, j_day)] = (cross_stats)
        cross_day_outcomes[(j_day, i_day)]= (cross_stats)
    dcross.append(cross_day_outcomes)

# %%
data_vars= {}
for k in dwithin:
    data_vars[k] = (['sim_iter', 'session', 'neuroid'], dwithin[k])
ds_within_sim = xr.Dataset(data_vars)


neuroid = 0

ds_cur = ds_within_sim.isel(neuroid = neuroid)

s = 2
fig, ax = plt.subplots(1, len(ds_cur.data_vars), figsize = (len(ds_cur.data_vars)*s, 1*s))
c=0
for v in ds_cur.data_vars:
    plt.sca(ax[c])
    plt.title(v)
    c+=1

    yy = ds_cur[v]
    xx = (ds_cur.session.values)
    plt.plot(xx,ds_gt[v].isel(neuroid = neuroid), lw = 3, color = 'blue', alpha = 0.3, label = 'ground truth')
    plt.plot(xx, yy.mean('sim_iter'), lw = 1, color = 'k', label = 'simulation expected value')
    utilz.fill_between_curves(xx, ylb_seq = yy.quantile(0.05, 'sim_iter').values,yub_seq = yy.quantile(0.95, 'sim_iter').values, lw=1, color='k', label = '95% sim. interval')
    plt.ylim([0, None])
plt.legend(loc = (1, 0))
plt.tight_layout()

plt.show()


# %% Assemble cross day metrics (sim, session_i, session_j, neuroid)
simlist = []
from tqdm import tqdm
for d in tqdm(dcross):
    ilist = []
    for i_day in range(nsessions):
        jlist = collections.defaultdict(list)
        for j_day in range(nsessions):
            dat = d[(i_day, j_day)]
            v = {}
            for k in dat:
                jlist[k] .append ( dat[k])
        for k in jlist:
            jlist[k] = (['session_j', 'neuroid'], jlist[k])
        ds_j = xr.Dataset(jlist, coords = dict(session_j = range(nsessions)))
        ilist.append(ds_j.assign_coords(session_i = i_day))
    ds_i = xr.concat(ilist, 'session_i')
    simlist.append(ds_i)
ds_cross_sim = xr.concat(simlist, 'sim_iter')

# %%
neuroid = 0
seed_session = 4
ds_cur = ds_cross_sim.isel(neuroid = neuroid, session_i = seed_session)
gt_cur = gt_cross.isel(neuroid = neuroid, session_i = seed_session)
s = 2
fig, ax = plt.subplots(1, len(ds_cur.data_vars), figsize = (len(ds_cur.data_vars)*s, 1*s))
c=0
for v in ds_cur.data_vars:
    plt.sca(ax[c])
    plt.title(v)
    c+=1

    yy = ds_cur[v]
    xx = (ds_cur.session_j.values)
    plt.plot(xx,gt_cur[v], lw = 3, color = 'blue', alpha = 0.3, label = 'ground truth')
    plt.plot(xx, yy.mean('sim_iter'), lw = 1, color = 'k', label = 'simulation expected value')
    utilz.fill_between_curves(xx, ylb_seq = yy.quantile(0.05, 'sim_iter').values,yub_seq = yy.quantile(0.95, 'sim_iter').values, lw=1, color='k', label = '95% sim. interval')
    plt.ylim([None, None])
plt.legend(loc = (1, 0))
plt.tight_layout()

plt.show()

# %%