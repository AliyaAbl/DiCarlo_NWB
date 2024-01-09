import collections

import utilz
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import os

utilz.set_my_matplotlib_defaults()
import time

# %%
ds_point = xr.load_dataset('ds_point.nc')

# %%
from nquality.statistics import get_within_session_metrics, get_cross_session_metrics

"""
:param mu_x: (stimulus_id, *). An unbiased estimate (the sample mean, over nreps) of the mean of the spiking rate, given a stimulus. Nan values if nreps is 0.
:param var_x: (stimulus_id, *). An unbiased estimator (the Bessel-corrected sample variance, over nreps) of the variance in the spike rate, given a stimulus. Nan values if nreps <= 1.
:param nreps: (stimulus_id, *). The number of repetitions used to form those estimates; >= 0.
:return: dict
"""
dim_order = 'stimulus_id', 'neuroid', 'session'
within_session_metrics = get_within_session_metrics(
    mu_x=ds_point.mu_x.transpose(*dim_order).values,
    var_x=ds_point.var_x.transpose(*dim_order).values,
    nreps=ds_point.nreps.transpose(*dim_order).values.astype(int),
)
data_vars = {}
for k in within_session_metrics:
    data_vars[k] = (list(dim_order[1:]), within_session_metrics[k])
ds_within = xr.Dataset(data_vars, coords={c: ds_point.coords[c] for c in dim_order})

# %% Cross-session metrics
dim_order = 'stimulus_id', 'neuroid'

import itertools

nsessions = (len(ds_point.session))

ij_metrics = [[None for _ in range(nsessions)] for _ in range(nsessions)]
for i_session, j_session in tqdm(itertools.combinations_with_replacement(range(nsessions), r=2)):
    ds_i = ds_point.isel(session=i_session)
    ds_j = ds_point.isel(session=j_session)

    mu_i = ds_i.mu_x.transpose(*dim_order).values
    var_i = ds_i.var_x.transpose(*dim_order).values
    nreps_i = ds_i.nreps.transpose(*dim_order).values.astype(int)
    mu_j = ds_j.mu_x.transpose(*dim_order).values
    var_j = ds_j.var_x.transpose(*dim_order).values
    nreps_j = ds_j.nreps.transpose(*dim_order).values.astype(int)
    cross_metrics = get_cross_session_metrics(
        mu_x=mu_i,
        var_x=var_i,
        nreps_x=nreps_i,
        mu_y=mu_j,
        var_y=var_j,
        nreps_y=nreps_j,
        same_session = i_session == j_session,
    )
    ij_metrics[i_session][j_session] = cross_metrics
    ij_metrics[j_session][i_session] = cross_metrics

# %%
dlist = []
for i_session in range(nsessions):
    ilist = ij_metrics[i_session]
    data_vars = collections.defaultdict(list)
    for e in ilist:
        for k in e:
            data_vars[k].append(e[k])
    for k in e:
        data_vars[k] = (['session_j', 'neuroid'], (data_vars[k]))
    ds_i = xr.Dataset(data_vars, coords = {'session_j': ds_point.session.values, 'neuroid': ds_point.neuroid.values})
    dlist.append(ds_i)

ds_cross = xr.concat(dlist, dim = 'session').assign_coords(session = ds_point.session.values)

# %%
ds_dashboard = xr.merge([ds_within, ds_cross, ds_point])
# %% Make single neuron dashboard

background_stimulus_id = 'Normalizer_26.png'
from tqdm import trange

i_seed_day = -1
for i_neuroid in trange(len(ds_point.neuroid)):
    ds_cur = ds_dashboard.isel(neuroid=i_neuroid)

    neuroid_id = str(ds_cur.neuroid.values)

    xx = ds_cur.session.values
    xx = xx - np.min(xx)
    xx = xx / (24 * 3600)
    fig, ax = plt.subplots(4, 1, figsize=(2, 5), sharex=True)

    plt.sca(ax[0])
    plt.title('Single Rep MSE')
    cross_cur = ds_cur.isel(session = i_seed_day)
    sr_MSE = ds_cur.single_rep_mse.isel(session = i_seed_day).values
    sr_MSE_floor = ds_cur.isel(session = i_seed_day).noise_variance

    # SR MSE a constant model of the day's baseline firing rate would achieve
    sr_MSE_ceiling = ds_cur.isel(session=i_seed_day).noise_variance + ds_cur.isel(session=i_seed_day).signal_variance
    plt.plot(xx, sr_MSE, color = 'black', lw = 1, marker = '.')
    plt.plot(xx[i_seed_day], (np.maximum(sr_MSE[i_seed_day], 0)), marker = 'o', color = 'white', mew = 1, mec = 'k')
    utilz.axhline(sr_MSE_floor, label = 'noise floor', ls = ':')
    plt.ylabel(f'Total error ($Hz^2$)')

    plt.sca(ax[1])
    plt.title('Noise Error')
    nseq = cross_cur.noise_error.values
    utilz.errorbar(xx, (np.maximum(nseq, 0)), yerr = None, lw = 1, color = 'red')
    plt.plot(xx[i_seed_day], (np.maximum(nseq[i_seed_day], 0)), marker = 'o', color = 'white', mew = 1, mec = 'k')
    utilz.axhline(ds_cur.noise_variance.values[i_seed_day])
    plt.ylabel(f'Noise error ($Hz^2$)')

    plt.ylim([0, None])

    plt.sca(ax[2])
    plt.title('Signal Error')
    plt.plot(xx[i_seed_day], cross_cur.squared_signal_error.values[i_seed_day], marker = 'o', color = 'white', mew = 1, mec = 'k')
    plt.plot(xx, cross_cur.squared_signal_error)
    utilz.axhline(0)
    utilz.axhline(ds_cur.signal_variance.values[i_seed_day])
    plt.ylabel(f'Signal error ($Hz^2$)')

    plt.sca(ax[3])
    plt.title('Baseline Drift')
    bseq = ds_cur.baseline_spike_rate.values
    bseq_std = ds_cur.baseline_spike_rate_std.values
    background_seq = ds_cur.mu_x.sel(stimulus_id = background_stimulus_id)
    background_seq_std = np.sqrt(ds_cur.var_x.sel(stimulus_id=background_stimulus_id) / ds_cur.nreps.sel(stimulus_id=background_stimulus_id))
    utilz.errorbar(xx, bseq, yerr = bseq_std, lw = 1, color = 'black')
    utilz.errorbar(xx, background_seq, yerr=background_seq_std, lw=1, ls = ':', color = 'gray', alpha = 0.5, label = 'background')
    plt.plot(xx[i_seed_day], bseq[i_seed_day], marker = 'o', color = 'white', mew = 1, mec = 'k')
    #utilz.axhline(bseq[i_seed_day])
    plt.ylabel('Average firing rate (Hz)')
    plt.ylim([0, None])

    plt.tight_layout()
    #plt.show()


    #
    utilz.savefig(os.path.join(f'./dashboards/single_neuron2/{neuroid_id}.png'))
    plt.close('all')


# %% Get multineuron dashboard
i_seed_day = -8

yy = ds_dashboard.single_rep_mse.isel(session = i_seed_day)
xx = yy.session_j
xx = xx - np.min(xx)
xx = xx / (24 * 3600)

yy = ds_dashboard.isel(session = i_seed_day)
yy_point = yy.mean('neuroid')
yy_std = np.sqrt(yy.var('neuroid', ddof = 1) / len(yy.neuroid))

v = 'single_rep_mse'
v = 'squared_signal_error'
#utilz.errorbar(xx, yy_point[v], yerr = yy_std[v], lw = 1)
#utilz.errorbar(xx, yy_point['noise_error'], yerr = yy_std['noise_error'], lw = 1)
ycur = yy_point[v]
yy_std_cur = yy_std[v]


utilz.errorbar(xx, ycur, yerr = yy_std, lw = 0)
utilz.axhline(0)
plt.show()
