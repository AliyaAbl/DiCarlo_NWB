import utilz

import numpy as np
import matplotlib.pyplot as plt

utilz.set_my_matplotlib_defaults()
import xarray as xr

import nquality.raw_data_template as data_template
import os

background_stimulus_id = 'Normalizer_26.png'


# %%
ds_quality = xr.load_dataset("ds_quality.nc")
ds_cross = xr.load_dataset('ds_cross.nc')

ds_quality = ds_quality.assign_coords(subregion =(['neuroid'], [v.split('_')[1] for v in ds_quality.neuroid.values]))
ds_cross = ds_cross.assign_coords(subregion =(['neuroid'], [v.split('_')[1] for v in ds_cross.neuroid.values]))

# %%
import importlib

importlib.reload(data_template)
all_neuroids = ds_quality.neuroid.values

seed_session = ds_quality.session.values[-4]
nsessions = len(ds_quality.session.values)

ds_seed = ds_cross.sel(session_j = seed_session)
i_seed_session = np.where(ds_quality.session.values == seed_session)[0][0]
xx_base = np.arange(nsessions)
xx = xx_base - i_seed_session

def plot(ds, v, *args, color=None, **kwargs):

    yy = ds[v].sel(stat='point')
    yy_point = yy.mean('neuroid')
    yy_std = np.sqrt(yy.var('neuroid', ddof = 1) / (len(yy.neuroid.values)))
    #yy_low = ds[v].sel(stat='CI_low')
    #yy_high = ds[v].sel(stat='CI_high')
    utilz.errorbar(xx, yy_point, yerr = yy_std, *args, lw = 1, color=color, **kwargs)
    #utilz.fill_between_curves(xx, yy_low, yy_high, color=color, alpha=0.2)
    return xx, yy_point


all_subregions = sorted(set(ds_quality.subregion.values))
for subregion in all_subregions:
    ds_cur = ds_quality.sel(neuroid = ds_quality.subregion == subregion)
    ds_cross_cur = ds_cross.sel(neuroid = ds_cross.subregion == subregion, session_j = seed_session)

    fig, ax = plt.subplots(4, 1, figsize = (3, 9), sharex = True)

    plt.sca(ax[0])
    plt.title('Signal and Noise Variance')
    xx, _ = plot(ds_cur, 'signal_variance',color = 'blue', marker = '.', label='signal variance')
    plot(ds_cur, 'noise_variance',color = 'red', marker = '.', label='noise variance')
    utilz.axhline(0)
    plt.legend()
    plt.ylabel(r'Spike variance (Hz$^2$)')
    plt.tight_layout()


    plt.sca(ax[1])
    plt.title('Baseline Activity')
    plot(ds_cur, 'baseline_spike_rate', color = 'black', marker = '.', label='normalizer average')
    ds_cur['gray_mu_x'] = ds_cur.mu_x.sel(stimulus_id = background_stimulus_id)
    plot(ds_cur, 'gray_mu_x', color = 'gray', ls = '--', marker = '.', label='blank screen')
    plt.legend()
    plt.ylabel("Spike rate (Hz)")
    utilz.axhline(0)
    plt.tight_layout()

    plt.sca(ax[2])
    plot(ds_cross_cur, 'squared_signal_error', color='blue', )

    ds_cross_cur['signal_ub'] = ds_cur.signal_variance.isel(session = i_seed_session-1) + ds_cur.signal_variance
    plot(
        ds_cross_cur,
         'signal_ub', color='black',ls = ':', label = 'upper bound', )
    utilz.axhline(0, label = 'lower bound', )
    plt.legend()
    plt.ylabel(r'Baseline/noise-corrected signal error (Hz$^2$)')

    plt.sca(ax[3])
    plt.title('# of neurons with signal')
    sig = ds_cur.signal_variance.sel(stat="CI_low") > 0
    ngood = sig.sum('neuroid')
    plt.bar(xx, ngood)
    plt.ylim([0, len(ds_cur.neuroid.values)*1.1])
    utilz.axhline(len(ds_cur.neuroid.values))
    plt.ylabel('# neurons')

    # Suffix
    plt.sca(ax[-1])
    plt.xlabel(r'$\Delta$ sessions')
    plt.tight_layout()
    utilz.savefig(os.path.join('dashboards', 'subregion_wise', subregion, f'{subregion}_seed{i_seed_session}.png'), dpi = 300)
    plt.show()

    plt.close('all')


# %%
x = ds_quality.mu_x.sel(stat = 'point').mean('session').transpose('stimulus_id', 'neuroid')
X = np.array(x.values)
Xc = X - X.mean(0)
U, s, _ = np.linalg.svd(Xc, full_matrices=False)
embed = U[:, [0, 1]] # [image, 2]

# %%
neuroid = 1

for i_neuroid in range(len(ds_quality.neuroid.values)):
    xx = ds_quality.mu_x.isel(neuroid=i_neuroid).sel(stat='point')
    xx = xx - xx.mean('stimulus_id')
    x, y = (xx.values @ embed).T
    plt.plot(x, y, '.-', lw = 1, alpha = 0.5)
plt.show()

# %%
session_j = 16
plt.plot(ds_cross.squared_signal_error.isel(session_j = session_j).isel(neuroid = 0).sel(stat = 'point'))
plt.plot(ds_cross.squared_signal_error.isel(session_j = session_j).isel(neuroid = 0).sel(stat = 'CI_low'))
plt.plot(ds_cross.squared_signal_error.isel(session_j = session_j).isel(neuroid = 0).sel(stat = 'CI_high'))
utilz.axhline(0)
plt.show()
