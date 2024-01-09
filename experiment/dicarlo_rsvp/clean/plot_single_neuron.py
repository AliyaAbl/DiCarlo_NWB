import utilz

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

utilz.set_my_matplotlib_defaults()
import xarray as xr

import nquality.raw_data_template as data_template
import os

background_stimulus_id = 'Normalizer_26.png'


# %%
ds_quality = xr.load_dataset("ds_quality.nc")
import importlib

importlib.reload(data_template)
all_neuroids = ds_quality.neuroid.values

neuroid = all_neuroids[-3]
seed_session = ds_quality.session.values[-1]
nsessions = len(ds_quality.session.values)


ds_cross = xr.load_dataset('ds_cross.nc')
ds_seed = ds_cross.sel(session_j = seed_session)


i_seed_session = np.where(ds_quality.session.values == seed_session)[0][0]
xx_base = np.arange(nsessions)
xx = xx_base - i_seed_session


def plot(ds, v, *args, color=None, **kwargs):

    yy = ds[v].sel(stat='point')
    yy_low = ds[v].sel(stat='CI_low')
    yy_high = ds[v].sel(stat='CI_high')
    plt.plot(xx, yy, *args, color=color, **kwargs)
    utilz.fill_between_curves(xx, yy_low, yy_high, color=color, alpha=0.2)
    return xx, yy



for neuroid in tqdm(all_neuroids):

    ds_cur = ds_quality.sel(neuroid=neuroid)

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


    plt.sca(ax[3])
    fingerprint=ds_cur.mu_x.sel(stat = 'point').transpose('stimulus_id', 'session')
    nstimuli = len(fingerprint['stimulus_id'])
    plt.imshow(fingerprint, extent = [xx.min(), xx.max(), 0, nstimuli], aspect = 'auto', interpolation = 'nearest')
    plt.ylabel('Normalizer image')
    plt.tight_layout()


    plt.sca(ax[2])
    ds_cross_cur = ds_seed.sel(neuroid=neuroid)
    plot(ds_cross_cur, 'squared_signal_error', color='blue', )

    ds_cross_cur['signal_ub'] = ds_cur.signal_variance.isel(session = i_seed_session-1) + ds_cur.signal_variance
    plot(
        ds_cross_cur,
         'signal_ub', color='black',ls = ':', label = 'upper bound', )
    utilz.axhline(0, label = 'lower bound', )
    plt.legend()
    plt.ylabel(r'Baseline/noise-corrected signal error (Hz$^2$)')

    # Suffix
    plt.sca(ax[-1])
    plt.xlabel(r'$\Delta$ sessions')
    plt.tight_layout()
    utilz.savefig(os.path.join('dashboards', 'single_neuron', f'seed{seed_session}', f'{neuroid}.png'), dpi = 300)
    plt.close('all')


# %% Overall health dashboard

ds_cross_overall = ds_cross.sel(session_j =i_seed_session-1)

# %%

i_session = -5
plt.figure()
plt.plot(ds_quality.baseline_spike_rate.sel(stat = 'point').mean('session'),
         ds_quality.noise_variance.sel(stat = 'point').mean('session'),
         '.', alpha = 0.5, color = 'gray')
utilz.unity(10, 1000)
plt.xscale("log")
plt.yscale("log")
#plt.axis([-100, 5000, -100, 5000])
plt.show()

