import utilz
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm

utilz.set_my_matplotlib_defaults()

# %%
da_neural = xr.load_dataarray('da_neural.nc')
da_neural = da_neural * 10  # convert to Hz

# %%
da_neural = da_neural.assign_coords(
    session_datetime=da_neural.session_datetime,
    samp_on_us=da_neural.samp_on_us,
    animal=da_neural.animal,
    image_file_name=da_neural.image_file_name,
    stimulus_id=da_neural.image_file_name,
    subregion=(['electrode_id'], [v.split('_')[1] for v in da_neural.electrode_id.values]),

)

# %% Timestamping
unix_timestamps = []
session = []
for v, samp_on_us_delta in zip(da_neural.session_datetime.values, da_neural.samp_on_us.values):
    session_t = float(v) / 1e9 + 18000
    presentation_t = session_t + samp_on_us_delta / 1e6
    unix_timestamps.append(presentation_t)
    session.append(session_t)

da_neural = da_neural.assign_coords(
    unix_timestamp=(['presentation'], unix_timestamps),
    session_start_unix_timestamp=(['presentation'], session)
)

da_neural = da_neural.assign_coords(
    neuroid_id=da_neural.electrode_id,
)

da_neural = da_neural.swap_dims({'electrode_id': 'neuroid_id'})

# %% Get data by date
session_to_data = {}
for date, da_date in da_neural.groupby('session_start_unix_timestamp'):
    session_to_data[date] = da_date
all_sessions = sorted(session_to_data.keys())

# %%
import nquality.raw_data_template as data_template
import importlib

importlib.reload(data_template)

dlist = []
for session in tqdm(all_sessions):
    da_date = session_to_data[session]
    data = data_template.SessionNeuralData(
        da_presentation=da_date
    )

    dlist.append(data.ds_point.assign_coords(session=session))

ds_point = xr.concat(dlist, dim='session')
# %%
ds_point = ds_point.sortby(ds_point.session)
ds_point = ds_point.sel(neuroid_id=~(np.isnan(ds_point.mu_x).sum('stimulus_id')).any('session'))


# %% Session metrics

def get_within_session_metrics(ds_point):
    ds_quality = xr.Dataset()
    # % Get single-session quality metrics
    nstimuli = (~np.isnan(ds_point.mu_x)).sum('stimulus_id')
    ds_quality['baseline_spike_rate'] = ds_point.mu_x.mean('stimulus_id')
    ds_quality['signal_variance'] = ds_point.mu_x.var('stimulus_id', ddof=1) - (ds_point.var_x / ds_point.nreps).mean('stimulus_id')
    ds_quality['signal_variance'] = np.maximum(ds_quality['signal_variance'], 0)
    ds_quality['noise_variance'] = ds_point.var_x.mean('stimulus_id')
    ds_quality['baseline_power'] = np.square(ds_quality.baseline_spike_rate) - ds_point.mu_x.var('stimulus_id', ddof=1) / nstimuli
    ds_quality['SNR'] = np.clip(ds_quality.signal_variance, 0, None) / (ds_quality.noise_variance + 0.1)

    ds_quality = xr.merge([ds_point, ds_quality])
    return ds_quality

ds_point = get_within_session_metrics(ds_point)

ds_point = ds_point.sortby(ds_point.session)

# %% Make single neuron dashboard
i_neuroid = -2

ds_cur = ds_point.isel(neuroid_id = i_neuroid)


neuroid_id = str(ds_cur.neuroid_id.values)

signal_variance = ds_cur.signal_variance
noise_variance = ds_cur.noise_variance
baseline_power = ds_cur.baseline_power
SNR = ds_cur.SNR
xx = (ds_cur.session.values - ds_cur.session.values.min()) / (24 * 3600 )
xx = np.arange(len(xx))
root = True

if root:
    signal_variance = np.maximum(signal_variance, 0)
    signal_cur = np.sqrt(signal_variance)
    signal_cur.values[np.isnan(signal_cur.values)] = 0
    noise_cur = np.sqrt(noise_variance)
    baseline_cur = np.sqrt(baseline_power)
else:
    signal_cur = signal_variance
    noise_cur = noise_variance
    baseline_cur = baseline_power


fig, ax = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
plt.suptitle(f'{neuroid_id} Dashboard')

plt.sca(ax[1])
plt.imshow(
    ds_cur.mu_x.transpose('stimulus_id', 'session'),
    aspect='auto',
    extent=[xx.min(), xx.max(), 0, len(ds_point.stimulus_id)]
)
plt.title('Neuron\'s fingerprint over sessions')
# plt.colorbar().ax.set_ylabel('Hz')
plt.ylabel('Normalizer Image')
# plt.xlabel('Session')
plt.tight_layout()
plt.xticks([])

plt.sca(ax[0])
plt.title('Spike drive vs. day')

# Noise power
utilz.fill_between_curves(xseq=xx, ylb_seq=np.zeros(len(noise_cur)), yub_seq=noise_cur, alpha=0.6, color='red')
plt.plot(xx, noise_cur, label='noise', color='red')

# Signal power
utilz.fill_between_curves(xseq=xx, ylb_seq=np.zeros(len(baseline_cur)), yub_seq=signal_cur, alpha=0.6, color='blue')

plt.plot(xx, signal_cur, label='signal', lw=2, color='blue')

# Baseline power
utilz.fill_between_curves(xseq=xx, ylb_seq=-(baseline_cur), yub_seq=np.zeros(len(baseline_cur)), alpha=0.5, color='gray')
plt.plot(xx, -baseline_cur, label='baseline', lw=2, color='gray')


plt.legend(loc=(1, 0.5))
ax_cur = plt.gca()
utilz.axhline(0)
ax_cur.set_yticklabels([str(abs(x)) for x in ax_cur.get_yticks()])
if root:
    plt.ylabel(r'Root spike drive ($Hz$)')
else:
    plt.ylabel(r'Spike drive ($Hz^2$)')

bound = max(np.abs(plt.ylim()))
plt.ylim(-bound, bound)
utilz.remove_spines(remove_bottom=True)


plt.sca(ax[1])
plt.xlabel('Session')
plt.tight_layout()
plt.show()

# %% Make population-level dashboard

fig, ax = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
plt.suptitle(f'Population Dashboard')

plt.sca(ax[0])
plt.title('Signal - Noise variance')
dat = ds_point.signal_variance - ds_point.noise_variance

b = np.nanquantile(np.abs(dat), 0.9)

dat = dat.transpose('neuroid_id', 'session')
plt.imshow(dat, aspect='auto', interpolation='nearest', vmin=-b, vmax=b, cmap='coolwarm_r')
plt.colorbar()
plt.show()

# %% Get between-session metrics

all_sessions = sorted(ds_point.session.values)

ilist = []
for session_i in all_sessions:
    ds_i = ds_point.sel(session=session_i)

    jlist = []
    for session_j in all_sessions:
        ds_j = ds_point.sel(session=session_j)

        signal_variance_i = ds_i.signal_variance
        signal_variance_j = ds_j.signal_variance
        noise_variance_i = ds_i.noise_variance
        noise_variance_j = ds_j.noise_variance

        ds_cross = xr.Dataset(coords={'session_j': session_j})

        B_i = ds_i.baseline_power
        B_j = + ds_j.baseline_power
        b_i = ds_i.baseline_spike_rate
        b_j = ds_j.baseline_spike_rate
        squared_baseline_error = B_i + B_j - 2 * b_i * b_j

        nstimuli = len(ds_i.stimulus_id)
        signal_covariance = ((ds_i.mu_x - b_i) * (ds_j.mu_x - b_j)).mean('stimulus_id')
        if session_i == session_j:
            signal_covariance = signal_variance_i

        lam = 1
        single_rep_mse = squared_baseline_error + signal_variance_i + signal_variance_j - 2 * signal_covariance + noise_variance_i + noise_variance_j
        signal_mse = signal_variance_i + signal_variance_j - 2 * signal_covariance

        pearsonR = utilz.calc_correlation(obs_dim='stimulus_id', y_pred=ds_i.mu_x, y_actual=ds_j.mu_x)

        # plt.plot(signal_covariance, np.square(pearsonR), '.')
        # plt.show()
        ds_cross['PearsonR'] = pearsonR
        ds_cross['SingleRepMSE'] = single_rep_mse
        ds_cross['SquaredBaselineError'] = squared_baseline_error
        ds_cross['SignalCovariance'] = signal_covariance
        ds_cross['SignalMSE'] = signal_mse
        ds_cross['NoiseMSE'] = noise_variance_j + noise_variance_i

        jlist.append(ds_cross)
    ds = xr.concat(jlist, 'session_j')
    ilist.append(ds.assign_coords(session_i=session_i))

ds_cross_all = xr.concat(ilist, 'session_i')

# %%
