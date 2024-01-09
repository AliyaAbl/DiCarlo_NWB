import utilz

from tqdm import tqdm
import matplotlib.pyplot as plt

utilz.set_my_matplotlib_defaults()
import xarray as xr

import nquality.raw_data_template as data_template
import nquality.quality_within_session as dashboard
import glob
import importlib
# %% Load data
neural_data_paths = glob.glob('/Users/mjl/PycharmProjects/ndashboard/experiment/dicarlo_rsvp/neural_data/da*.nc')
session_to_data = {}
for path in tqdm(neural_data_paths):
    da = xr.load_dataarray(path)
    session = float(da.session.values)
    session_to_data[session] = data_template.SessionNeuralData(
        da_presentation=da,
        presentation_dim='presentation',
        neuroid_dim = 'neuroid',
        stimulus_id_coord='stimulus_id',
        nboot=1000,
        boot_seed=None,
    )


# %%
all_sessions = sorted(session_to_data.keys())
importlib.reload(dashboard)
dlist = []
for session in tqdm(all_sessions):
    session_data = session_to_data[session]
    ds_session_quality = dashboard.estimate_within_session_quality(session_data=session_data)
    dlist.append(ds_session_quality)
ds_quality = xr.concat(dlist, dim='session').assign_coords(session=all_sessions)
ds_quality.to_netcdf('ds_quality.nc')

# %%
plt.plot(ds_quality.signal_variance.sel(stat = 'point'), ds_quality.pvalue_signal, '.', alpha = 0.05, color = 'gray')
utilz.axhline(0.01)
plt.xlim([-50, 1000])
plt.show()


# %% Get cross metrics
import itertools
ij_list = [[None for _ in range(len(all_sessions))] for _ in range(len(all_sessions))]
import importlib
importlib.reload(dashboard)
for (session_i, session_j) in tqdm(itertools.combinations_with_replacement(iterable = all_sessions, r = 2)):
    session_data_i = session_to_data[session_i]
    session_data_j = session_to_data[session_j]
    ds_cross = dashboard.compare_sessions(
        session_i=session_data_i,
        session_j=session_data_j,
        CI_width = 0.95,
        same_session= session_i == session_j,
    )

    i = all_sessions.index(session_i)
    j = all_sessions.index(session_j)
    ij_list[i][j] = ds_cross
    ij_list[j][i] = ds_cross

# %%
ilist = []
for i in range(len(all_sessions)):
    jlist = ij_list[i]
    ds_i = xr.concat(jlist, dim='session_j').assign_coords(session_j=all_sessions)

    ilist.append(ds_i)

ds_cross = xr.concat(ilist, 'session')
ds_cross.to_netcdf('ds_cross.nc')