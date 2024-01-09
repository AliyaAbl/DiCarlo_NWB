import collections

import utilz
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm

utilz.set_my_matplotlib_defaults()
import time

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

da_neural = da_neural.swap_dims({'electrode_id': 'neuroid'})
da_neural['neuroid'] = da_neural.neuroid_id

# %% Get data by date
session_to_data = {}
savedir = './neural_data'
import os
for date, da_date in da_neural.groupby('session_start_unix_timestamp'):
    print(date)
    savepath = os.path.join(savedir, f'da_{date}.nc')
    utilz.make_savepath_dir(savepath)
    da_date = da_date.assign_coords(session = date)
    da_date.to_netcdf(os.path.join(savepath))

