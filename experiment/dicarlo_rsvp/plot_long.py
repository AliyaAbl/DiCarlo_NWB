# Plot a projection

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
import nquality
