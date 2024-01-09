import ndashboard.nquality.raw_data_template as data_template
from typing import List
import ndashboard.nquality.quality_within_session as quality_within_session
from tqdm import tqdm
import xarray as xr
import ndashboard.nquality.statistics as stats
import numpy as np
#import utilz
import collections

class LongitudinalQuality(object):
    def __init__(
            self,
            session_data: List[data_template.SessionNeuralData],
            nboot: int = 1000,
            boot_seed: int = 0,
            common_neurons_only = True,
    ):

        self.session_seq = []


        for dat in session_data:

            sess = quality_within_session.Session(session_data=dat, boot_seed=boot_seed, nboot=nboot)
            self.session_seq.append(sess)

    @property
    def ds_point(self):
        if not hasattr(self, '_ds_point'):
            dlist = []
            for sess in tqdm(self.session_seq, desc='Concatenating session point data'):
                ds_point = sess.ds_point.assign_coords(timestamp_start=sess.timestamp_start)
                dlist.append(ds_point)
            ds_point = xr.concat(dlist, dim='session')
            ds_point = ds_point.assign_coords(
                session=np.arange(len(self.session_seq)),
            )
            self._ds_point = ds_point
        return self._ds_point

    @property
    def ds_quality(self):
        if not hasattr(self, '_ds_quality'):

            # Within session quality
            wlist = []
            for sess in self.session_seq:
                ds_within_session_quality = sess.ds_quality.assign_coords(
                    timestamp_start=sess.timestamp_start,
                )
                wlist.append(ds_within_session_quality)
            ds_within_session_quality = xr.concat(wlist, dim='session').assign_coords(
                session=np.arange(len(self.session_seq)),
            )

            # Cross session quality
            ds_cross = self._compare_all_sessions()
            ds_cross = ds_cross.assign_coords(
                session=np.arange(len(self.session_seq)),
                timestamp_start=(['session'], [sess.timestamp_start for sess in self.session_seq]),
                session_j=np.arange(len(self.session_seq)),
                timestamp_start_j=(['session_j'], [sess.timestamp_start for sess in self.session_seq]),
            )
            self._ds_quality = xr.merge([ds_within_session_quality, ds_cross])
        return self._ds_quality

    def _compare_all_sessions(self):
        ij_dat = {}

        ilist = []
        for i, sess_i in tqdm(enumerate(self.session_seq)):
            jlist = []
            for j, sess_j in enumerate(self.session_seq):
                if (i, j) in ij_dat:
                    ds_quality_ij = ij_dat[(i, j)]
                else:
                    ds_quality_ij = compare(sess_i=sess_i, sess_j=sess_j, same_session=i == j)
                    ij_dat[(i, j)] = ds_quality_ij
                    ij_dat[(j, i)] = ds_quality_ij
                jlist.append(ds_quality_ij)
            ds_j = xr.concat(jlist, dim='session_j')
            ilist.append(ds_j)
        ds_quality = xr.concat(ilist, dim='session')
        return ds_quality


def compare(
        sess_i: quality_within_session.Session,
        sess_j: quality_within_session.Session,
        CI_width=0.99,
        same_session: bool = False,
):
    # %% Get point estimates of cross-session quality

    ds_point_x = sess_i.ds_point
    ds_boot_x = sess_i.ds_boot_over_images_and_reps
    ds_point_y = sess_j.ds_point
    ds_boot_y = sess_j.ds_boot_over_images_and_reps

    # Compare them only on the basis of the same stimuli and neurons
    assert sess_i.stimulus_id_coord == sess_j.stimulus_id_coord, f'Stimulus id coords must match, found {sess_i.stimulus_id_coord} and {sess_j.stimulus_id_coord}'
    stimulus_id_dim = sess_i.stimulus_id_coord
    common_stimuli = sorted(set(ds_point_x[stimulus_id_dim].values).intersection(set(ds_point_y[stimulus_id_dim].values)))
    assert len(common_stimuli) == len(ds_point_x[stimulus_id_dim].values), f'Not all stimuli are common, found {len(common_stimuli)} common stimuli out of {len(ds_point_x[stimulus_id_dim].values)}'

    # Select only the same neurons
    assert sess_i.neuroid_dim == sess_j.neuroid_dim, f'Neuron id coords must match, found {sess_i.neuroid_dim} and {sess_j.neuroid_dim}'
    neuroid_dim = sess_i.neuroid_dim
    common_neuroids = sorted(set(ds_point_x[neuroid_dim].values).intersection(set(ds_point_y[neuroid_dim].values)))

    # %% Selection
    ds_point_x = ds_point_x.sel({stimulus_id_dim: common_stimuli}).sel({neuroid_dim: common_neuroids})
    ds_point_y = ds_point_y.sel({stimulus_id_dim: common_stimuli}).sel({neuroid_dim: common_neuroids})

    ds_point_cross = stats.get_cross_session_metrics(
        mu_x=ds_point_x.mu_x,
        var_x=ds_point_x.var_x,
        nreps_x=ds_point_x.nreps,
        mu_y=ds_point_y.mu_x,
        var_y=ds_point_y.var_x,
        nreps_y=ds_point_y.nreps,
        same_session=same_session,
        stimulus_dim=stimulus_id_dim,
    )
    ds_point_cross = xr.Dataset(ds_point_cross).assign_coords(
        stat=['point']
    )

    # %% Bootstrap cross metrics
    ds_boot_x = ds_boot_x.sel({neuroid_dim: common_neuroids})
    ds_boot_y = ds_boot_y.sel({neuroid_dim: common_neuroids})

    boot_cross_metrics = stats.get_cross_session_metrics(
        mu_x=ds_boot_x.mu_x,
        var_x=ds_boot_x.var_x,
        nreps_x=ds_boot_x.nreps,
        mu_y=ds_boot_y.mu_x,
        var_y=ds_boot_y.var_x,
        nreps_y=ds_boot_y.nreps,
        same_session=same_session,
        stimulus_dim=stimulus_id_dim,
    )
    ds_boot_cross = xr.Dataset(boot_cross_metrics)

    # %% Compute bootstrapped confidence intervals
    alpha = 1 - CI_width

    empirical_gt_metrics = stats.get_ground_truth_cross_session_metrics(
        mu_x=ds_point_x.mu_x,
        var_x=ds_point_x.var_x,
        mu_y=ds_point_y.mu_x,
        var_y=ds_point_y.var_x,
        same_session=same_session,
        stimulus_dim=stimulus_id_dim,
    )

    ds_empirical_gt = xr.Dataset(empirical_gt_metrics)

    delta1 = ds_empirical_gt - ds_boot_cross.quantile(alpha / 2, dim='boot_iter')  # No, this is not reversed.
    delta2 = ds_boot_cross.quantile(1 - alpha / 2, dim='boot_iter') - ds_empirical_gt
    ds_CI_low = ds_point_cross - delta2
    ds_CI_high = ds_point_cross + delta1
    ds_CI = xr.concat([ds_CI_low, ds_CI_high], dim='stat')
    ds_CI = ds_CI.assign_coords(
        stat=[f'CI_low', 'CI_high'],
        CI_width=CI_width,
    )
    del ds_CI['quantile']

    ds_cross_quality = xr.concat([ds_point_cross, ds_CI], dim='stat')

    # Get one-tailed p-values for signal_covariance, where H0: signal_covariance = 0.
    # This is an extremely permissive criterion for a cross-session comparison; a neuron that is at all "similar"
    # to itself between days should pass this.
    v = 'signal_covariance'
    bvals = ds_point_cross[v] + ds_empirical_gt[v] - ds_boot_cross[v]
    p_value = (bvals <= 0).mean(dim='boot_iter')
    ds_cross_quality['pvalue_signal_covariance'] = p_value

    return ds_cross_quality


# %%
if __name__ == '__main__':
    path = '/Users/mjl/PycharmProjects/ndashboard/experiment/dicarlo_rsvp/da_date_example1.nc'
    da = xr.load_dataarray(path)
    da = da.dropna('neuroid')
    session1 = (data_template.SessionNeuralData(da_presentation=da))

    path = '/Users/mjl/PycharmProjects/ndashboard/experiment/dicarlo_rsvp/da_date_example0.nc'
    da = xr.load_dataarray(path)
    da = da.dropna('neuroid')
    session2 = (data_template.SessionNeuralData(da_presentation=da))
    import glob

    pathlist = glob.glob('/Users/mjl/PycharmProjects/ndashboard/experiment/dicarlo_rsvp/neural_data/da*.nc')
    sessionlist = []
    for path in pathlist:
        da = xr.load_dataarray(path)
        da = da.dropna('neuroid')
        da = da.sel(neuroid = ['cIT' not in v for v in da.neuroid.values])
        session = (data_template.SessionNeuralData(da_presentation=da))
        sessionlist.append(session)

    # %%
    long = LongitudinalQuality(
        session_data=sessionlist,
        nboot=1000,
        boot_seed=0
    )

    # %%
    ds_quality = long.ds_quality
    ds_point = long.ds_point

    # %%
    import matplotlib.pyplot as plt

    utilz.set_my_matplotlib_defaults()

    def plot(ds, v, *args, color=None, **kwargs):
        ds = ds.sortby('timestamp_start')

        xx = np.arange(len(ds.session))

        if 'stat' in ds[v].dims:
            point = ds[v].sel(stat='point')
            CI_low = ds[v].sel(stat='CI_low')
            CI_high = ds[v].sel(stat='CI_high')
            utilz.fill_between_curves(xx, ylb_seq=CI_low, yub_seq=CI_high, alpha=0.2, color=color, )

        else:
            point = ds[v]
        plt.plot(xx, point, *args, color=color, **kwargs)

    plt.figure(figsize=(6, 3))
    i_neuroid = 55
    plot(ds_quality.isel(neuroid=i_neuroid), 'noise_variance', color = 'red')
    plot(ds_quality.isel(neuroid=i_neuroid), 'signal_variance', color = 'blue')
    utilz.axhline(0)
    plt.show()

    # %%
    ncomparisons = ds_quality.sizes['neuroid']
    FW_alpha = 0.01
    alpha = FW_alpha / ncomparisons

    naccepted = (ds_quality.pvalue_signal_variance < alpha).sum('neuroid')
    plt.plot(naccepted, '.-')
    utilz.axhline(0)
    utilz.axhline(ncomparisons)
    plt.ylabel('# of signal neurons')
    plt.show()

    # %%
    plt.figure(figsize = (3,3))
    plt.plot(ds_quality.SNR.sel(stat = 'point'), np.maximum(ds_quality.pvalue_signal_variance,1e-3),  '.', alpha = 0.05, color = 'gray')
    plt.xlabel("SNR")
    plt.ylabel('p-value')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # %%
    plt.figure(figsize = (3,3))
    x = 'signal_variance'
    y = 'noise_variance'
    plt.scatter(
        ds_quality[x].sel(stat = 'point'),
        ds_quality[y].sel(stat = 'point'),
        c = ds_quality.pvalue_signal_variance,
        alpha = 0.05,
    )
    plt.xlabel('Signal variance')
    plt.ylabel('Noise variance')
    plt.show()
    # %%
    plt.title('SN')
    plt.plot(ds_quality.SNR.sel(stat = 'point'), np.maximum(ds_quality.pvalue_signal_variance,1e-3),  '.', alpha = 0.25, color = 'gray')
    plt.xlabel("SNR")
    plt.ylabel('p-value')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
