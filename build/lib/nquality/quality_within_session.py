import collections

import ndashboard.nquality.raw_data_template as data_template
import xarray as xr
import ndashboard.nquality.statistics as quality_metrics
import numpy as np
from tqdm import trange


class Session(object):
    """
    A class which computes quality metrics given assumed i.i.d. data from a single experiment.
    """

    def __init__(
            self,
            session_data: data_template.SessionNeuralData,
            boot_seed: int,
            nboot: int = 1000,
    ):

        stimulus_id_to_da = session_data.stimulus_id_to_da

        self.timestamp_start = session_data.timestamp_start
        self.stimulus_id_to_da = stimulus_id_to_da  # stimulus_id: (rep, neuroid)
        self.boot_seed = boot_seed
        self.nboot = nboot

        self.stimulus_id_coord = session_data.stimulus_id_coord
        self.neuroid_dim = session_data.neuroid_dim

        return

    @property
    def ds_point(self):
        if not hasattr(self, '_ds_point'):
            ds_point = _get_ds_point(stimulus_id_to_da=self.stimulus_id_to_da)

            # Attach bootstrapped estimates of uncertainty for var_x and mu_x
            print('Bootstrapping standard error estimates for mu_x and var_x.')
            ds_boot = _get_ds_point_bootstrapped(
                stimulus_id_to_da=self.stimulus_id_to_da,
                boot_seed=self.boot_seed,
                nboot=self.nboot,
                resample_stimuli=False,
            )

            boot_std = ds_boot.std('boot_iter', ddof=1)
            ds_point['SE_mu_x'] = boot_std['mu_x']
            ds_point['SE_var_x'] = boot_std['var_x']
            self._ds_point = ds_point
        return self._ds_point

    @property
    def ds_quality(self):
        if not hasattr(self, '_ds_quality'):
            ds_quality = self._estimate_within_session_quality()
            self._ds_quality = ds_quality
        return self._ds_quality

    @property
    def ds_boot_over_images_and_reps(self):
        """
        Bootstrap resampled experiments, including over images.
        :return:
        """
        if not hasattr(self, '_ds_boot_over_images_and_reps'):
            print('Bootstrap resampling over images and reps.')
            ds_boot = _get_ds_point_bootstrapped(
                stimulus_id_to_da=self.stimulus_id_to_da,
                boot_seed=self.boot_seed,
                nboot=self.nboot,
                resample_stimuli=True,
            )
            self._ds_boot_over_images_and_reps = ds_boot
        return self._ds_boot_over_images_and_reps

    def _estimate_within_session_quality(self) -> xr.Dataset:
        """
        Assesses the quality of a single session. There are a few metrics that we regard as centrally important:
        - pvalue_signal_variance: (neuroid). The p-value of the null hypothesis that the signal variance is zero. Lower is better.
            Choosing to keep all neurons with p < 0.01 (for example) would ensure that any such resultant retained neurons have a 99% chance of having a non-zero signal variance.
            This is automatically corrected for multiple comparisons; set to the number of neuroids.

        The following statistics, with bootstrapped 95% CIs.
        - mu_x: (neuroid, stimulus_id, stat)
        - var_x: (neuroid, stimulus_id, stat)
        - nreps: (neuroid, stimulus_id)
        - signal_variance: (neuroid, stat). The signal variance, in Hz^2.
        - noise_variance: (neuroid, stat). The noise variance of the neuron, in Hz^2.
        - baseline_activity: (neuroid, stat). The baseline activity of the neuron, on average over, in Hz.


        :param session_data:
        :param CI_width:
        :return:
        """

        CI_width = 0.99
        ds_point = self.ds_point

        point_quality_metrics = quality_metrics.get_within_session_metrics(
            mu_x=ds_point.mu_x,
            var_x=ds_point.var_x,
            nreps=ds_point.nreps,
            stimulus_dim=self.stimulus_id_coord,
        )

        ds_quality_point_estimates = xr.Dataset(point_quality_metrics)

        # Bootstrap quality metrics data
        ds_boot = self.ds_boot_over_images_and_reps
        boot_quality_metrics = quality_metrics.get_within_session_metrics(
            mu_x=ds_boot.mu_x,
            var_x=ds_boot.var_x,
            nreps=ds_boot.nreps,
            stimulus_dim=self.stimulus_id_coord,
        )

        ds_quality_boot_estimates = xr.Dataset(boot_quality_metrics)

        # Compute bootstrapped confidence intervals
        alpha = 1 - CI_width

        empirical_gt_metrics = quality_metrics.get_ground_truth_within_session_metrics(
            mu_x_gt=ds_point.mu_x,
            var_x_gt=ds_point.var_x,
            stimulus_dim=self.stimulus_id_coord,
        )

        ds_empirical_gt = xr.merge([ds_point, empirical_gt_metrics])
        delta1 = ds_empirical_gt - ds_quality_boot_estimates.quantile(alpha / 2, dim='boot_iter')  # No, this is not reversed.
        delta2 = ds_quality_boot_estimates.quantile(1 - alpha / 2, dim='boot_iter') - ds_empirical_gt
        ds_CI_low = ds_quality_point_estimates - delta2
        ds_CI_high = ds_quality_point_estimates + delta1
        ds_CI = xr.concat([ds_CI_low, ds_CI_high], dim='stat')
        ds_CI = ds_CI.assign_coords(
            stat=[f'CI_low', 'CI_high'],
            CI_width=CI_width,
        )

        del ds_CI['quantile']

        # Get one-tailed p-values for signal_variance, where H0: signal_variance = 0.
        v = 'signal_variance'
        bvals = ds_quality_point_estimates[v] + ds_empirical_gt[v] - boot_quality_metrics[v]
        p_value = (bvals <= 0).mean(dim='boot_iter')  # Probability reject
        ds_quality = xr.concat(
            [
                ds_quality_point_estimates.assign_coords(stat='point'),
                ds_CI
            ],
            dim='stat'
        )

        ds_quality['pvalue_signal_variance'] = p_value

        return ds_quality


def _get_ds_point(
        stimulus_id_to_da: dict
):
    """
    Get estimates of the mean firing rate and variance in the firing rate from the data.
    :param neuroid_to_stimulus_id_to_spike_counts:
    :return:
    """
    all_stimuli = sorted(stimulus_id_to_da.keys())

    d = collections.defaultdict(list)
    coords = collections.defaultdict(list)
    for i_stim, stim in enumerate(all_stimuli):
        da = stimulus_id_to_da[stim]
        mu_x = da.mean('presentation')  # [neuroid, *]
        var_x = da.var('presentation', ddof=1)  # [neuroid, *]
        n_observations = da.sizes['presentation']  # [neuroid, *]
        d['mu_x'].append(mu_x)
        d['var_x'].append(var_x)
        coords['nreps'].append(n_observations)

    for k in d:
        d[k] = xr.concat(d[k], dim='stimulus_id')
    for k in coords:
        coords[k] = (['stimulus_id'], coords[k])

    coords['stimulus_id'] = all_stimuli

    ds_point = xr.Dataset(
        data_vars=d,
        coords=coords,
    )

    return ds_point


def _get_ds_point_bootstrapped(
        stimulus_id_to_da: dict,
        boot_seed: int,
        nboot: int,
        resample_stimuli: bool,
) -> xr.Dataset:
    """
    Returns bootstrapped point estimates (mu(x) and var(x)) of the data, where the presentation reps are resampled with replacement.
        - Note that this means any noise covariance between different neurons is preserved in bootstrapping.

    If resample_stimuli is True, then the stimuli are also bootstrap resampled with replacement, and "boot_stimulus_id" is returned.
        - The resampling of stimuli is deterministic, given the boot_seed and the same set of stimulus keys.

    ds_boot:
        mu_x: (boot_iter, stimulus_id, neuroid)
        var_x: (boot_iter, stimulus_id, neuroid)
        nreps: (boot_iter, stimulus_id)

    :param boot_seed:
    :param nboot:
    :return:
    """

    all_stimuli = sorted(stimulus_id_to_da.keys())
    nstimuli = len(all_stimuli)

    RS = np.random.RandomState(seed=boot_seed)

    # Ensure alignment of data
    all_neuroids = set()
    for stim in all_stimuli:
        # Check that all stimuli have the same set of neuroids
        neuroid_set = set(stimulus_id_to_da[stim].neuroid.values)
        all_neuroids = all_neuroids.union(neuroid_set)
    all_neuroids = sorted(all_neuroids)

    stimulus_id_to_dat = {
        stim: stimulus_id_to_da[stim].transpose('presentation', 'neuroid').sel(neuroid=all_neuroids).values for stim in all_stimuli
    }

    b_samples_mu_x = np.ones((nboot, nstimuli, len(all_neuroids))) * np.nan
    b_samples_var_x = np.ones((nboot, nstimuli, len(all_neuroids))) * np.nan
    b_samples_nreps = np.ones((nboot, nstimuli)) * np.nan
    i_stimuli_boot = np.arange(nstimuli)

    if resample_stimuli:
        RS_stim = np.random.RandomState(seed=boot_seed)
    else:
        RS_stim = None

    for i_boot in trange(nboot, desc='bootstrapping session'):
        if resample_stimuli:
            i_stimuli_boot = RS_stim.choice(nstimuli, size=nstimuli, replace=True)

        for i_boot_stim, i_stim in enumerate(i_stimuli_boot):

            # Retrieve empirical distribution
            stim = all_stimuli[i_stim]
            dat = stimulus_id_to_dat[stim]  # presentation, neuroid
            nreps = dat.shape[0]

            # Bootstrap resample reps
            i_reps = RS.choice(nreps, size=nreps, replace=True)
            x = dat[i_reps]  # presentation, neuroid

            # Compute point estimates
            if not resample_stimuli:
                i_boot_stim = i_stim
            b_samples_mu_x[i_boot, i_boot_stim] = x.mean(axis=0)
            b_samples_var_x[i_boot, i_boot_stim] = x.var(axis=0, ddof=1)
            b_samples_nreps[i_boot, i_boot_stim] = nreps

    coords = {
        'neuroid': all_neuroids,
        'nreps': (['boot_iter', 'stimulus_id'], b_samples_nreps),
    }
    if resample_stimuli:
        coords['stimulus_id'] = ['boot_stimulus_%d' % v for v in np.arange(nstimuli)]
    else:
        coords['stimulus_id'] = all_stimuli

    ds_boot = xr.Dataset(
        data_vars={
            'mu_x': (['boot_iter', 'stimulus_id', 'neuroid'], b_samples_mu_x),
            'var_x': (['boot_iter', 'stimulus_id', 'neuroid'], b_samples_var_x),
        },
        coords=coords
    )

    return ds_boot


# %%
if __name__ == '__main__':
    path = '/Users/mjl/PycharmProjects/ndashboard/experiment/dicarlo_rsvp/da_date_example1.nc'
    da = xr.load_dataarray(path)
    da = da.dropna('neuroid')
    session = (data_template.SessionNeuralData(da_presentation=da))

    session = Session(session, boot_seed=0, nboot=1000)
    ds_quality = session.ds_quality

    # %%
    import utilz
    import matplotlib.pyplot as plt

    i_neuroid = 100
    utilz.axhline(ds_quality.MSE_SR_constant.sel(stat='point', ).isel(neuroid=i_neuroid), color='k')
    utilz.axhline(ds_quality.noise_variance.sel(stat='point', ).isel(neuroid=i_neuroid), color='k')
    plt.show()
