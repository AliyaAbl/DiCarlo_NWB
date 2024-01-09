import numpy as np
from scipy import stats as ss
from typing import Union
import xarray as xr

min_assumed_noise = 0.1


def _check_inputs(
        mu: Union[xr.DataArray, np.ndarray],
        var: Union[xr.DataArray, np.ndarray],
        nreps: Union[xr.DataArray, np.ndarray],
        stimulus_dim,
):
    """
    :param mu: (stimulus_id, *). An unbiased estimate (the sample mean, over nreps) of the mean of the spiking rate, given a stimulus. Nan values if nreps is 0.
    :param var: (stimulus_id, *). An unbiased estimator (the Bessel-corrected sample variance, over nreps) of the variance in the spike rate, given a stimulus. Nan values if nreps <= 1.
    :param nreps: (stimulus_id, *). The number of repetitions used to form those estimates; >= 0.
    :return: None
    """
    if isinstance(mu, xr.DataArray):
        assert isinstance(var, xr.DataArray)
        assert isinstance(nreps, xr.DataArray)
        assert stimulus_dim in mu.dims, f'{stimulus_dim} not in {mu.dims}'
        assert stimulus_dim in var.dims, f'{stimulus_dim} not in {var.dims}'
        assert stimulus_dim in nreps.dims, f'{stimulus_dim} not in {nreps.dims}'
        assert np.all(nreps >= 2), f'Need at least 2 reps per presentation'
        nstimuli = len(mu[stimulus_dim])
    else:
        assert mu.shape[0] >= 2, f"Need at least 2 stimuli, {mu.shape[0]} found."
        # Checks
        assert mu.shape == var.shape, (mu.shape, var.shape)
        assert len(mu.shape) >= 1, f"mu_x must be at least 1D, {mu.shape}"

        assert mu.dtype == var.dtype == float, f'{mu.dtype}, {var.dtype}'
        assert nreps.dtype == int, f'{nreps.dtype}'
        assert np.any(np.isnan(nreps)) == False, f'{nreps}'
        assert np.all(nreps >= 2), f'Need at least 2 reps per presentation'
        assert np.all(var >= 0), f'Variance estimates should be positive. Found min value: {var.min()}'
        assert np.all(mu >= 0), f'Spike rates should be positive. Found min value:{mu.min()}'
        nstimuli = mu.shape[stimulus_dim]

    assert nstimuli >= 2, f"Need at least 2 stimuli, {nstimuli} found."
    return nstimuli


def get_ground_truth_within_session_metrics(
        mu_x_gt: Union[xr.DataArray, np.ndarray],
        var_x_gt: Union[xr.DataArray, np.ndarray],
        stimulus_dim: Union[str, int] = 0,
):
    """
    Returns the ground truth parameters of the distribution described by these two statistics.
    Useful for executing various bootstrapping algorithms, and verifying simulations.
    :param mu_x_gt:
    :param var_x_gt:
    :param stimulus_dim:
    :return:
    """

    noise_variance = var_x_gt.mean(stimulus_dim)
    signal_variance = mu_x_gt.var(stimulus_dim, ddof=0)

    baseline_spike_rate = mu_x_gt.mean(stimulus_dim)
    baseline_power = np.square(baseline_spike_rate)

    SNR = signal_variance / noise_variance
    if isinstance(SNR, xr.DataArray):
        SNR = SNR.fillna(signal_variance / min_assumed_noise)
    elif isinstance(SNR, np.ndarray):
        SNR[np.isnan(SNR)] = signal_variance / min_assumed_noise

    within_day_metrics_gt = dict(
        noise_variance=noise_variance,  # [*]
        signal_variance=signal_variance,  # [*]
        baseline_power=baseline_power,  # [*]
        baseline_spike_rate=baseline_spike_rate,  # [*]
        SNR=SNR,
        MSE_SR_constant=signal_variance + noise_variance,  # [*]. The score incurred by a constant model on the MSE SR.
    )
    return within_day_metrics_gt


def get_within_session_metrics(
        mu_x: Union[xr.DataArray, np.ndarray],
        var_x: Union[xr.DataArray, np.ndarray],
        nreps: Union[xr.DataArray, np.ndarray],
        stimulus_dim: Union[str, int] = 0,
):
    """
    :param mu_x: (stimulus_id, *). An unbiased estimate (the sample mean, over nreps) of the mean of the spiking rate, given a stimulus. Nan values if nreps is 0.
    :param var_x: (stimulus_id, *). An unbiased estimator (the Bessel-corrected sample variance, over nreps) of the variance in the spike rate, given a stimulus. Nan values if nreps <= 1.
    :param nreps: (stimulus_id, *). The number of repetitions used to form those estimates; >= 0.
    :return: dict
    """
    nstimuli = _check_inputs(
        mu_x,
        var_x,
        nreps,
        stimulus_dim
    )
    # % Compute signal variance
    standard_var_of_mu = var_x / nreps  # The standard variance of the estimator of the mean, given a stimulus.
    signal_variance = mu_x.var(stimulus_dim, ddof=1) - standard_var_of_mu.mean(stimulus_dim)

    #  % Compute baseline variance

    standard_var_of_baseline = mu_x.var(stimulus_dim, ddof=1) / nstimuli
    baseline_spike_rate = mu_x.mean(stimulus_dim)

    # Noise variance
    noise_variance = var_x.mean(stimulus_dim)

    # SNR
    SNR = signal_variance / noise_variance
    if isinstance(SNR, np.ndarray):
        SNR[np.isnan(SNR)] = 0
    elif isinstance(SNR, xr.DataArray):
        SNR = SNR.fillna(0)
    else:
        raise Exception
    # SNR = np.nanmax(SNR, 0)

    # Metrics have reduced over the stimulus_id dimension
    within_day_metrics = dict(
        noise_variance=noise_variance,  # [*]
        signal_variance=signal_variance,  # [*]
        baseline_power=np.square(baseline_spike_rate) - standard_var_of_baseline,  # [*]
        baseline_spike_rate=baseline_spike_rate,  # [*]
        SNR=SNR,
        MSE_SR_constant = signal_variance + noise_variance,  # [*]. Estimate of the score incurred by a constant model on the MSE SR.
    )

    return within_day_metrics


def get_cross_session_metrics(
        mu_x: Union[xr.DataArray, np.ndarray],
        var_x: Union[xr.DataArray, np.ndarray],
        nreps_x: Union[xr.DataArray, np.ndarray],
        mu_y: Union[xr.DataArray, np.ndarray],
        var_y: Union[xr.DataArray, np.ndarray],
        nreps_y: Union[xr.DataArray, np.ndarray],
        stimulus_dim: Union[str, int] = 0,
        same_session: bool = False,
):
    """
    The estimates mu_x, var_x and mu_y, var_y are assumed to be recorded over the same stimuli, but
    are otherwise assumed to be statistically independent.

    :param mu_x:
    :param var_x:
    :param nreps_x:
    :param mu_y:
    :param var_y:
    :param nreps_y:
    :param same_session: If True, then return the estimates under the assumption that mu_x == mu_y, and var_x == var_y are the same instances of the random variable.
    :return:
    """

    metrics_x = get_within_session_metrics(mu_x, var_x, nreps_x, stimulus_dim=stimulus_dim)
    metrics_y = get_within_session_metrics(mu_y, var_y, nreps_y, stimulus_dim=stimulus_dim)

    bx = metrics_x['baseline_spike_rate']
    by = metrics_y['baseline_spike_rate']
    baseline_power_x = metrics_x['baseline_power']
    baseline_power_y = metrics_y['baseline_power']
    signal_variance_x = metrics_x['signal_variance']
    signal_variance_y = metrics_y['signal_variance']

    noise_variance_x = metrics_x['noise_variance']
    noise_variance_y = metrics_y['noise_variance']

    if isinstance(mu_x, np.ndarray):
        assert isinstance(mu_y, np.ndarray)
        assert mu_x.shape == mu_y.shape
        nstimuli = mu_x.shape[stimulus_dim]
    else:
        assert isinstance(mu_y, xr.DataArray)
        assert mu_x.shape == mu_y.shape
        nstimuli = mu_x.sizes[stimulus_dim]

    if not same_session:
        signal_covariance = ((mu_x - bx) * (mu_y - by)).mean(stimulus_dim) * (nstimuli / (nstimuli - 1))
        squared_baseline_error = baseline_power_x + baseline_power_y - 2 * bx * by + 2 * (signal_covariance / nstimuli)
        squared_signal_error = signal_variance_x + signal_variance_y - 2 * signal_covariance
        noise_error = noise_variance_x + noise_variance_y
    else:
        assert np.all(mu_x == mu_y), f'If same_session is True, then mu_x and mu_y must be the same.'
        assert np.all(var_x == var_y), f'If same_session is True, then var_x and var_y must be the same.'
        squared_baseline_error = baseline_power_x - baseline_power_x  # By definition, 0
        squared_signal_error = signal_variance_x - signal_variance_x  # By definition, 0
        signal_covariance = signal_variance_x
        noise_error = noise_variance_x + noise_variance_y

    cross_day_metrics = dict(
        squared_baseline_error=squared_baseline_error,
        squared_signal_error=squared_signal_error,
        signal_covariance=signal_covariance,
        noise_error=noise_error,
        single_rep_mse=squared_baseline_error + squared_signal_error + noise_error,
    )

    return cross_day_metrics


def get_ground_truth_cross_session_metrics(
        mu_x: Union[xr.DataArray, np.ndarray],
        var_x: Union[xr.DataArray, np.ndarray],
        mu_y: Union[xr.DataArray, np.ndarray],
        var_y: Union[xr.DataArray, np.ndarray],
        stimulus_dim: Union[str, int] = 0,
        same_session: bool = False,
):
    """
    The estimates mu_x, var_x and mu_y, var_y are assumed to be recorded over the same stimuli, but
    are otherwise assumed to be statistically independent.

    :param mu_x:
    :param var_x:
    :param nreps_x:
    :param mu_y:
    :param var_y:
    :param nreps_y:
    :param same_session: If True, then return the estimates under the assumption that mu_x == mu_y, and var_x == var_y are the same instances of the random variable.
    :return:
    """

    metrics_x = get_ground_truth_within_session_metrics(
        mu_x_gt=mu_x,
        var_x_gt=var_x,
        stimulus_dim=stimulus_dim
    )
    metrics_y = get_ground_truth_within_session_metrics(
        mu_x_gt=mu_y,
        var_x_gt=var_y,
        stimulus_dim=stimulus_dim
    )

    bx = metrics_x['baseline_spike_rate']
    by = metrics_y['baseline_spike_rate']

    signal_variance_x = metrics_x['signal_variance']
    signal_variance_y = metrics_y['signal_variance']

    noise_variance_x = metrics_x['noise_variance']
    noise_variance_y = metrics_y['noise_variance']

    if not same_session:
        signal_covariance = ((mu_x - bx) * (mu_y - by)).mean(stimulus_dim)
        squared_baseline_error = np.square(bx - by)
        squared_signal_error = signal_variance_x + signal_variance_y - 2 * signal_covariance
        noise_error = noise_variance_x + noise_variance_y
    else:
        assert np.all(mu_x == mu_y), f'If same_session is True, then mu_x and mu_y must be the same.'
        assert np.all(var_x == var_y), f'If same_session is True, then var_x and var_y must be the same.'
        squared_baseline_error = signal_variance_x - signal_variance_x  # By definition, 0
        squared_signal_error = signal_variance_x - signal_variance_x  # By definition, 0
        signal_covariance = signal_variance_x  # By definition, the variance
        noise_error = noise_variance_x + noise_variance_y

    cross_day_metrics_gt = dict(
        squared_baseline_error=squared_baseline_error,
        squared_signal_error=squared_signal_error,
        signal_covariance=signal_covariance,
        noise_error=noise_error,
        single_rep_mse=squared_baseline_error + squared_signal_error + noise_error,
    )

    return cross_day_metrics_gt
