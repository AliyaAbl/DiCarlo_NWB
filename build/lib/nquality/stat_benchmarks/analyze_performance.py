import nquality.raw_data_template as data_template
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import utilz
from tqdm import trange

# %% Get the false positive rate (type 1 error rate) of the classifier in detecting signal neurons
# %% Estimate the true positive rate (1 - type 2 error rate, or power) of the classifier, given an assumed distribution of signal neurons
# %% Calibrate the classifier to achieve a fixed false positive rate -- e.g. 0.01%.

class Decider(object):

    def decide(self, session_data: data_template.SessionNeuralData):
        """
        Returns a list of neuroids that are considered to carry signal.
        :param session_data:
        :return:
        """
        # Positive dv indicates belief in signal.

        return {'dv':[]}


import nquality.quality_within_session as dashboard


class MyDecider(Decider):

    def decide(self, session_data: data_template.SessionNeuralData):
        """
        Returns a list of neuroids that are considered to carry signal.
        :param session_data:
        :return:
        """
        ds_quality = dashboard.estimate_within_session_quality(session_data=session_data)

        stats = {
            'dv': 1-ds_quality.pvalue_signal_variance,
            'signal_variance': ds_quality.signal_variance,
            'noise_variance': ds_quality.noise_variance,
            'baseline_spike_rate': ds_quality.baseline_spike_rate,
        }

        return stats


class PearsonR_SR_Decider(Decider):

    def decide(self, session_data: data_template.SessionNeuralData):
        """
        Computes DV based on split-half pearson r
        :param session_data:
        :return:
        """

        all_neuroids = session_data.ds_point.neuroid.values
        all_stimuli = session_data.ds_point.stimulus_id.values
        niter = 5

        v0 = np.zeros((niter, len(all_neuroids), len(all_stimuli)))
        v1 = np.zeros((niter, len(all_neuroids), len(all_stimuli)))
        RS = np.random.RandomState(seed=0)

        neuroid_to_i_neuroid = {neuroid: i_neuroid for i_neuroid, neuroid in enumerate(all_neuroids)}
        stimulus_id_to_i_stimulus = {stimulus_id: i_stimulus for i_stimulus, stimulus_id in enumerate(all_stimuli)}

        for i_iter in trange(niter, desc='Computing split-half pearson r', disable=True):
            for neuroid in all_neuroids:
                stim_to_spike_counts = session_data.neuroid_to_stimulus_id_to_spike_counts[neuroid]
                i_neuroid = neuroid_to_i_neuroid[neuroid]
                for stim in stim_to_spike_counts:
                    i_stimulus = stimulus_id_to_i_stimulus[stim]
                    p = RS.permutation(stim_to_spike_counts[stim])
                    dat0 = p[:len(p) // 2].mean()
                    dat1 = p[len(p) // 2:].mean()
                    v0[i_iter, i_neuroid, i_stimulus] = dat0
                    v1[i_iter, i_neuroid, i_stimulus] = dat1

        da0 = xr.DataArray(v0, dims=['iter', 'neuroid', 'stimulus_id'], coords={'neuroid': all_neuroids, 'stimulus_id': all_stimuli})
        da1 = xr.DataArray(v1, dims=['iter', 'neuroid', 'stimulus_id'], coords={'neuroid': all_neuroids, 'stimulus_id': all_stimuli})
        pearsonr = utilz.calc_correlation(obs_dim='stimulus_id', y_pred=da0, y_actual=da1).mean('iter')

        stats = {
            'dv': pearsonr
        }

        return stats


class SI_Decider(Decider):

    def decide(self, session_data: data_template.SessionNeuralData):
        """
        Computes DV based on a basic selectivity index
        :param session_data:
        :return:
        """
        selectivity_index = (session_data.ds_point.mu_x.max('stimulus_id') - session_data.ds_point.mu_x.min('stimulus_id')) / (session_data.ds_point.mu_x.mean('stimulus_id'))

        stats = {
            'dv': selectivity_index
        }

        return stats


# %% Evaluate the performance of a Decider

class Benchmark(object):
    def __init__(
            self,
    ):
        return

    def simulate_session(self, seed):
        """
        Returns simulated data
        :param sham:
        :return:
        """
        signal_variance_gt, noise_variance_gt, baseline_activity_gt = [], [], []
        return data_template.SessionNeuralData(), signal_variance_gt, noise_variance_gt, baseline_activity_gt

    def score_decider(self, decider: Decider, seed):

        mlist = []
        session_data, signal_variance_gt, noise_variance_gt, baseline_activity_gt = self.simulate_session(seed=seed,)
        metrics = decider.decide(session_data=session_data)

        assert 'dv' in metrics

        result = xr.Dataset(
            metrics,
            coords={
                'signal_variance_gt': (['neuroid'], signal_variance_gt),
                'noise_variance_gt': (['neuroid'], noise_variance_gt),
                'baseline_spike_rate_gt': (['neuroid'], baseline_activity_gt),
            }
        )
        SNR_gt = result.signal_variance_gt / result.noise_variance_gt
        epsilon = 1e-16
        SNR_gt.values[SNR_gt.values<epsilon] = 0
        result = result.assign_coords(SNR_gt=SNR_gt)

        return result


class PoissonBenchmark(Benchmark):
    def __init__(self,
                 signal_parameter: np.ndarray,
                 baseline_activity: np.ndarray,
                 nstimuli=86,
                 nreps=20
                 ):
        super().__init__()
        self.signal_parameter = signal_parameter
        self.baseline_activity = baseline_activity

        self.nstimuli = nstimuli
        self.nreps = nreps
        self.num_neurons = len(signal_parameter)

    def simulate_session(self, seed):
        """
        Returns simulated data
        :param sham:
        :return:
        """
        self.RS = np.random.RandomState(seed=seed)
        max_firing_rate = 250  # rough upper bound, in Hz
        neural_signal_rate = (np.square(self.RS.randn(self.nstimuli, self.num_neurons) * self.signal_parameter[None, :])) + 10
        c = (self.baseline_activity + 1) / (neural_signal_rate.mean(0) + 1)
        neural_signal_rate = neural_signal_rate * c[None, :]
        neural_signal_rate = np.clip(neural_signal_rate, 0, max_firing_rate)  # [stimulus, neuroid]

        spike_counts = self.RS.poisson(lam=neural_signal_rate[..., None], size=(self.nstimuli, self.num_neurons, self.nreps))

        # Add some random additive noise
        additive_noise_rate = 10
        spike_counts = spike_counts + self.RS.poisson(lam=additive_noise_rate, size=spike_counts.shape)

        ds_dat = xr.DataArray(
            spike_counts,
            dims=['stimulus_id', 'neuroid', 'rep'],
        )
        ds_dat = ds_dat.stack(presentation=['stimulus_id', 'rep'])
        ds_dat = ds_dat.reset_index('presentation')
        del ds_dat['rep']

        signal_variance_gt = np.var(neural_signal_rate, axis=0, ddof=0)
        noise_variance_gt = np.mean(neural_signal_rate, axis=0) + additive_noise_rate
        baseline_gt = noise_variance_gt
        return data_template.SessionNeuralData(ds_dat), signal_variance_gt, noise_variance_gt, baseline_gt


class EmpiricalDataBenchmark(Benchmark):
    """
    Assume the data taken from an experiment is the ground truth distribution.
    """
    def __init__(self,
                 signal_parameter: np.ndarray,
                 baseline_activity: np.ndarray,
                 nstimuli=86,
                 nreps=20
                 ):
        super().__init__()
        self.signal_parameter = signal_parameter
        self.baseline_activity = baseline_activity

        self.nstimuli = nstimuli
        self.nreps = nreps
        self.num_neurons = len(signal_parameter)

    def simulate_session(self, seed):
        """
        Returns simulated data
        :param sham:
        :return:
        """
        self.RS = np.random.RandomState(seed=seed)
        max_firing_rate = 250  # rough upper bound, in Hz
        neural_signal_rate = (np.square(self.RS.randn(self.nstimuli, self.num_neurons) * self.signal_parameter[None, :])) + 10
        c = (self.baseline_activity + 1) / (neural_signal_rate.mean(0) + 1)
        neural_signal_rate = neural_signal_rate * c[None, :]
        neural_signal_rate = np.clip(neural_signal_rate, 0, max_firing_rate)  # [stimulus, neuroid]

        spike_counts = self.RS.poisson(lam=neural_signal_rate[..., None], size=(self.nstimuli, self.num_neurons, self.nreps))

        # Add some random additive noise
        additive_noise_rate = 10
        spike_counts = spike_counts + self.RS.poisson(lam=additive_noise_rate, size=spike_counts.shape)

        ds_dat = xr.DataArray(
            spike_counts,
            dims=['stimulus_id', 'neuroid', 'rep'],
        )
        ds_dat = ds_dat.stack(presentation=['stimulus_id', 'rep'])
        ds_dat = ds_dat.reset_index('presentation')
        del ds_dat['rep']

        signal_variance_gt = np.var(neural_signal_rate, axis=0, ddof=0)
        noise_variance_gt = np.mean(neural_signal_rate, axis=0) + additive_noise_rate
        baseline_gt = noise_variance_gt
        return data_template.SessionNeuralData(ds_dat), signal_variance_gt, noise_variance_gt, baseline_gt


# %% Plot results

"""
TPR / FPR as function of signal variance; noise variance
"""

sig = np.array(list(np.zeros(50)) + list(np.logspace(-0.8, 0.3, 20)))
base = np.ones(len(sig)) * 50
bench = PoissonBenchmark(
    signal_parameter=sig,
    baseline_activity=base,
)

all_deciders = [MyDecider(), SI_Decider(), PearsonR_SR_Decider()]

nsims = 10
dlist = []
for _ in trange(nsims):
    ds_dv = bench.score_decider(MyDecider(), seed=_)

    dlist.append(ds_dv)

# %%
ds_sim = xr.concat(dlist, 'sim')
SNR_gt = ds_sim.SNR_gt.mean('sim')

# %% Plot ROC curves
tmin = float(ds_sim.dv.min())
tmax = float(ds_sim.dv.max())
thresholds = np.linspace(tmin, tmax, 500)
FPR = []
alpha = 0.01
ds_null = ds_sim.sel(neuroid=SNR_gt == 0)
for t in thresholds:
    report_signal = ds_null.dv > t

    false_positive_rate = report_signal.mean('sim').mean('neuroid')
    FPR.append(float(false_positive_rate))
FPR = np.array(FPR)
i_threshold = np.argmin(np.abs(FPR - alpha))
ds_sim['report'] = ds_sim.dv > thresholds[i_threshold]
mask = SNR_gt > 0

plt.plot(SNR_gt.sel(neuroid=mask), ds_sim.report.mean('sim').sel(neuroid=mask), '.-')
plt.ylabel('True positive rate')
plt.title(f'Method power (FPR <= {alpha})')
plt.xlabel("Poisson Neuron SNR (ground truth)")
#plt.xscale('log')
utilz.logx_ticks(pseq = [0.001, 0.01, 0.1, 1,10])
plt.show()