import matplotlib.pyplot as plt
import utilz
import numpy as np
utilz.set_my_matplotlib_defaults()
np.random.seed(0)

nimages = 50
y_gen = np.square(np.random.randn(nimages) * 2.2) + 10
#sigmas = np.square(np.random.randn(len(y_gen)) * 2)

nreps = 20
def sample_Y():


    Y = np.random.poisson(y_gen[:,None], size = (len(y_gen), nreps)) + np.random.poisson(100, size = (len(y_gen), nreps))
    return Y
y_gt = np.array([sample_Y().mean(1) for _ in range(1000)]).mean(0)

# %%
Y = sample_Y()
#Y = np.random.randn(len(y_gt), nreps) * sigmas[:, None] + y_gt[:, None]

plt.figure()
plt.title('Example single-neuron experiment')
plt.ylabel('Image')
plt.xlabel("Rep")
plt.imshow(Y, aspect = 'auto', )
utilz.savefig('example_single_neuron_experiment.png', dpi = 400)
plt.show()

# %% Plot model scatters
x_model = y_gt + np.random.randn(len(y_gt)) * 10 * y_gt / y_gt.max()
utilz.errorbar(x_model, Y.mean(1), yerr = Y.std(1) / np.sqrt(nreps), fmt = 'o', label = 'Model')
plt.xlabel('model predicted firing rate')
plt.ylabel('neuron firing rate')
utilz.unity(Y.mean(1).min(), Y.mean(1).max())
utilz.remove_ticks()
utilz.savefig('model_prediction_scatter.png', dpi = 400)
plt.title('Example model')
plt.show()

# %% Compute split half
import collections
import scipy.stats as ss

RS = np.random.RandomState(0)
d = collections.defaultdict(list)
nsplits = 1000
from tqdm import trange
for _ in trange(nsplits):
    p = RS.permutation(nreps)
    i_0 = p[:len(p)//2]
    i_1 = p[len(p)//2:]
    v0 = Y[:, i_0].mean(1)
    v1 = Y[:, i_1].mean(1)

    PearsonR = ss.pearsonr(v0, v1)[0]
    d['r'].append(PearsonR)
    d['rmodel'].append(ss.pearsonr(x_model, v1)[0])

# %%
fig, ax = plt.subplots(1, 3, figsize = (9, 3))
plt.sca(ax[0])
plt.title('Example split-half scatter (r=%0.2f)'%(np.mean(d['rmodel'])))
plt.plot(x_model, v1, '.')
plt.xlabel('Model')
plt.ylabel('Firing Rate Split 1')
utilz.unity(v0.min(), v1.max())
utilz.remove_ticks()

plt.sca(ax[1])
plt.title('Example split-half scatter (r=%0.2f)'%(np.mean(d['r'])))
plt.plot(v0, v1, '.')
plt.xlabel('Firing Rate Split 0')
plt.ylabel('Firing Rate Split 1')
utilz.unity(v0.min(), v1.max())
utilz.remove_ticks()
plt.sca(ax[2])
plt.title('Split half Pearson R')
plt.hist(d['r'])
plt.hist(d['rmodel'])
#utilz.axvline(np.mean(d['rmodel']), color = 'blue')
plt.xlabel("Split-half Pearson R")
plt.ylabel('nsplits')
#plt.xlim([0, 1])
plt.xlim([0, 1])
plt.tight_layout()
utilz.savefig('example_split_half_correlations.png', dpi = 400)

plt.show()

# %%
