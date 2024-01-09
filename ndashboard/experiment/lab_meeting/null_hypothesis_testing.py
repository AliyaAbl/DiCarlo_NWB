import matplotlib.pyplot as plt
import utilz
import numpy as np

import collections

utilz.set_my_matplotlib_defaults()
np.random.seed(0)
npoints = 2
y_gt = np.array([0.5488135 , 0.71518937])
x_gt = np.array([0.7 , 0.47])


def get_phat(p, nreps):
    if nreps == np.inf:
        phat = p
        varhat_phat = np.zeros(len(p))
    else:
        phat = np.random.binomial(nreps, p) / nreps
        varhat_phat = phat * (1 - phat) / (nreps - 1)

    return phat, varhat_phat

def compute_msen(phat_x, varhat_x, phat_y, varhat_y):
    Nx = np.mean(varhat_x)
    Ny = np.mean(varhat_y)

    msen = np.mean((phat_x - phat_y) ** 2) - (Nx + Ny)
    return msen

nreps_y = 50
nreps_x = np.inf

niter = 100
msen_dist = []

d = collections.defaultdict(list)
for i in range(niter):
    phat_x, varhat_x = get_phat(x_gt, nreps_x)
    phat_y, varhat_y = get_phat(y_gt, nreps_y)
    d['phat_y'].append(phat_y)
    d['phat_x'].append(phat_x)

    # Get estimate of uncertainty using bootstrapping

    msen = compute_msen(phat_x, varhat_x, phat_y, varhat_y)
    msen_dist.append(msen)

# %%
fig, ax = plt.subplots(3, 4, figsize = (8, 6))

true_mse = np.mean((x_gt - y_gt) ** 2)
bins = np.linspace(-true_mse * 0.5, true_mse * 2, 50)

for i, nreps in enumerate([20, 50, 100, 1000]):
    plt.sca(ax[0, i])
    plt.title('nreps = %d' % nreps)
    nreps_y = nreps
    niter = 1000
    msen_dist = []
    d = collections.defaultdict(list)
    nreps_x = 1000
    import scipy.stats as ss
    for _ in range(niter):
        phat_x, varhat_x = get_phat(x_gt, nreps_x)
        phat_y, varhat_y = get_phat(y_gt, nreps_y)
        d['phat_y'].append(phat_y)
        d['phat_x'].append(phat_x)
        d['pearsonr'].append(ss.pearsonr(phat_x, phat_y)[0])
        # Get estimate of uncertainty using bootstrapping

        msen = compute_msen(phat_x, varhat_x, phat_y, varhat_y)
        msen_dist.append(msen)

    plt.plot(x_gt[0], x_gt[1], 'go')
    plt.plot(y_gt[0], y_gt[1], 'ko')
    phat_y = np.array(d['phat_y'])
    phat_x = np.array(d['phat_x'])
    plt.plot(phat_y[:, 0], phat_y[:, 1], 'k.', alpha = 0.5, ms = 3)
    plt.plot(phat_x[:, 0], phat_x[:, 1], 'g.', alpha=0.2, ms=3)
    plt.axis([0, 1.01, 0, 1.01])

    plt.sca(ax[1, i])
    plt.hist(msen_dist, bins = bins, density=False, color = 'blue')
    utilz.axvline(true_mse, color = 'r', label = 'reality')
    #utilz.axvline(np.mean(msen_dist), color = 'k', label = 'avg')
    if i == 0:
        plt.title(r'$MSE_{nc}$ over experiments')
        plt.ylabel('n')
        plt.xlabel(r'$MSE_{nc}$')

    plt.xlim([bins.min(), bins.max()])
    plt.ylim([0, 200])
    utilz.axvline(0, ls = '-', color = 'k')

    plt.sca(ax[2, i])
    plt.hist(d['pearsonr'], bins = np.linspace(-1.1, 1.1, len(bins)), density=False, color = 'blue')
    if i == 0:
        plt.title(r'$PearsonR$ over experiments')
        plt.ylabel('n')
        plt.xlabel(r'$PearsonR$')
    plt.ylim([0, 1000])

plt.tight_layout()
utilz.savefig('msenc_sampling_distribution.png', dpi = 400)

plt.show()
