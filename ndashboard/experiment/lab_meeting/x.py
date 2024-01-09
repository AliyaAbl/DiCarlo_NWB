import utilz
import matplotlib.pyplot as plt
import numpy as np
import os

savedir = './figures_lab_meeting'

utilz.set_my_matplotlib_defaults()
npoints = 80

np.random.seed(0)
y = np.square(np.random.randn(npoints) * 2) + 10
yerr = np.random.randn(npoints) * 2
x = y + np.random.randn(npoints) * 0.2


def plot(x, y, yerr):
    utilz.errorbar(x, y, yerr=yerr, mec='gray', mew=0.3, color='blue', alpha=1)
    utilz.unity(0, 50)
    plt.axis('equal')
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    plt.xlabel('Predicted spikes (Hz)')


fig, ax = plt.subplots(1, 4, figsize=(6, 2), sharey=True)
plt.sca(ax[3])
plot(x - 10, y, yerr)

plt.sca(ax[2])
plot(0.3 * x + 3, y, yerr)

plt.sca(ax[1])
plot(np.square(np.random.randn(len(x)) * 3) + 5, y, yerr)

plt.sca(ax[0])
plot(x, y, yerr)
# plot(, y, yerr)


plt.sca(ax[0])
plt.ylabel('Actual spike rate (Hz)')

plt.tight_layout()
utilz.savefig(os.path.join(savedir, 'lab_meeting_offset.png'), dpi=300)
plt.show()


# %% Behavior version

utilz.set_my_matplotlib_defaults()
npoints = 50

np.random.seed(0)
y = np.random.rand(npoints) * 0.5 + 0.5
yerr = np.sqrt(y * (1 - y) / 5)
ylogit = np.log(y / (1 - y))
xlogit = (ylogit + (np.random.randn(npoints) * 0.5))
x = 1 / (1 + np.exp(-xlogit))
x = np.clip( x + np.random.randn(npoints) * 0.05, 0.45, 1)

nper = np.random.randint(4, 20, size = len(y))
x = np.random.binomial(nper, y, ) / nper




def plot(px, py,  nreps_x=np.inf, nreps_y=5, ):


    perrx = np.sqrt(px * (1 - px) / nreps_x)
    perry = np.sqrt(py * (1 - py) / nreps_y)

    if nreps_x == np.inf:
        perrx = None
    if nreps_y == np.inf:
        perry = None


    utilz.errorbar(px, py,xerr=perrx, yerr=perry, mec='gray', mew=0.3, color='blue', alpha=1)
    utilz.unity(0, 1)
    plt.axis('equal')
    plt.axis([0.3, 1.05, 0.3, 1.05])
    utilz.origin(0.5, 0.5)
    #plt.xlim([0, 50])
    #plt.ylim([0, 50])


plt.figure(figsize = (3, 3))
plot(x, y, np.inf, np.inf)
plt.title('Performance')
plt.xlabel("Model")
plt.ylabel("Human")
plt.tight_layout()
utilz.savefig(os.path.join(savedir, 'behavior/no_errorbars.png'), dpi=300)

plt.show()

plt.figure(figsize = (3, 3))
plot(x, y, np.inf, 50)
plt.title('Performance')
plt.xlabel("Model")
plt.ylabel("Human")
plt.tight_layout()
utilz.savefig(os.path.join(savedir, 'behavior/with_errorbars.png'), dpi=300)
plt.show()



plt.figure(figsize = (3, 3))
plot(x, y, 50, 50)
plt.title('Performance')
plt.xlabel("Model")
plt.ylabel("Human")
plt.tight_layout()
utilz.savefig(os.path.join(savedir, 'behavior/with_model_errorbars.png'), dpi=300)
plt.show()




# %% Behavior version

utilz.set_my_matplotlib_defaults()
npoints = 50

np.random.seed(0)
y = np.random.rand(npoints) * 0.5 + 0.5
yerr = np.sqrt(y * (1 - y) / 5)
ylogit = np.log(y / (1 - y))
xlogit = (ylogit + (np.random.randn(npoints) * 0.5))
x = 1 / (1 + np.exp(-xlogit))
x = np.clip( x + np.random.randn(npoints) * 0.05, 0.45, 1)

nper = np.random.randint(15, 50, size = len(y))
x = np.random.binomial(nper, y, ) / nper



def plot(px, py,  nreps_x=np.inf, nreps_y=5, lapse = 0):

    px = px * (1 -lapse) + lapse / 2
    perrx = np.sqrt(px * (1 - px) / nreps_x)
    perry = np.sqrt(py * (1 - py) / nreps_y)

    if nreps_x == np.inf:
        perrx = None
    if nreps_y == np.inf:
        perry = None


    utilz.errorbar(px, py,xerr=perrx, yerr=perry, mec='gray', mew=0.3, color='blue', alpha=1)
    utilz.unity(0, 1)
    plt.axis('equal')
    plt.axis([0, 1.05, 0, 1.05])
    #utilz.origin(0.5, 0.5)
    #plt.xlim([0, 50])
    #plt.ylim([0, 50])
    plt.tight_layout()


fig, ax = plt.subplots(1, 4, figsize=(6, 2), sharey=True)

plt.sca(ax[0])
plot(x, y, np.inf, np.inf)


plt.sca(ax[1])
xsham = np.random.rand(npoints)*0.8 + 0.2
plot(xsham, y, np.inf, np.inf, lapse = 0.)

plt.sca(ax[2])

plot(x, y, np.inf, np.inf, lapse = 0.8)
plt.tight_layout()


plt.sca(ax[3])

plot(x-0.3, y, np.inf, np.inf, lapse = 0)
plt.tight_layout()

plt.sca(ax[0])
plt.title('Performance')
plt.xlabel("Model")
plt.ylabel("Human")

plt.tight_layout()

utilz.savefig(os.path.join(savedir, 'behavior/four_scenarios_noerrorbars.png'), dpi=300)
plt.show()

# %%

fig, ax = plt.subplots(1, 4, figsize=(6, 2), sharey=True)

plt.sca(ax[0])
plot(x, y, np.inf, 20)


plt.sca(ax[1])
plot(xsham, y, np.inf, 20, lapse = 0.)

plt.sca(ax[2])

plot(x, y, np.inf, 20, lapse = 0.8)
plt.tight_layout()


plt.sca(ax[3])

plot(x-0.3, y, np.inf, 20, lapse = 0)
plt.tight_layout()

plt.sca(ax[0])
plt.title('Performance')
plt.xlabel("Model")
plt.ylabel("Human")

plt.tight_layout()

utilz.savefig(os.path.join(savedir, 'behavior/four_scenarios_witherrorbars.png'), dpi=300)
plt.show()
