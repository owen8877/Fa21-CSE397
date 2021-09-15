# Code to problem 1

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Series

sns.set_theme()


def spectrum_of_continuous_operator(indices: np.ndarray, k, T, L) -> np.ndarray:
    return np.exp(-k * T * (np.pi * indices / L) ** 2)


def spectrum_of_discrete_operator(indices: np.ndarray, k, nx, nt, T, L) -> np.ndarray:
    h = L / nx
    dt = T / nt
    mu = (4 * k / h ** 2) * np.sin(np.pi * indices / 2 / nx) ** 2
    return (1 + dt * mu) ** (-nt)


# Part c: compare the spectrum for different parameter k
L = 1
T = 1
ks = np.power(10.0, [-4, -3, -2, -1])
nx = 100
spectrums = DataFrame({k: spectrum_of_continuous_operator(np.arange(1, nx), k, T, L) for k in ks})

plt.figure(constrained_layout=True, figsize=(7, 5))
sns.lineplot(data=spectrums)
plt.legend(title='k')
plt.gca().set(xlabel='index',
              ylabel='eigenvalue',
              ylim=[10 ** -16, 2],
              yscale='log')
plt.savefig('p1c.eps')

# Part d: plot the spectrum of the discrete one against the continuous one
k = 0.01
T = 0.1
nxt_pairs = (20, 20), (40, 40), (80, 80), (160, 160)

spectrums_dict = {
    'continuous': spectrum_of_continuous_operator(np.arange(1, 100), k, T, L)
}
for nx, nt in nxt_pairs:
    spectrums_dict[(nx, nt)] = spectrum_of_discrete_operator(np.arange(1, nx), k, nx, nt, T, L)
spectrums = DataFrame({k: Series(v) for k, v in spectrums_dict.items()})

plt.figure(constrained_layout=True, figsize=(7, 5))
sns.lineplot(data=spectrums)
plt.legend(title='(nx, nt)')
plt.gca().set(xlabel='index',
              ylabel='eigenvalue',
              ylim=[10 ** -16, 2],
              yscale='log')
plt.savefig('p1d.eps')

plt.show()
