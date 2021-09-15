# Code to problem 2


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from scipy.sparse import spdiags

sns.set_theme()


# Part b: compare the spectrum of the discrete and continuous operator
def spectrum_of_continuous_operator(indices: np.ndarray, k, L) -> np.ndarray:
    return L ** 2 / indices ** 2 / np.pi ** 2 / k


def spectrum_of_discrete_operator(indices: np.ndarray, k, nx, L) -> np.ndarray:
    h = L / nx
    return h ** 2 / (4 * k) / np.sin(np.pi * indices * h / 2 / L) ** 2


k = 1
L = 1
nx = 100

spectrums = DataFrame({
    'continuous': spectrum_of_continuous_operator(np.arange(1, nx), k, L),
    'discrete': spectrum_of_discrete_operator(np.arange(1, nx), k, nx, L),
})

plt.figure(constrained_layout=True, figsize=(7, 5))
sns.lineplot(data=spectrums)
plt.legend()
plt.gca().set(xlabel='index',
              ylabel='eigenvalue',
              yscale='log')
plt.savefig('p2b.eps')


# Part c: try naive inversion for noise data
def get_K_matrix(k, h, n):
    """
    (Copied from notebook)
    The function assembles the matrix K
    Here:
    - k is the thermal diffusivity coefficient
    - h is the spacial discretization size
    - n is the number of subintervals
    """
    # To assemble matrix K, first create 3 diagonals
    # and then call scipy function `spdiags`,
    # which creates a sparse matrix given the diagonals
    diagonals = np.zeros((3, n))  # 3 diagonals
    diagonals[0, :] = -1.0 / h ** 2
    diagonals[1, :] = 2.0 / h ** 2
    diagonals[2, :] = -1.0 / h ** 2
    K = k * spdiags(diagonals, [-1, 0, 1], n, n)
    return K


def naive_inversion(d: np.ndarray, F: np.matrix) -> np.ndarray:
    return np.linalg.solve(F, d[:, np.newaxis])


def naive_inversion_test(k, L, n, noise_sigma, noise_type):
    # Get forward operator
    F = np.linalg.inv(get_K_matrix(k, L / n, n - 1).todense())

    # Generate ground truth and the accompanied noise
    xs = np.linspace(0, 1, n + 1)[1:-1]
    m_truth = np.clip(1 - np.abs(1 - 4 * xs), 0, None) + 100 * xs ** 10 * (1 - xs) ** 2

    if noise_type == 'gaussian':
        eta = noise_sigma * np.random.randn(n - 1)
    elif noise_type == 'poisson':
        eta = (np.random.poisson(1, size=(n - 1)) - 1) * noise_sigma
    else:
        raise Exception(f'Cannot recognize noise type {noise_type}!')
    d_truth = np.array(F @ m_truth).flatten()
    d = d_truth + eta

    m = naive_inversion(d, F)

    # Plot the comparison
    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 5), sharex=True)
    axs[0].plot(xs, m_truth, 'r--', label='ground truth')
    axs[0].plot(xs, m, 'b-', label='reconstructed')
    axs[0].legend(loc='best')
    axs[0].set(xlabel='x',
               ylabel='body force',
               title='true and reconstructed $m(x)$')

    axs[1].plot(xs, d_truth, 'r--', label='ground truth')
    axs[1].plot(xs, d, 'b-', label='observed')
    axs[1].legend(loc='best')
    axs[1].set(xlabel='x',
               ylabel='displacement',
               title='true and observed $d(x)$')

    return fig


# naive_inversion_test(
#     k=1,
#     L=1,
#     n=10,
#     noise_sigma=10 ** -4,
#     noise_type='gaussian',
# ).savefig('p2c-n=10-inversion.eps')
#
# naive_inversion_test(
#     k=1,
#     L=1,
#     n=30,
#     noise_sigma=10 ** -4,
#     noise_type='gaussian',
# ).savefig('p2c-n=30-inversion-gaussian.eps')
#
# naive_inversion_test(
#     k=1,
#     L=1,
#     n=30,
#     noise_sigma=10 ** -4,
#     noise_type='poisson',
# ).savefig('p2c-n=30-inversion-poisson.eps')
#
# naive_inversion_test(
#     k=1,
#     L=1,
#     n=100,
#     noise_sigma=10 ** -4,
#     noise_type='gaussian',
# ).savefig('p2c-n=100-inversion-gaussian.eps')
#
# naive_inversion_test(
#     k=1,
#     L=1,
#     n=100,
#     noise_sigma=10 ** -4,
#     noise_type='poisson',
# ).savefig('p2c-n=100-inversion-poisson.eps')
#
# naive_inversion_test(
#     k=10,
#     L=1,
#     n=30,
#     noise_sigma=10 ** -4,
#     noise_type='gaussian',
# ).savefig('p2c-n=30-inversion-gaussian-large-k.eps')
#
# naive_inversion_test(
#     k=10,
#     L=1,
#     n=100,
#     noise_sigma=10 ** -4,
#     noise_type='gaussian',
# ).savefig('p2c-n=100-inversion-gaussian-large-k.eps')


# Part d: use tikhonov regularization to get a more regularized solution
def inversion_with_tikhonov_regularization(d: np.ndarray, F: np.matrix, alpha: float) -> np.ndarray:
    return np.linalg.solve(F.T @ F + alpha * np.identity(F.shape[1]), F.T @ d[:, np.newaxis])


n = 200
K = 1
k = 1
noise_sigma = 1e-4

# Get forward operator
F = np.linalg.inv(get_K_matrix(k, L / n, n - 1).todense())

# Generate ground truth and the accompanied noise
xs = np.linspace(0, 1, n + 1)[1:-1]
m_truth = np.clip(1 - np.abs(1 - 4 * xs), 0, None) + 100 * xs ** 10 * (1 - xs) ** 2

eta = noise_sigma * np.random.randn(n - 1)
d_truth = np.array(F @ m_truth).flatten()
d = d_truth + eta

solution_dict = {alpha: inversion_with_tikhonov_regularization(d, F, alpha)
                 for alpha in np.power(10.0, np.arange(-7, -1))}

# Plot the comparison
fig = plt.figure(constrained_layout=True, figsize=(7, 5))
plt.plot(xs, m_truth, 'r--', label='ground truth')
for alpha, m in solution_dict.items():
    plt.plot(xs, m, label=f'{alpha:.0e}')
plt.legend(loc='best', title='$\\alpha$')
plt.savefig('p2d.eps')

# Part e: use L-curve criterion to pick the best alpha
fig = plt.figure(constrained_layout=True, figsize=(7, 5))
L_curve_dict = dict()
m_norm_label = 'norm of model param ||m||'
e_norm_label = 'norm of error in observation ||Fm-d||'
for alpha, m in solution_dict.items():
    L_curve_dict[alpha] = {
        m_norm_label: np.linalg.norm(np.array(m).flatten(), ord=2),
        e_norm_label: np.linalg.norm(np.array(F @ m - d[:, np.newaxis]).flatten(), ord=2),
    }
L_curve = DataFrame(L_curve_dict).T
sns.scatterplot(data=L_curve, x=e_norm_label, y=m_norm_label)
for alpha, pair in L_curve.iterrows():
    plt.text(pair[e_norm_label] + 0.01, pair[m_norm_label] + 0.1, f'$\\alpha$={alpha:.0e}')
plt.savefig('p2e.eps')

# Part f: use morozov's criterion to find the best alpha
fig = plt.figure(constrained_layout=True, figsize=(7, 5))
delta = noise_sigma * np.sqrt(n)

error_in_observation = Series({
    alpha: np.linalg.norm(np.array(F @ m - d[:, np.newaxis]).flatten(), ord=2)
    for alpha, m in solution_dict.items()
})

sns.lineplot(data=error_in_observation)
plt.gca().set(
    xlabel='$\\alpha$',
    ylabel='norm of error in observation ||Fm-d||',
    xscale='log',
    yscale='log',
)
plt.gca().axhline(delta, ls='--')
plt.savefig('p2f.eps')

# Part g: use morozov's criterion to find the best alpha
fig = plt.figure(constrained_layout=True, figsize=(7, 5))

error_in_m = Series({
    alpha: np.linalg.norm(m - m_truth, ord=2)
    for alpha, m in solution_dict.items()
})

sns.lineplot(data=error_in_m)
plt.axvline(error_in_m.idxmin(), c='r', ls='--')
plt.gca().set(
    xlabel='$\\alpha$',
    ylabel='norm of error in m $||m_{truth}-m_{\\alpha}||$',
    xscale='log',
    yscale='log',
)
plt.savefig('p2g.eps')

plt.show()
