import matplotlib.pyplot as plt
import numpy as np
import os
import math
from scipy.stats import multivariate_normal

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # if is_semi_supervised:
    #     considered = x_all
    # else:
    #     considered = x
    considered = x
    n = considered.shape[0]
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    mu = []  # [2,] * 4
    sigma = []  # [2, 2] * 4
    random_assignments = np.random.randint(0, K, n)
    for i in range(K):
        in_cluster = considered[random_assignments == i]
        # mu[i] = np.mean(in_cluster, axis=0)
        mu.append(np.mean(in_cluster, axis=0))
        sigma.append(np.cov(in_cluster, rowvar=False))
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones(K) / K
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones([considered.shape[0], K]) / K
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


# def gaussian(xvar, mu, sigma) -> float:
#     dist = multivariate_normal(mean=mu, cov=sigma)
#     return dist.pdf(xvar)

def gaussian(xvar, mu, sigma) -> float:
        d = sigma.shape[0]
        # Multivariate Gaussian
        f = 1 / (
            ((2 * np.pi) ** (d / 2)) * (np.linalg.det(sigma) ** (1 / 2))
        )
        xvar = xvar.reshape(-1, 1)
        ker = np.squeeze(np.exp(- 0.5 * np.matmul(np.matmul(xvar.T, np.linalg.inv(sigma)), xvar)))
        return float(f * ker)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n, d).
        w: Initial weight matrix of shape (n, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (d,).
        sigma: Initial cluster covariances, list of k arrays of shape (d, d).

    Returns:
        Updated weight matrix of shape (n, d) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    n, d = x.shape

    # Gaussian distributions for each category
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        it += 1
        # pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        # Use Bayes Rule.
        # w[i, j] = p(z=j|xi)
        # for i in range(n):
        #     denominator = 0.0
        #     for j in range(K):
        #         # p(xi|zi=j) * p(zi=j)
        #         posterior = gaussian(x[i], mu[j], sigma[j]) * phi[j]
        #         denominator += posterior
        #     for j in range(K):
        #         w[i, j] = gaussian(x[i], mu[j], sigma[j]) * phi[j] / denominator
        #     # total = np.array(total)
        #     # w[i, :] = total / total.sum()
        for i in range(n):
            denominator = 0
            for j in range(K):
                numerator = multivariate_normal.pdf(
                    x[i, :], mu[j], sigma[j]) * phi[j]
                denominator += numerator
                w[i, j] = numerator
            w[i, :] /= denominator
        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(K):
            const = w[:, j].sum()
            phi[j] = 1 / n * const
            mu_j = np.zeros(d)
            sigma_j = np.zeros((d, d))
            for i in range(n):
                mu_j += (x[i, :] * w[i, j])
                temp = (x[i, :] - mu[j]).reshape(-1, 1)
                sigma_j += w[i, j] * (temp) @ (temp).T
            mu[j] = mu_j / const
            sigma[j] = sigma_j / const
        # *** Previous M-Step ***
        # new_phi = w.sum(axis=0) / n
        # # new_phi = w.sum(axis=0) / w.sum()
        # # new_mu = (w.T @ x) / w.sum(axis=0).reshape(-1, 1)  # An vectorized implementation.
        # new_mu = list()
        # new_sigma = list()
        # for j in range(K):
        #     # mu[j, :] = np.matmul((w[:, 1].reshape(1, -1)), x).reshape(-1) / mu[j, :].sum()
        #     total_vec = np.zeros(d)
        #     denominator = 0.0
        #     for i in range(n):
        #         total_vec += w[i, j] * x[i]
        #         denominator += w[i, j]
        #     new_mu.append(total_vec / denominator)
        #     sigma_placeholder = np.zeros_like(sigma[j])
        #     for i in range(n):
        #         ci = (x[i] - mu[j]).reshape(-1, 1)
        #         partial = w[i, j] * np.matmul(ci, ci.T)
        #         sigma_placeholder += partial
        #     new_sigma.append(sigma_placeholder / (denominator))
        # # Assign new values
        # phi[:] = new_phi
        # for j in range(K):
        #     mu[j] = new_mu[j]
        #     sigma[j] = new_sigma[j]
        # *** End ***

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        ll = 0
        for i in range(n):
            temp = 0
            for j in range(K):
                temp += multivariate_normal.pdf(x[i, :], mu[j], sigma[j]) * phi[j]
            ll += np.log(temp)
        if prev_ll is not None:
            assert ll - prev_ll > 1e-10
            print("Checked!")
        print("Iteration: {}, {}".format(it, ll))
        # *** END CODE HERE ***
    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n, d).
        x_tilde: Design matrix of labeled examples of shape (n_tilde, d).
        z_tilde: Array of labels of shape (n_tilde, 1).
        w: Initial weight matrix of shape (n, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (d,).
        sigma: Initial cluster covariances, list of k arrays of shape (d, d).

    Returns:
        Updated weight matrix of shape (n, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000
    # Load necessary variables
    n, d = x.shape
    n_tilde = z_tilde.shape[0]
    k = w.shape[1]
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE ***
        it += 1
        # (1) E-step: Update your estimates in w
        # Only update for unlabelled observations.
        # *** Loop Approach ***
        # for i in range(n):
        #     denominator = 0.0
        #     for j in range(K):
        #         denominator += gaussian(x[i], mu[j], sigma[j]) * phi[j]
        #     for j in range(K):
        #         w[i, j] = gaussian(x[i], mu[j], sigma[j]) * phi[j] / denominator
        # *** End ***

        for i in range(n):
            denominator = 0
            for j in range(K):
                numerator = multivariate_normal.pdf(
                    x[i, :], mu[j], sigma[j]) * phi[j]
                denominator += numerator
                w[i, j] = numerator
            w[i, :] /= denominator

        # (2) M-step: Update the model parameters phi, mu, and sigma
        # Update phi
        new_phi = np.zeros_like(phi)
        for j in range(K):
            phi_j = 0.0
            for i in range(n):
                phi_j += w[i, j]
            for i in range(n_tilde):
                if z_tilde[i] == j:
                    phi_j += alpha
            new_phi[j] = phi_j / (n + alpha * n_tilde)
        # Update mu
        new_mu = list()
        for j in range(K):
            muj = np.zeros(d)
            for i in range(n):
                muj += w[i, j] * x[i]
            for i in range(n_tilde):
                if z_tilde[i] == j:
                    muj += alpha * x_tilde[i]
            denominator = 0.0
            for i in range(n):
                denominator += w[i, j]
            for i in range(n_tilde):
                if z_tilde[i] == j:
                    denominator += alpha
            # muj = muj / (w.sum(axis=0)[j] + alpha * np.sum(z_tilde == j))
            new_mu.append(muj / denominator)
        # update sigma
        for j in range(K):
            const = w[:, j].sum() + alpha * np.sum(z_tilde == j)
            phi[j] = 1 / (n + alpha * n_tilde) * const
            mu_j = np.zeros(d)
            sigma_j = np.zeros((d, d))
            for i in range(n):
                mu_j += (x[i, :] * w[i, j])
                temp = (x[i, :] - mu[j]).reshape(-1, 1)
                sigma_j += w[i, j] * (temp) @ (temp).T
            for i in range(n_tilde):
                mu_j += alpha * (x_tilde[i, :] * int(z_tilde[i] == j))
                temp = (x_tilde[i, :] - mu[j]).reshape(-1, 1)
                sigma_j += alpha * int(z_tilde[i] == j) * (temp) @ (temp).T
            mu[j] = mu_j / const
            sigma[j] = sigma_j / const
        # new_sigma = list()
        # for j in range(K):
        #     # denominator = w.sum(axis=0)[j] + alpha * np.sum(z_tilde == j)
        #     # Compute normalizer
        #     denominator = 0.0
        #     for i in range(n):
        #         denominator += w[i, j]
        #     for i in range(n_tilde):
        #         # if z_tilde[i] == j:
        #         #     denominator += alpha
        #         denominator += int(z_tilde[i] == j)
        #     sigma_j = np.zeros([d, d])
        #     for i in range(n):
        #         current = (x[i] - mu[j]).reshape(-1, 1)
        #         sigma_j += w[i, j] * np.matmul(current, current.T)
        #     for i in range(n_tilde):
        #         if z_tilde[i] == j:
        #             current = (x_tilde[i] - mu[j]).reshape(-1, 1)
        #             sigma_j += alpha * np.matmul(current, current.T)
        #     new_sigma.append(sigma_j / denominator)
        # # assign new vars.
        # for j in range(K):
        #     phi[j] = new_phi[j]
        #     mu[j] = new_mu[j]
        #     sigma[j] = new_sigma[j]
        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        # ll_unsup = 0.0
        # for i in range(n):
        #     counter = 0.0
        #     for j in range(K):
        #         counter += gaussian(x[i], mu[j], sigma[j]) * phi[j]
        #     ll_unsup += np.log(counter)
        ll_unsup = 0
        for i in range(n):
            temp = 0
            for j in range(K):
                temp += multivariate_normal.pdf(x[i, :], mu[j], sigma[j]) * phi[j]
            ll_unsup += np.log(temp)
        ll_sup = 0
        for i in range(n_tilde):
            sup_z = int(z_tilde[i])
            ll_sup += np.log(
                multivariate_normal.pdf(x_tilde[i, :], mu[sup_z], sigma[sup_z]) * phi[sup_z]
            )
        ll = ll_unsup + alpha * ll_sup
        if prev_ll is not None:
            assert ll - prev_ll > 1e-10
            print("Checked!")
        print("Iteration: {}, {}".format(it, ll))
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)
        main(is_semi_supervised=True, trial_num=t)
        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        # *** END CODE HERE ***
