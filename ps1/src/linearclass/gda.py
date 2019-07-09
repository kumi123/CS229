import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()
    model.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=False)
    image_path = save_path[:-3] + "png"
    theta = np.concatenate(model.theta)
    assert theta.shape == (x_val.shape[1]+1, 1)
    util.plot(
        x=x_val, y=y_val,
        theta=theta,
        save_path=image_path
    )
    # Use np.savetxt to save outputs from validation set to save_path
    prob_val = model.predict(x_val)
    np.savetxt(save_path, prob_val)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape  # Get the shape
        if self.theta is None:
            self.theta = [np.zeros(1), np.zeros([d, 1])]
        # Find phi, mu_0, mu_1, and sigma
        self.phi = np.mean(y)
        self.mu_0 = np.mean(x[y == 0], axis=0).reshape(d, 1)
        self.mu_1 = np.mean(x[y == 1], axis=0).reshape(d, 1)
        self.mu = [self.mu_0, self.mu_1]
        total = np.zeros([d, d])
        for i in range(n):
            c = x[i, :].reshape(d, 1) - self.mu[int(y[i])]
            total += np.matmul(c, c.T)
        self.sigma = total / n
        # Write theta in terms of the parameters
        # theta_0
        self.theta[0] = np.log(self.phi / (1 - self.phi)) + 0.5 * np.matmul(
            np.matmul(self.mu_0.T, np.linalg.inv(self.sigma)),
            self.mu_0
        ) - 0.5 * np.matmul(
            np.matmul(self.mu_1.T, np.linalg.inv(self.sigma)),
            self.mu_1
        )
        print("Theta 0 shape: ", self.theta[0].shape)
        self.theta[1] = np.matmul(np.linalg.inv(self.sigma).T, (self.mu_1 - self.mu_0))
        print("Theta 1 shape: ", self.theta[1].shape)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape

        def g(z): return 1 / (1 + np.exp(-z))
        z = np.matmul(x, self.theta[1]) + (np.ones([n, 1]) * self.theta[0])
        prob = g(z).reshape(n,)
        return prob
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
