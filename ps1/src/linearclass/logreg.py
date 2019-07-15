import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    image_path = save_path[:-3] + "png"
    util.plot(
        x=x_val, y=y_val,
        theta=model.theta, save_path=image_path
    )
    # Use np.savetxt to save predictions on eval set to save_path
    prob_val = model.predict(x_val)
    np.savetxt(save_path, prob_val)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
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

    def fit(self, x, y) -> None:
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape
        if self.theta is None:
            self.theta = np.zeros(d)

        def g(z): return 1 / (1 + np.exp(-z))

        def hessian(x, y):
            """A helper function to compute the hessian"""
            h = np.zeros([d, d])
            for i in range(d):
                for j in range(d):
                    h[i][j] = np.mean(
                        g(np.matmul(x, self.theta)) * (1 - g(np.matmul(x, self.theta))) * x[:, i] * x[:, j]
                    )
            return h

        delta = np.inf  # change in parameter to determine convergence.
        step = 0  # iteration
        while delta >= self.eps and step < self.max_iter:
            nabla = np.zeros(d)
            for i in range(d):
                nabla[i] = - np.mean(
                    (y - g(np.matmul(x, self.theta))) * x[:, i]
                )
            h = hessian(x, y)
            assert h.shape == (d, d)
            new_theta = self.theta - np.matmul(np.linalg.inv(h), nabla)
            assert new_theta.shape == self.theta.shape
            # update
            delta = np.linalg.norm(new_theta - self.theta)
            self.theta = new_theta
            if self.verbose:
                print(f"Newton's method epoch {step}: parameter change: {delta}.")
            step += 1
        print(f"Newton's method converges after {step} epochs.")
        # *** END CODE HERE ***

    def predict(self, x) -> np.array:
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        def g(z): return 1 / (1 + np.exp(-z))
        prob = g(np.matmul(x, self.theta))
        assert prob.shape == (x.shape[0],)
        return prob
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
