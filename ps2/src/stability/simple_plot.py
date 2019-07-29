import matplotlib.pyplot as plt


def plot(x, y, save_path=None, abline: bool = False):
    """Visualize dataset for binary classification.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        save_path: Path to save the plot. If None, plot is shown using GUI.
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')

    if abline:
        abline_x = [0, 1]
        abline_y = [1, 0]
        plt.plot(abline_x, abline_y)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

