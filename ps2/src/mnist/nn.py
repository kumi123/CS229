import numpy as np
import matplotlib.pyplot as plt
import argparse


def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    B, K = x.shape
    # The softmax is invariant for constant c: exp(z) / sum[exp(z)] = exp(z-c) / sum[exp(z-c)]
    max_array = np.max(x, axis=1, keepdims=True)  # Keep dimensions for boardcasting.
    assert max_array.shape == (B, 1)
    x_normalized = x - max_array
    result = np.exp(x_normalized) / np.sum(np.exp(x_normalized), axis=1, keepdims=True)
    assert result.shape == (B, K)
    return result
    # *** END CODE HERE ***


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    # return 1 / (1 + np.exp(-x))
    # robust against overflow.
    array = np.zeros_like(x)
    array[x > 0] = 1 / (1 + np.exp(-x[x > 0]))
    array[x <= 0] = np.exp(x[x <= 0]) / (np.exp(x[x <= 0]) + 1)
    return array
    # *** END CODE HERE ***


def get_initial_params(input_size, num_hidden, num_output) -> dict:
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer
    b1 is the bias vector for the hidden layer
    W2 is the weight matrix for the output layers
    b2 is the bias vector for the output layer

    As specified in the PDF, weight matrices should be initialized with a random normal distribution.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    W1 = np.random.randn(input_size, num_hidden)
    b1 = np.zeros(num_hidden)
    W2 = np.random.randn(num_hidden, num_output)
    b2 = np.zeros(num_output)
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return params
    # *** END CODE HERE ***


def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    N, P = data.shape  # P: num features
    K = labels.shape[1]
    a_hidden = sigmoid(np.matmul(data, params["W1"]) + params["b1"])
    # print("Hidden shape:", a_hidden.shape)
    out = softmax(np.matmul(a_hidden, params["W2"]) + params["b2"])
    # print("Out shape:", out.shape)
    epsilon = 1e-32  # In some extreme cases, out = 0.
    loss = - np.multiply(labels, np.log(out + epsilon))
    loss = loss.sum() / N
    # loss = -1.0 * np.array([
    #     np.dot(labels[i, :], out[i, :])
    #     for i in range(N)
    #     ])
    # print("loss shape:", loss.shape)
    # loss = np.mean(loss)
    # print("loss: ", loss)
    return a_hidden, out, loss
    # *** END CODE HERE ***


def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    N, input_size = data.shape
    K = labels.shape[1]
    H = params["W2"].shape[0]  # Hidden size
    # Retrive params
    W1 = params["W1"]
    W2 = params["W2"]
    b1 = params["b1"]
    b2 = params["b2"]
    # Report
    # print(f"Input size: {input_size}, num labels: {K}, hidden size: {H}")
    # Forward pass
    a1, a2, l = forward_prop_func(data, labels, params)
    # ==== Vectorized Version
    delta2 = (a2 - labels).reshape(N, K)
    delta1 = (delta2 @ W2.T) * (a1 * (1 - a1))

    dW2 = a1.T @ delta2
    assert dW2.shape == W2.shape
    dW1 = data.T @ delta1
    assert dW1.shape == W1.shape

    db2 = np.sum(delta2, axis=0)
    # assert np.all(db2.shape == b2), f"Shape got: {db2.shape}, expected: {b2.shape}"
    assert len(db2) == len(b2)
    db1 = np.sum(delta1, axis=0)
    # assert np.all(db1.shape == b1), f"Shape got: {db1.shape}, expected: {b1.shape}"
    assert len(db2) == len(b2)
    return {
        "W1": dW1 / N,
        "W2": dW2 / N,
        "b1": db1 / N,
        "b2": db2 / N,
    }
    # *** END CODE HERE ***


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    unreg_grads = backward_prop(data, labels, params, forward_prop_func)
    W1 = params["W1"]
    W2 = params["W2"]
    grads = dict()
    grads["W1"] = unreg_grads["W1"] + (2 * reg) * W1
    grads["W2"] = unreg_grads["W2"] + (2 * reg) * W2
    grads["b1"] = unreg_grads["b1"]
    grads["b2"] = unreg_grads["b2"]
    return grads
    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The am2c: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    # perform batch gradient descent
    N, input_size = train_data.shape
    split = np.linspace(N, batch_size).astype(int)[::-1]
    for index in split:
        # each mini batch
        x = train_data[index-batch_size:index, :]
        y = train_labels[index-batch_size:index, :]
        grads = backward_prop_func(x, y, params, forward_prop_func)
        # update gradient
        for var in params.keys():
            params[var] -= learning_rate * grads[var]
    # *** END CODE HERE ***

    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        print("Training Step: ", epoch)
        gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output, train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t, cost_train,'r', label='train')
    ax1.plot(t, cost_dev, 'b', label='dev')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    # ax1.set_title('With Regularization')
    ax1.set_title(name)
    ax1.legend()

    ax2.plot(t, accuracy_train,'r', label='train')
    ax2.plot(t, accuracy_dev, 'b', label='dev')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy')
    ax2.legend()

    fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))

def main():
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:10000,:]
    dev_labels = train_labels[0:10000,:]
    train_data = train_data[10000:,:]
    train_labels = train_labels[10000:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }

    run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs)
    run_train_test('regularized', all_data, all_labels, 
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs)

if __name__ == '__main__':
    main()
