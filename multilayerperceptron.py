import numpy as np
from tqdm import tqdm


class MultiLayerPerceptron():
    """A feedforward neural network"""
    def __init__(self, architecture, lambda_=0):
        self.architecture = architecture
        self.depth = len(architecture) - 1
        self.lambda_ = lambda_
        self.output_layer = architecture[-1][1]

        # Randomly initialize weights
        self.thetas = []
        for i in range(self.depth):
            init_theta = self.rand_initialize_weights(
                architecture[i][0],
                architecture[i + 1][0])
            self.thetas.append(init_theta)

    def rand_initialize_weights(self, l_in, l_out):
        epsilon_init = np.sqrt(6) / np.sqrt(l_in + l_out)
        weights = np.random.default_rng(1337).uniform(size=(l_out, l_in + 1))
        weights = weights * 2 * epsilon_init - epsilon_init
        return weights

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exps = np.exp(z - z.max(axis=1, keepdims=True))
        return exps / exps.sum(axis=1, keepdims=True)

    def feedforward(self, X, training=False):
        m = X.shape[0]
        intercept = np.ones((m, 1))
        activations = []
        masks = []
        dropout_rate = self.architecture[0][2]
        if dropout_rate > 0 and training:
            dropout_mask = np.random.default_rng().binomial(1, 1 - dropout_rate,
                                              size=a.shape)
            X = X * dropout_mask / (1 - dropout_rate)
        else:
            dropout_mask = None
        a = np.concatenate((intercept, X), axis=1)
        activations.append(a)
        masks.append(dropout_mask)
        for i in range(self.depth - 1):
            activation = self.architecture[i + 1][1]
            dropout_rate = self.architecture[i + 1][2]
            if activation == 'sigmoid':
                a = self.sigmoid(a @ self.thetas[i].T)
            elif activation == 'relu':
                a = self.relu(a @ self.thetas[i].T)
            if dropout_rate > 0 and training:
                dropout_mask = np.random.default_rng().binomial(1, 1 - dropout_rate,
                                                  size=a.shape)
                a = a * dropout_mask / (1 - dropout_rate)
            else:
                dropout_mask = None
            a = np.concatenate((intercept, a), axis=1)
            activations.append(a)
            masks.append(dropout_mask)
        if self.output_layer == 'softmax':
            h = self.softmax(a @ self.thetas[-1].T)
        elif self.output_layer == 'sigmoid':
            h = self.sigmoid(a @ self.thetas[-1].T)
        activations.append(h)
        masks.append(None)
        return activations, masks

    def crossentropy_loss(self, y_matrix, h, reg=False):
        m = h.shape[0]
        if self.output_layer == 'sigmoid':
            J = - (y_matrix * np.log(h + 1e-8)
                   + (1 - y_matrix) * np.log(1 - h + 1e-8)).sum() / m
        elif self.output_layer == 'softmax':
            J = - (y_matrix * np.log(h)).sum() / m

        if reg:
            # Add regularization term
            J = J + self.lambda_ / (2 * m) * sum(
                [(theta[:, 1:] ** 2).sum() for theta in self.thetas])
        return J

    def nn_cost_function(self, X, y, grads=False, training=False):
        # Feedforward the neural network
        activations, masks = self.feedforward(X, training=training)
        h = activations.pop()
        masks.pop()

        # One-hot encode the labels
        n_out = self.architecture[-1][0]
        if n_out > 1:
            labels_vec = np.eye(n_out)
            y_matrix = labels_vec[y.ravel(), :]
        else:
            y_matrix = y

        # Calculate the cost function
        J = self.crossentropy_loss(y_matrix, h, self.lambda_ > 0)
        if not grads:
            return J

        # Backpropagation
        m = X.shape[0]
        thetas_grad = []
        delta = h - y_matrix
        theta_grad = delta.T @ activations[-1] / m
        thetas_grad.append(theta_grad)
        for i in range(self.depth - 1):
            activation, dropout_rate = self.architecture[-2 - i][1], self.architecture[-2 - i][2]
            dropout_mask = masks[-1 - i]
            if activation == 'sigmoid':
                activation_derivative = activations[-1 - i] *\
                    (1 - activations[-1 - i])
            elif activation == 'relu':
                activation_derivative = (activations[-1 - i] > 0).astype(int)
            delta = (delta @ self.thetas[-1 - i])
            if dropout_rate > 0 and training:
                delta[:, 1:] = delta[:, 1:] * dropout_mask / (1 - dropout_rate)
            delta = delta * activation_derivative
            delta = delta[:, 1:]
            theta_grad = delta.T @ activations[-2 - i] / m
            thetas_grad.append(theta_grad)
        thetas_grad = thetas_grad[::-1]

        # Add regularization term
        for i in range(self.depth):
            reg = self.thetas[i].copy()
            reg[:, 0] = 0
            thetas_grad[i] = thetas_grad[i] + self.lambda_ * reg / m

        return (J, thetas_grad)

    def fit(self, x_train, y_train, x_valid, y_valid, alpha, epochs,
            batch_size=-1):
        indices = range(batch_size, x_train.shape[0], batch_size)
        x_batches = np.split(x_train, indices)
        y_batches = np.split(y_train, indices)
        J_train_history, J_valid_history = [], []
        for i in range(epochs):
            J_train_batches = []
            for x_batch, y_batch in tqdm(
                zip(x_batches, y_batches),
                total=len(x_batches),
                leave=True,
                desc=f'epoch {i + 1}/{epochs}'):
                J_train_batch, thetas_grad = self.nn_cost_function(
                    x_batch, y_batch, grads=True, training=True)
                J_train_batches.append(J_train_batch)
                for theta, theta_grad in zip(self.thetas, thetas_grad):
                    theta -= alpha * theta_grad
            J_train = np.mean(J_train_batches)
            J_valid = self.nn_cost_function(x_valid, y_valid)
            J_train_history.append(J_train)
            J_valid_history.append(J_valid)
            print(f'epoch {i + 1}/{epochs} - loss: {J_train:.4f}\
 - val_loss: {J_valid:.4f}')

            # Early stopping
            min_J_valid_idx = np.argmin(J_valid_history)
            if i - min_J_valid_idx > 10:
                break

        return J_train_history, J_valid_history

    def predict(self, X, verbose=False, y=None):
        activations, _ = self.feedforward(X, training=False)
        h = activations[-1]
        if self.output_layer == 'softmax':
            predictions = np.argmax(h, axis=1).reshape(-1, 1)
        elif self.output_layer == 'sigmoid':
            predictions = h
        if (verbose and self.output_layer == 'softmax'):
            for _, raw in zip(np.hstack([y, predictions]), h):
                if (_[0] == _[1]):
                    print(f'-> {tuple(_)} - raw{raw}')
                else:
                    print(f'-> {tuple(_)} - raw{raw} <<')
        return predictions