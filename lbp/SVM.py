
import numpy as np

class SVM_RBF:
    def __init__(self, learning_rate=0.01, regularization_strength=0.1, gamma=0.1, num_epochs=1000):
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.gamma = gamma
        self.num_epochs = num_epochs

    def _rbf_kernel(self, X1, X2):

        pairwise_sq_dists = np.sum(X1 ** 2, axis=1)[:, np.newaxis] + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * pairwise_sq_dists)

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y
        self.alpha = np.zeros(n_samples)

        # Training
        for epoch in range(self.num_epochs):
            for i in range(n_samples):
                kernel_i = self._rbf_kernel(X[i:i + 1], self.X_train)
                prediction = np.sum(self.alpha * self.y_train * kernel_i)

                if y[i] * prediction < 1:
                    self.alpha[i] += self.learning_rate * (1 - y[i] * prediction)
                else:
                    self.alpha[i] -= self.learning_rate * self.regularization_strength * self.alpha[i]
            self.alpha = np.clip(self.alpha, 0, 1)
        self.support_vectors_idx = self.alpha > 1e-5
        self.support_vectors = self.X_train[self.support_vectors_idx]
        self.support_vector_labels = self.y_train[self.support_vectors_idx]
        self.alpha_support_vectors = self.alpha[self.support_vectors_idx]

        self.bias = np.mean(
            self.support_vector_labels - np.sum(self.alpha_support_vectors * self.support_vector_labels *
                                                self._rbf_kernel(self.support_vectors, self.support_vectors), axis=0))

    def predict(self, X):

        kernel = self._rbf_kernel(X, self.support_vectors)
        predictions = np.dot(kernel, self.alpha_support_vectors * self.support_vector_labels) + self.bias
        return np.sign(predictions)

    def score(self, X, y):

        predictions = self.predict(X)
        return np.mean(predictions == y)