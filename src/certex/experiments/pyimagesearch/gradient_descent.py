import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs

EPOCHS = 100
ALPHA = 0.01


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def predict(X, W):
    predictions = sigmoid_activation(X.dot(W))
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0] = 1
    return predictions


def main():
    X, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
    y = y.reshape((y.shape[0], 1))
    X = np.c_[X, np.ones(X.shape[0])]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=4)
    print('[INFO] training...')
    W = np.random.randn(X.shape[1], 1)
    losses = []
    for epoch in np.arange(0, EPOCHS):
        predictions = sigmoid_activation(train_x.dot(W))
        error = predictions - train_y
        loss = np.sum(error ** 2)
        losses.append(loss)
        d = error * sigmoid_deriv(predictions)
        gradient = train_x.T.dot(d)
        W += ALPHA * gradient
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print('[INFO] epoch={}, loss={:.7f}'.format(int(epoch+1), loss))
    print('[INFO] evaluating...')
    predictions = predict(test_x, W)
    print(classification_report(test_y, predictions))
    plt.style.use('ggplot')
    plt.figure()
    plt.title('Data')
    plt.scatter(test_x[:, 0], test_x[:, 1], marker='o', c=test_y[:, 0], s=30)
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), losses)
    plt.title('Training loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.show()

    # THIS IS NOT GIVING THE RIGHT OUTPUT!
    # TRAINING LOSS SKYROCKETS...


if __name__ == '__main__':
    main()
