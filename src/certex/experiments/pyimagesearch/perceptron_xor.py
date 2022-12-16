import numpy as np
from pyimagesearch.nn.perceptron import Perceptron


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print('[INFO] training perceptron...')
p = Perceptron(X.shape[1])
p.fit(X, y, epochs=20)

print('[INFO] testing perceptron...')
for x, target in zip(X, y):
    prediction = p.predict(x)
    print(f'[INFO] data={x}, ground_truth={target[0]}, prediction={prediction}')
