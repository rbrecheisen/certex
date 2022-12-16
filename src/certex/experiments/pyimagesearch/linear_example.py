import cv2
import numpy as np


def main():
    labels = ['dog', 'cat', 'panda']
    np.random.seed(1)
    W = np.random.randn(3, 3072)
    b = np.random.randn(3)
    org = cv2.imread('beagle.png')
    image = cv2.resize(org, (32, 32)).flatten()
    scores = W.dot(image) + b
    for (label, score) in zip(labels, scores):
        print('[INFO] {}: {:.2f}'.format(label, score))
    cv2.putText(org, 'Label: {}'.format(labels[np.argmax(scores)]), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Image', org)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
