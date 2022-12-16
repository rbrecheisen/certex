import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def main():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    # Could be you get CERTIFICATE_ERROR when downloading. Go to /Applications/Python 3.9 and double-click
    # Install certificates.command
    # https://intellij-support.jetbrains.com/hc/en-us/community/posts/360000117730-SSL-error-in-accessing-MNIST-dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(train_images.shape)
    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()
    # Arrays are read-only so make copy first!
    train_images = train_images.copy() / 255.0
    test_images = test_images.copy() / 255.0
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Test accuracy: {test_accuracy}')
    prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = prob_model.predict(test_images)
    print(predictions[0])
    label = np.argmax(predictions[0])
    print(label)
    print(class_names[label])


if __name__ == '__main__':
    main()
