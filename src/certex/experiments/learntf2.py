import tensorflow as tf


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


def main():
    data = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = data.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis].astype('float32')
    x_test = x_test[..., tf.newaxis].astype('float32')
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    model = MyModel()

    @tf.function
    def train_step(imgs, lbls):
        with tf.GradientTape() as tape:
            predictions = model(imgs, training=True)
            loss = loss_object(lbls, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(lbls, predictions)

    @tf.function
    def test_step(imgs, lbls):
        predictions = model(imgs, training=False)
        t_loss = loss_object(lbls, predictions)
        test_loss(t_loss)
        test_accuracy(lbls, predictions)

    for epoch in range(5):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for images, labels in train_ds:
            train_step(images, labels)
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
        print(
            f'Epoch {epoch+1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result()}, '
            f'Test loss: {test_loss.result()}, '
            f'Test accuracy: {test_accuracy.result() * 100}'
        )


if __name__ == '__main__':
    main()
