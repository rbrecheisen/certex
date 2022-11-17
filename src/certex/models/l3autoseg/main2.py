import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import ImageOps

IMG_DIR = '/Users/Ralph/Desktop/images'
LAB_DIR = '/Users/Ralph/Desktop/annotations/trimaps'
MOD_DIR = '/Users/Ralph/Desktop/model'
IMG_SIZE = (160, 160)
NUM_CLASSES = 3
BATCH_SIZE = 32


class OxfordPets(tf.keras.utils.Sequence):

    def __init__(self, batch_size, img_size, img_paths, lab_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_paths = img_paths
        self.lab_paths = lab_paths

    def __len__(self):
        return len(self.lab_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_img_paths = self.img_paths[i:i+self.batch_size]
        batch_lab_paths = self.lab_paths[i:i+self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='float32')
        for j, path in enumerate(batch_img_paths):
            img = tf.keras.preprocessing.image.load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,), self.img_size + (1,), dtype='uint8')
        for j, path in enumerate(batch_lab_paths):
            img = tf.keras.preprocessing.image.load_img(path, target_size=self.img_size, color_mode='grayscale')
            y[j] = np.expand_dims(img, 2)
            y[j] -= 1
        return x, y


def get_model(img_size, num_classes):
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    previous_block_activation = x
    for filters in [64, 128, 256]:
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
        residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding='same')(previous_block_activation)
        x = tf.keras.layers.add([x, residual])
        previous_block_activation = x
    for filters in [256, 128, 64, 32]:
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D(2)(x)
        residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = tf.keras.layers.Conv2D(filters, 1, padding='same')(residual)
        x = tf.keras.layers.add([x, residual])
        previous_block_activation = x
    outputs = tf.keras.layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def display(img_path, lab_path):
    img = tf.keras.preprocessing.image.load_img(img_path)
    lab = ImageOps.autocontrast(tf.keras.preprocessing.image.load_img(lab_path))
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.title('Input image')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('True mask')
    plt.imshow(lab)
    plt.axis('off')
    plt.show()


# curl -O https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
# curl -O https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
def main():
    img_paths = sorted([os.path.join(IMG_DIR, fname) for fname in os.listdir(IMG_DIR) if fname.endswith('.jpg')])
    lab_paths = sorted([os.path.join(LAB_DIR, fname) for fname in os.listdir(LAB_DIR) if fname.endswith('.png')])
    print(f'nr. samples: {len(img_paths)}')
    for img_path, lab_path in zip(img_paths[:10], lab_paths[:10]):
        print(img_path, '|', lab_path)
    # display(img_paths[9], lab_paths[9])
    tf.keras.backend.clear_session()
    model = get_model(IMG_SIZE, NUM_CLASSES)
    model.summary()
    val_samples = 1000
    random.Random(1337).shuffle(img_paths)
    random.Random(1337).shuffle(lab_paths)
    train_img_paths = img_paths[:-val_samples]
    train_lab_paths = lab_paths[:-val_samples]
    validation_img_paths = img_paths[-val_samples:]
    validation_lab_paths = lab_paths[-val_samples:]
    train_data = OxfordPets(BATCH_SIZE, IMG_SIZE, train_img_paths, train_lab_paths)
    validation_data = OxfordPets(BATCH_SIZE, IMG_SIZE, validation_img_paths, validation_lab_paths)
    model.compile(loss='sparse_categorical_crossentropy')  # optimizer=rmsprop
    callbacks = [tf.keras.callbacks.ModelCheckpoint('oxford_seg.h5', save_best_only=True)]
    epochs = 15
    model.fit(train_data, epochs=epochs, validation_data=validation_data, callbacks=callbacks)
    model.save(MOD_DIR)
    model.save_weights(MOD_DIR)


def validate(validation_img_paths, validation_lab_paths):
    model = tf.keras.models.load_model(MOD_DIR)
    validation_data = OxfordPets(BATCH_SIZE, IMG_SIZE, validation_img_paths, validation_lab_paths)
    validation_predictions = model.predict(validation_data)
    print(validation_predictions)


if __name__ == '__main__':
    main()
