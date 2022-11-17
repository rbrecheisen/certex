import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np


BATCH_SIZE = 64
BUFFER_SIZE = 1000


def resize(input_img, input_msk):
    input_img = tf.image.resize(input_img, (128, 128), method='nearest')
    input_msk = tf.image.resize(input_img, (128, 128), method='nearest')
    return input_img, input_msk


def augment(input_img, input_msk):
    if tf.random.uniform(()) > 0.5:
        input_img = tf.image.flip_left_right(input_img)
        input_msk = tf.image.flip_left_right(input_msk)
    return input_img, input_msk


def normalize(input_img, input_msk):
    input_img = tf.cast(input_img, tf.float32) / 255.0
    input_msk -= 1
    return input_img, input_msk


def load_img_tr(datapoint):
    input_img = datapoint['image']
    input_msk = datapoint['segmentation_mask']
    input_img, input_msk = resize(input_img, input_msk)
    input_img, input_msk = augment(input_img, input_msk)
    input_img, input_msk = normalize(input_img, input_msk)
    return input_img, input_msk


def load_img_te(datapoint):
    input_img = datapoint['image']
    input_msk = datapoint['segmentation_mask']
    input_img, input_msk = resize(input_img, input_msk)
    # skip augmentation
    input_img, input_msk = normalize(input_img, input_msk)
    return input_img, input_msk


def display(display_list):
    # plt.figure(figsize=(15, 15))
    title = ['Input image', 'True mask', 'Predicted mask']
    for i in range(len(display_list)):
        print(np.unique(display_list[i]))
        # plt.subplot(1, len(display_list), i+1)
        # plt.title(title[i])
        # plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        # plt.axis('off')
    # plt.show()


def main():
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    dataset_tr = dataset['train']\
        .map(load_img_tr, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_te = dataset['test']\
        .map(load_img_te, num_parallel_calls=tf.data.AUTOTUNE)
    batches_tr = dataset_tr.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    batches_tr = batches_tr.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    batches_va = dataset_te.take(3000).batch(BATCH_SIZE)
    batches_te = dataset_te.skip(3000).take(669).batch(BATCH_SIZE)
    sample_batch = next(iter(batches_tr))
    random_idx = np.random.choice(sample_batch[0].shape[0])
    sample_img, sample_msk = sample_batch[0][random_idx], sample_batch[1][random_idx]
    display([sample_img, sample_msk])


if __name__ == '__main__':
    main()
