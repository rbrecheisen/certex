import os
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import ImageOps


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
    img_dir = '/Users/Ralph/Desktop/images'
    lab_dir = '/Users/Ralph/Desktop/annotations/trimaps'
    img_size = (160, 160)
    num_classes = 3
    batch_size = 32
    img_paths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg')])
    lab_paths = sorted([os.path.join(lab_dir, fname) for fname in os.listdir(lab_dir) if fname.endswith('.png')])
    print(f'nr. samples: {len(img_paths)}')
    for img_path, lab_path in zip(img_paths[:10], lab_paths[:10]):
        print(img_path, '|', lab_path)
    display(img_paths[9], lab_paths[9])


if __name__ == '__main__':
    main()
