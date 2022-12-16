import os
import cv2
import random
import numpy as np


class SimpleDatasetLoader:

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths, verbose=-1, max_nr=0):
        data = []
        labels = []
        nr_image_paths = len(image_paths)
        if max_nr > 0:
            nr_image_paths = max_nr
            random.shuffle(image_paths)
        count = 0
        for (i, image_path) in enumerate(image_paths):
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print('[INFO] processed {}/{}'.format(i+1, nr_image_paths))
            if count > nr_image_paths:
                break
            count += 1
        return np.array(data), np.array(labels)
