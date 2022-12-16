import argparse
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.data.simpledatasetloader import SimpleDatasetLoader

DATASET = '/Users/Ralph/data/surfdrive/documents/cds/deeplearning/rosebrock/starter/code_and_data/datasets'
NEIGHBORS = 1
JOBS = -1


def main():
    print('[INFO] loading images...')
    image_paths = []
    for root, dirs, files in os.walk(DATASET):
        for f in files:
            if f.endswith('.jpg'):
                image_paths.append(os.path.join(root, f))
    preprocessor = SimplePreprocessor(32, 32)
    loader = SimpleDatasetLoader(preprocessors=[preprocessor])
    data, labels = loader.load(image_paths, verbose=500, max_nr=3000)
    data = data.reshape((data.shape[0], 3072))
    print('[INFO] features matrix: {:.1f}MB'.format(data.nbytes / (1024 * 1024.0)))
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25, random_state=42)
    print('[INFO] evaluating k-NN classifier...')
    model = KNeighborsClassifier(n_neighbors=NEIGHBORS, n_jobs=JOBS)
    model.fit(train_x, train_y)
    print(classification_report(test_y, model.predict(test_x), target_names=encoder.classes_))

# OUTPUT:
# -----------------------------------------------------
#               precision    recall  f1-score   support
#
#         cats       0.47      0.35      0.40       239
#         dogs       0.36      0.19      0.25       263
#   negatives7       0.76      0.83      0.79      2396
#        panda       0.89      0.24      0.38       229
#   positives7       0.51      0.58      0.54       915
#
#     accuracy                           0.67      4042
#    macro avg       0.60      0.44      0.47      4042
# weighted avg       0.66      0.67      0.65      4042
#
# Comments:
# ---------
# Precision, also called Positive Predictive Value, is the fraction of relevant instances
# among the retrieved instances. Recall, also called sensitivity, is the fraction of
# relevant instances retrieved in total. Both are measures of relevance.
# F1-score is a combination of precision and recall in a single number.
# Support is the number of occurences of each class in y_true


if __name__ == '__main__':
    main()
