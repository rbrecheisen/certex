import os

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.data.simpledatasetloader import SimpleDatasetLoader

DATASET = '/Users/Ralph/data/surfdrive/documents/cds/deeplearning/rosebrock/starter/code_and_data/datasets'


def main():
    print('[INFO] loading images...')
    image_paths = []
    for root, dirs, files in os.walk(DATASET):
        for f in files:
            if f.endswith('.jpg'):
                image_paths.append(os.path.join(root, f))
    preprocessor = SimplePreprocessor(32, 32)
    loader = SimpleDatasetLoader(preprocessors=[preprocessor])
    data, labels = loader.load(image_paths, verbose=500, max_nr=5000)
    data = data.reshape((data.shape[0], 3072))
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25, random_state=42)
    for r in (None, 'l1', 'l2'):
        print(f'[INFO] training model with {r} penalty...')
        model = SGDClassifier(loss='log', penalty=r, max_iter=10, learning_rate='constant',
                              tol=1e-3, eta0=0.01, random_state=12)
        model.fit(train_x, train_y)
        accuracy = model.score(test_x, test_y)
        print(f'[INFO] {r} penalty accuracy: {accuracy * 100}')


if __name__ == '__main__':
    main()
