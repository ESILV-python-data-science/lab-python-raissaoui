import argparse
import logging
import time
import sys

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm, metrics
import numpy as np

logger = logging.getLogger('mnist_svm.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def test(LearningObject, X, y, other_options=False):
    predictions = LearningObject.predict(X)
    accuracy = accuracy_score(y, predictions)
    if other_options:
        logger.info(
            "accuracy = {}".format(accuracy))
        logger.info(
            "confusion matrix:\n {}".format(metrics.confusion_matrix(y, predictions)))
        logger.info(
            "Classification report: \n {}".format(classification_report(y, predictions)))
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract features, train a classifier on images and test the classifier')
    parser.add_argument('--load-features', help='read features and class from pickle file')
    parser.add_argument('--limit-samples', type=int, help='limit the number of samples to consider for training')
    methods = parser.add_mutually_exclusive_group(required=True)
    methods.add_argument('--kernel', help='optimize a SVM classifier on the MNIST database.')
    methods.add_argument('--tuned-parameters',action='store_true', help='optimize a SVM classifier on the MNIST database.')

    args = parser.parse_args()

    df_features = pd.read_pickle(args.load_features)
    y = df_features['class']
    X = df_features.drop('class', axis=1)

    if args.limit_samples:
        X = X.head(args.limit_samples)
        y = y.head(args.limit_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    logger.info("Train set size is {}".format(X_train.shape))
    logger.info("Test set size is {}".format(X_test.shape))
    logger.info('Use the svm model to classify the MINST data')

    if args.kernel:

        # --load-features save_features.pickle --kernel linear
        if args.kernel == "linear":
            # Answer 9
            logger.info("we'll use a linear kernel")
            clf = SVC(kernel='linear')
            logger.info("we'll continue")
            clf.fit(X_train, y_train)
            logger.info("we'll get there")
            test(clf, X_train, y_train, other_options=True)
            logger.info("Almost done")
            test(clf, X_test, y_test, other_options=True)
            logger.info("Finally")

        # --load-features save_features.pickle --kernel RBF
        if args.kernel == "RBF":
            # Answer 11
            logger.info("we'll use a RBF kernel")
            clf = SVC(kernel='rbf')
            logger.info("we'll continue")
            clf.fit(X_train, y_train)
            logger.info("we'll get there")
            test(clf, X_train, y_train, other_options=True)
            logger.info("Almost done")
            test(clf, X_test, y_test, other_options=True)
            logger.info("Finally")

    if args.tuned_parameters:
        # Anwser 11 bis
        logger.info("we'll use a RBF kernel, with tuned parameters")
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]}]
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='accuracy')
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        test(clf, X_train, y_train, other_options=True)
        test(clf, X_test, y_test, other_options=True)