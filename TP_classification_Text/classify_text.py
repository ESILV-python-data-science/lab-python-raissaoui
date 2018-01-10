# -*- coding: utf-8 -*-
"""
Classify digit images

R. Aissaoui - 2018
"""

import argparse
import logging
import time
import sys
import math

import matplotlib
from tqdm import tqdm
import pandas as pd
# import tokenize
# import nltk
# from PIL import Image, ImageFilter
# from sklearn.cluster import KMeans
from sklearn import svm, metrics, neighbors
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
# Setup logging
logger = logging.getLogger('classify_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)
MISSING = object()  # silly way to check if argument is missing

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

def test_summary(LearningObject, X_train, y_train, X_test, y_test, X_valid=MISSING, y_valid=MISSING, other_options=False):
    # train
    logger.info("------------------------------------TRAIN SET--------------------------------")
    test(LearningObject, X_train, y_train, other_options)

    # test
    logger.info("------------------------------------TEST SET--------------------------------")
    test(LearningObject, X_test, y_test, other_options)

    # validation
    if X_valid is not MISSING and y_valid is not MISSING:
        logger.info("------------------------------------VALID SET--------------------------------")
        return test(LearningObject, X_valid, y_valid, other_options=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Extract features, train a classifier on texts and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    # input_group.add_argument('--load-csv', help='read text from csv file')
    parser.add_argument('--save-text', help='save text in pickle format')

    args = parser.parse_args()

    if args.load_csv:
        df = pd.read_csv("LeMonde2003.csv", sep='\t', engine='python')
        df = df.dropna(how="all")
        categories = ['ENT', 'INT', 'ART', 'SOC', 'FRA', 'SPO', 'LIV', 'TEL', 'UNE']
        data = df[df.category.isin(categories)]
        # logger.info('Saved data to data:', data)

    Y = df['category']
    X = np.array(data)

    if args.save_text:
        # convert X to dataframe with pd.DataFrame and save to pickle with to_pickle
        df_features = pd.DataFrame(X)
        df_features['category'] = Y
        df_features.to_pickle('save_text.pickle')

    # endregion

        logger.info('Saved {} text and category to {}'.format(df_features.shape, args.save_text))

    if args.features_only:
        logger.info('No classifier to train, exit')
        sys.exit()

    else:
        logger.error('No classifier specified')
        sys.exit()

