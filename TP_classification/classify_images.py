# -*- coding: utf-8 -*-
"""
Classify digit images

C. Kermorvant - 2017
"""

import argparse
import logging
import time
import sys
import math

from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn import svm, metrics, neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import numpy as np
# Setup logging
logger = logging.getLogger('classify_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)
MISSING = object()  # silly way to check if argument is missing


def extract_features_subresolution(img, img_feature_size=(8, 8)):
    """
        Compute the subresolution of an image and return it as a feature vector

        :param img: the original image (can be color or gray)
        :type img: pillow image
        :return: pixel values of the image in subresolution
        :rtype: list of int in [0,255]

        """
    # Answer 2

    # convert color images to grey level
    gray_img = img.convert('L')
    # find the min dimension to rotate the image if needed
    min_size = min(img.size)
    if img.size[1] == min_size:
        # convert landscape  to portrait
        rotated_img = gray_img.rotate(90, expand=1)
    else:
        rotated_img = gray_img

    # reduce the image to a given size
    reduced_img = rotated_img.resize(
        img_feature_size, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]


def knn_train(k, X_train, y_train):
    # knn_train() il permet juste de lancer un model d'apprentissage knn en lui donnant le K souhaité
    knnObject = KNeighborsClassifier(n_neighbors=k)
    knnObject.fit(X_train, y_train)
    return knnObject


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


def nearestNeighbors_train(X_train, y_train, k=MISSING, limit=10):
    # s'entrainer sur une serie de KNN
    # elle renvoie une liste contenant tout les modèles
    # meme si on utilise qu'une seule valeur  (si K is not MISSING), il faut renvoyer une liste
    # pour avoir une facon unique d'évaleur le modèle  (cf. nearestNeighbors_test)
    t0 = time.time()
    knns = []
    if k is MISSING:
        # train with multiple values
        for i in range(1, limit + 1):
            knns.append(knn_train(i, X_train, y_train))
    else:
        # train with one value only
        knns.append(knn_train(k, X_train, y_train))
    logger.info("Training  done in %0.3fs" % (time.time() - t0))
    return knns


def nearestNeighbors_test(knns, X_train, y_train, X_test, y_test, X_valid=MISSING, y_valid=MISSING):
    # tester les knn modèles sur leurs bases VALID (pas test) et recupérer leurs accuracy
    # Si knns.length > 1 :  plotter leurs accuracy en fonction de K
    # Si knns.length == 1 : évaluer son accuracy et s'arreter la .
    t0 = time.time()
    k = 1
    accuracies = []
    for knn in knns:
        logger.info("Testing KNN with k={}".format(k))
        accuracies.append(test_summary(knn, X_train, y_train, X_test, y_test, X_valid, y_valid))
        k += 1
    logger.info("Testing  done in %0.3fs" % (time.time() - t0))
    if k != 2:  # when it's more than one element in the accuracies' list
        plt.plot(range(1, k), accuracies, marker='o', linestyle='--', color='r')
        plt.axis([1, k, min(accuracies) - 0.005, max(accuracies) + 0.005])
        plt.xlabel('KNN neighbors')
        plt.ylabel('Accuracy')
        plt.title('Accuracies with different K in KNN algorithm')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract features, train a classifier on images and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images-list',
                             help='file containing the image path and image class, one per line, comma separated')
    input_group.add_argument('--load-features', help='read features and class from pickle file')
    parser.add_argument('--save-features', help='save features in pickle format')
    parser.add_argument('--limit-samples', type=int, help='limit the number of samples to consider for training')
    classifier_group = parser.add_mutually_exclusive_group(required=True)
    classifier_group.add_argument('--nearest-neighbors', type=int)
    classifier_group.add_argument('--logistic-regression', action='store_true',
                                  help='train the model using logistic regression')
    classifier_group.add_argument('--features-only', action='store_true',
                                  help='only extract features, do not train classifiers')
    classifier_group.add_argument('--learning-curve', action='store_true',
                                  help='study the impact of a growing training set on the accuracy.')
    classifier_group.add_argument('--testing-curve', action='store_true',
                                  help='study the impact of a growing testing set on the accuracy.')
    args = parser.parse_args()

    if args.load_features:
        df_features = pd.read_pickle(args.load_features)
        y = df_features['class']
        X = df_features.drop('class', axis=1)

    else:
        # region Answer 1
        # file = "MNIST_all.csv"
        # Load the image list from CSV file using pd.read_csv
        # df_features = pd.read_csv(file, na_values=['.'], error_bad_lines=False,sep=";")
        # see the doc for the option since there is no header ;
        # specify the column names :  filename , class

        file_list = pd.read_csv(args.images_list, header=None, names=['filename', 'class'], engine='python')

        # logger.info('Loaded {} images in {}'.format(all_df.shape,args.images_list))
        # endregion

        # region Answer 2
        # Extract the feature vector on all the pages found
        # Modify the extract_features from TP_Clustering to extract 8x8 subresolution values
        # white must be 0 and black 255
        data = []
        for i_path in tqdm(file_list['filename']):
            page_image = Image.open(i_path)
            data.append(extract_features_subresolution(page_image))

        y = file_list['class']

        # check that we have data
        if not data:
            logger.error("Could not extract any feature vector or class")
            sys.exit(1)

        # convert to np.array
        X = np.array(data)
        # endregion

        # region Answer 3
        # save features
        if args.save_features:
            # convert X to dataframe with pd.DataFrame and save to pickle with to_pickle
            df_features = pd.DataFrame(X)
            df_features['class'] = y
            df_features.to_pickle('save_features.pickle')
        # endregion

        logger.info('Saved {} features and class to {}'.format(df_features.shape, args.save_features))

    if args.features_only:
        logger.info('No classifier to train, exit')
        sys.exit()

    if args.limit_samples:
        X = X.head(args.limit_samples)
        y = y.head(args.limit_samples)

    if args.nearest_neighbors:
        # create KNN classifier with args.nearest_neighbors as a parameter
        logger.info('Use kNN classifier with k = {}'.format(args.nearest_neighbors))

        # region Answer 4 and 5

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))
        knnForOneNeighbor = nearestNeighbors_train(X_train, y_train, k=1)
        # train with one value for k only (ici 1) mais peut-être remplacé par k=args.nearest_neighbors
        #  pour définir le nombre de cluster par les paramètres
        nearestNeighbors_test(knnForOneNeighbor, X_train, y_train, X_test, y_test)
        # endregion

        # region Answer 6
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=0)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))
        logger.info("Valid set size is {}".format(X_valid.shape))
        knnsForNeighbors = nearestNeighbors_train(X_train, y_train, limit=20)
        # train with different values from 1 to (limit can be changed!)
        nearestNeighbors_test(knnsForNeighbors, X_train, y_train, X_test, y_test, X_valid, y_valid)
        # 3 is the best KNN neighbor according to the result on the valid set!!
        # endregion

    elif args.logistic_regression:
        # region Answer 7
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))
        logger.info('Use the logisitic model to classify the MINST data')
        logreg = LogisticRegression()
        # lancer un model logistic
        logreg.fit(X_train, y_train)
        # 3 is the best KNN value found in Answer 6
        knn = knn_train(3, X_train, y_train)
        test_summary(knn, X_train, y_train, X_test, y_test, other_options=True)
        test_summary(logreg, X_train, y_train, X_test, y_test, other_options=True)
        # endregion

    elif args.learning_curve:
        # region Answer 8
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        logger.info("Initial train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))
        percents = [1, 10, 20, 40, 50, 80, 100]
        logisiticsTestAccuracies = []
        logisiticsTrainAccuracies = []
        knnsTestAccuracies = []
        knnsTrainAccuracies = []
        knnsToTest = [1, 2, 3]
        for i in knnsToTest:
            # for 3 KNNs
            knnsTestAccuracies.append([])
            knnsTrainAccuracies.append([])
        for percent in percents:
            logger.info("using {}% of the training set ".format(percent))
            Nrows = round(percent * len(X_train.index) / 100)
            current_X_train = X_train.head(Nrows)
            current_y_train = y_train.head(Nrows)
            logger.info("Train set size is {}".format(current_X_train.shape))
            # logistic
            logreg = LogisticRegression()
            logreg.fit(current_X_train, current_y_train)
            logisiticsTrainAccuracies.append(test(logreg, current_X_train, current_y_train))
            logisiticsTestAccuracies.append(test(logreg, X_test, y_test))
            counter = 0
            for i in knnsToTest:
                knn = knn_train(i, current_X_train, current_y_train)
                knnsTrainAccuracies[counter].append(test(knn, current_X_train, current_y_train))
                knnsTestAccuracies[counter].append(test(knn, X_test, y_test))
                counter += 1

        temp = logisiticsTestAccuracies + logisiticsTrainAccuracies
        plt.plot(percents, logisiticsTrainAccuracies, marker='o', linestyle='--', color='r', label="Train accuracy")
        plt.plot(percents, logisiticsTestAccuracies, marker='o', linestyle='--', color='b', label="Test accuracy")
        plt.axis([1, 100, (min(temp)) - 0.005, (max(temp) + 0.005)])
        plt.xlabel('Training set size (percent)')
        plt.ylabel('Accuracy')
        plt.title('Accuracies using logistic model with different training set size')
        plt.legend()
        plt.show()
        counter = 0
        for i in range(0, 3):
            temp = knnsTestAccuracies[i] + knnsTrainAccuracies[i]
            plt.plot(percents, knnsTrainAccuracies[i], marker='o', linestyle='--', color='r', label="Train accuracy")
            plt.plot(percents, knnsTestAccuracies[i], marker='o', linestyle='--', color='b', label="Test accuracy")
            plt.axis([1, 100, (min(temp)) - 0.005, (max(temp) + 0.005)])
            plt.xlabel('Training set size (percent)')
            plt.ylabel('Accuracy')
            plt.title('Accuracies using KNN{} model with different training set size'.format(knnsToTest[counter]))
            plt.legend()
            plt.show()
            counter += 1
        # endregion
    # region Answer 10
    elif args.testing_curve:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Initial test set size is {}".format(X_test.shape))
        percents = [1, 10, 20, 40, 50, 80, 100]
        logisiticsTestAccuracies = []
        logisiticsTestAccuracies_std = []
        knnsTestAccuracies = []
        knnsTestAccuracies_std = []
        knnsToTest = [1, 2, 3]
        for i in knnsToTest:
            # for 3 KNNs
            knnsTestAccuracies.append([])
            knnsTestAccuracies_std.append([])

        # learning
        # logistic
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        # knns
        knnsModels = []
        for i in knnsToTest:
            knnsModels.append(knn_train(i, X_train, y_train))
        # test decreasing prop
        for percent in percents:
            logger.info("using {}% of the test set ".format(percent))
            Nrows = round(percent * len(X_test.index) / 100)
            logistictemp = []
            knnTemp = []
            for i in knnsToTest:  # for 3 KNNs
                knnTemp.append([])
            for i in range(0,10):
                current_X_test = X_test.sample(frac =percent/100, replace=True)
                current_y_test = y_test.sample(frac = percent/100, replace=True)
                logger.info("Test set size is {}".format(current_X_test.shape))
                logistictemp.append(test(logreg, current_X_test, current_y_test))
                counter = 0
                for knnModel in knnsModels:
                    knnTemp[counter].append(test(knnModel, current_X_test, current_y_test))
                    counter += 1
            logisticTempNumpy = np.array(logistictemp)
            logisiticsTestAccuracies.append(logisticTempNumpy.mean())
            logisiticsTestAccuracies_std.append(logisticTempNumpy.std())
            counter = 0
            for i in knnTemp:
                knnNumpy = np.array(i)
                knnsTestAccuracies[counter].append(knnNumpy.mean())
                knnsTestAccuracies_std[counter].append(knnNumpy.std())
                counter += 1

        plt.errorbar(percents, logisiticsTestAccuracies,logisiticsTestAccuracies_std, linestyle='None', marker='^')
        plt.xlabel('Testing set size (percent)')
        plt.ylabel('Accuracy')
        plt.title('Accuracies using logistic model with different testing set size')
        plt.show()
        counter = 0
        for i in knnsToTest:
            plt.errorbar(percents, knnsTestAccuracies[counter], knnsTestAccuracies_std[counter], linestyle='None', marker='^')
            plt.xlabel('Testing set size (percent)')
            plt.ylabel('Accuracy')
            plt.title('Accuracies using knn{} model with different testing set size'.format(counter))
            plt.show()
            counter += 1
        # endregion
    else:
        logger.error('No classifier specified')
        sys.exit()
