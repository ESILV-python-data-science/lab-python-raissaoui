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
from sklearn.metrics import accuracy_score

import numpy as np

# sauvergader les features (première étape)
# python classify_images.py --images-list MNIST_all.csv --features-only
#  --save-features "C:\Users\Ramez Aissaoui\lab-python-raissaoui\TP_classification"

# Setup logging
logger = logging.getLogger('classify_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def extract_features_subresolution(img,img_feature_size = (8, 8)):
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

def distanceTES(observation,columns,distanceTo) :
    between_value  = 0
    for i in columns :
        between_value += (observation[i] - distanceTo)**2
    return math.sqrt(between_value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, train a classifier on images and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images-list',help='file containing the image path and image class, one per line, comma separated')
    input_group.add_argument('--load-features',help='read features and class from pickle file')
    parser.add_argument('--save-features',help='save features in pickle format')
    parser.add_argument('--limit-samples',type=int, help='limit the number of samples to consider for training')
    classifier_group = parser.add_mutually_exclusive_group(required=True)
    classifier_group.add_argument('--nearest-neighbors',type=int)
    classifier_group.add_argument('--features-only', action='store_true', help='only extract features, do not train classifiers')
    args = parser.parse_args()


    if args.load_features:
        df_features = pd.read_pickle(args.load_features)
        file_list = df_features['class']
        X = df_features.drop('class', axis=1)
        print(X)

    else:
        # Answer 1
        # file = "MNIST_all.csv"
        # Load the image list from CSV file using pd.read_csv
        # df_features = pd.read_csv(file, na_values=['.'], error_bad_lines=False,sep=";")
        # see the doc for the option since there is no header ;
        # specify the column names :  filename , class

        file_list = pd.read_csv(args.images_list, header=None, names = ['filename','class'], engine='python')

        # logger.info('Loaded {} images in {}'.format(all_df.shape,args.images_list))


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





        # save features
        if args.save_features:
            print(y)
            print(X)
            # Answer 3
            # convert X to dataframe with pd.DataFrame and save to pickle with to_pickle
            df_features = pd.DataFrame(X)
            df_features['class'] = y
            df_features.to_pickle('save_features.pickle')


        logger.info('Saved {} features and class to {}'.format(df_features.shape, args.save_features))


    if args.features_only:
        logger.info('No classifier to train, exit')
        sys.exit()

        # Train classifier
        logger.info("Training Classifier")

        if args.nearest_neighbors:
            # create KNN classifier with args.nearest_neighbors as a parameter
            logger.info('Use kNN classifier with k= {}'.format(args.nearest_neighbors))
        else:
            logger.error('No classifier specified')
            sys.exit()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # Use train_test_split to create train/test split
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))

        # Do Training@
        t0 = time.time()
        # exampleImage = X_train.iloc[0]
        # print(exampleImage)
        # print(distance.euclidean(X_train.iloc[0],X_train.iloc[0]))
        # distances = []
        # for idx,row in X_train.iterrows():
        #    distances.append(distance.euclidean(row,exampleImage))
        # print(len(distances))
        # df_distance = pd.DataFrame(data={"dist":distances,"idx":distances.index})
        # print(len(df_distance.index))
        # print(distances)

        knn = KNeighborsClassifier(n_neighbors=args.nearest_neighbors)
        knn.fit(X_train, y_train)
        predictions_train = knn.predict(X_train)
        predictions_test = knn.predict(X_test)

        logger.info(
            "accuracy got got the X_test is equals to {0}".format(metrics.accuracy_score(y_test, predictions_test)))
        logger.info(
            "accuracy got got the X_train is equals to {0}".format(metrics.accuracy_score(y_train, predictions_train)))
        logger.info("Training  done in %0.3fs" % (time.time() - t0))

        # Do testing
        logger.info("Testing Classifier")
        t0 = time.time()
        #          rouge   predicted = clf.predict(X_test)

        # Print score produced by metrics.classification_report and metrics.accuracy_score
        logger.info("Testing  done in %0.3fs" % (time.time() - t0))
