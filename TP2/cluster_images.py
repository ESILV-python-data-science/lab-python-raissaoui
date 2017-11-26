# -*- coding: utf-8 -*-
"""
Cluster images based on visual similarity

C. Kermorvant - 2017
"""


import argparse
import glob
import logging
import os
import shutil
import time
import sys
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from tqdm import tqdm
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
import numpy as np



# default sub-resolution
IMG_FEATURE_SIZE = (12, 16)

# Setup logging
logger = logging.getLogger('cluster_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def extract_features(img):
    """
    Compute the subresolution of an image and return it as a feature vector

    :param img: the original image (can be color or gray)
    :type img: pillow image
    :return: pixel values of the image in subresolution
    :rtype: list of int in [0,255]

    """

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
        IMG_FEATURE_SIZE, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]


def copy_to_dir(images, clusters, cluster_dir):
    """
    Move images to a directory according to their cluster name

    :param images: list of image names (path)
    :type images: list of path
    :param clusters: list of cluster values (int), such as given by cluster.labels_, associated to each image
    :type clusters: list
    :param cluster_dir: prefix path where to copy the images is a drectory corresponding to each cluster
    :type images: path
    :return: None
    """

    for img_path, cluster in zip(images, clusters):
        # define the cluster path : for example "CLUSTERS/4" if the image is in cluster 4
        clst_path = os.path.join(cluster_dir, str(cluster))
        # create the directory if it does not exists
        if not os.path.exists(clst_path):
            os.mkdir(clst_path)
        # copy the image into the cluster directory
        shutil.copy(img_path, clst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, cluster images and move them to a directory')
    parser.add_argument('--images-dir',required=True)
    parser.add_argument('--move-images')
    args = parser.parse_args()


    if args.move_images:
        CLUSTER_DIR = args.move_images
        # Clean up
        if os.path.exists(CLUSTER_DIR):
            shutil.rmtree(CLUSTER_DIR)
            logger.info('remove cluster directory %s' % CLUSTER_DIR)
        os.mkdir(CLUSTER_DIR)

    # find all the pages in the directory
    images_path_list = []
    data = []
    if args.images_dir:
        SOURCE_IMG_DIR = args.images_dir
        images_path_list = glob.glob("{0}/*.jpg".format(SOURCE_IMG_DIR))
        logger.info('images_path_list intliazed, we have {0} jpg images in {1} '.format(len(images_path_list),SOURCE_IMG_DIR))
    if not images_path_list:
        logger.warning("Did not found any jpg image in %s"%args.images_dir)
        sys.exit(0)
    logger.info('handling the feature vector')
    pbar = tqdm(total=len(images_path_list))
    for img in images_path_list:
        pbar.update(1)
        data.append(extract_features(Image.open(img)))
    pbar.close()
    logger.info('Feature vector intalized')

    # cluster the feature vectors
    if not data:
        logger.error("Could not extract any feature vector")
        sys.exit(1)

    # convert to np array (default format for scikit-learn)
    X = np.array(data)
    logger.info("Running clustering")
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    logger.debug('centroides')
    logger.debug(centroids)
    logger.debug(len(centroids))
    for centre in centroids:
        logger.debug('each centroide')
        logger.debug(len(centre))
    logger.info('labels:')
    logger.debug(labels)
    logger.debug(len(labels))

    colors = ["g.", "r.", "c."]

    for i in range(len(X)):
        logger.debug("coordinate:", X[i], "label:", labels[i])
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)

    plt.show()
    copy_to_dir(images_path_list, labels,CLUSTER_DIR)
    logger.info("The end, check the cluster images in {0}".format(CLUSTER_DIR))
