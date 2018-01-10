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

df = pd.read_csv("LeMonde2003.csv", sep='\t',engine='python')
df = df.dropna(how="all")
categories = ['ENT', 'INT', 'ART', 'SOC', 'FRA', 'SPO', 'LIV', 'TEL', 'UNE']
data = df[df.category.isin(categories)]
# print(data)
plt.figure(figsize=(12, 8))
fig = sns.countplot(x="category", data=df)
matplotlib.pyplot.savefig(fig)