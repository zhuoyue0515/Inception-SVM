#! /usr/bin/env python
# -*- coding:utf-8 -*-

import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def load_features():
    f = open("eight_frame_feature_not.pkl", "rb")
    features = pickle.load(f)
    f.close()
    print(len(features))
    f = open("eight_frame_feature_ad.pkl", "rb")
    ads = pickle.load(f)
    print(len(ads))
    features.extend(ads)
    f.close()

    return features


features = load_features()
labels = [0.0]*3587
labels.extend([1.0]*3604)
features_mat = []

for item in features:
    if len(item) != 8:
        print("Error")
    features_mat.extend(item)

pca = PCA(n_components=84)
features_84 = pca.fit_transform(features_mat)


def mat_to_vec(features):
    features_vec = []
    for i in range(7191):
        # concatenate 4 feature vectors as one
        # features_vec.append(np.concatenate(features_300[4*i:4*i+4]))

        # mean 4 feature vectors as one
        features_vec.append(np.mean(features[8 * i:8 * i + 8], axis=0))

        # sum 4 feature vectors as one
        # features_vec.append(np.sum(features_300[4*i:4*i+4],axis=0))
    return features_vec


features_vec = mat_to_vec(features_84)
norm = preprocessing.normalize(features_vec)
X_train, X_test, y_train, y_test = train_test_split(features_vec, labels, test_size=0.25)
clf = SVC(C=3.5, gamma=0.02, cache_size=1000)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(acc)
