#! /usr/bin/env python
# -*- coding:utf-8 -*-

import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

"""
输入广告／非广告特征向量的pickle文件，进行特征融合
将融合的特征使用SVM进行训练并保存
"""

f = open("features-audio-not.pkl", "rb")  # 读取特征向量的pickle文件
audio = pickle.load(f)
f.close()
f = open("features-audio-ad.pkl", "rb")
audio_ad = pickle.load(f)
f.close()
f = open("features-image-not.pkl", "rb")
image = pickle.load(f)
f.close()
f = open("features-image-ad.pkl", "rb")
image_ad = pickle.load(f)
f.close()
image.extend(image_ad)
audio.extend(audio_ad)
labels = [0.0]*(len(image)-len(image_ad))  # 根据广告非广告的数据长度分好标签
labels.extend([1.0]*len(image_ad))
image_mat = []
audio_mat = []
for item in image:
    if len(item) != 4:
        print("Error")
    image_mat.extend(item)
for item in audio:
    audio_mat.extend(item)
pca = PCA(n_components=84)
image_84 = pca.fit_transform(image_mat)  # 进行特征降维度


def mat_to_vec(features):  # 将特征矩阵合并成一个向量
    features_vec = []
    for i in range(len(labels)):
        # concatenate 4 feature vectors as one
        # features_vec.append(np.concatenate(features_300[4*i:4*i+4]))

        # mean 4 feature vectors as one
        features_vec.append(np.mean(features[4 * i:4 * i + 4], axis=0))

        # sum 4 feature vectors as one
        # features_vec.append(np.sum(features_300[4*i:4*i+4],axis=0))
    return features_vec


image_vec = mat_to_vec(image_84)
image_vec = np.array(image_vec)
audio_84 = pca.fit_transform(audio_mat)
feature = np.hstack([image_vec, audio_84])  # 直接拼接图片和音频的特征向量
norm = preprocessing.normalize(feature)  # 进行归一化
X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.25)
clf = SVC(C=2.9, gamma=0.007, cache_size=1000)
clf.fit(X_train, y_train)  # 分割数据集／测试集并训练
acc = clf.score(X_test, y_test)
print(acc)  # 得出准确率
joblib.dump(clf, 'SVM_fusion.pkl')  # 保存svm模型为job文件