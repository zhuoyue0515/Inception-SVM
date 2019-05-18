#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from sklearn.externals import joblib

"""
对未打标签的视频文件进行镜头级别的分类
输入视频的路径
输出广告／非广告分类结果
"""


def get_file_list(path):
    """
    get a list of completed sorted path.
    Input : The path include file.
    Output : a sorted list of filename with completed path.
    """
    names = []
    for filename in os.listdir(path):
        if filename != "._.DS_Store":  # 防止mac系统的系统文件干扰路径（Ubuntu可忽略）
            name = os.path.join(path, filename)
            names.append(name)
    return sorted(names)


def shot_files(path="/Users/wangzhuoyue/PycharmProject/adDetection/test"):
    """
    get a list of completed sorted path.
    Input : The path include file.
    Output : a sorted list of filename with completed path.
    """
    image_path = []
    for filename in os.listdir(path):  # video dir
        if filename != ".DS_Store":
            if os.path.isdir(os.path.join(path, filename)):
                path1 = os.path.join(path, filename)
            for filename1 in os.listdir(path1):
                if os.path.isdir(os.path.join(path1, filename1)):
                    image_path.append(os.path.join(path1, filename1))  # image shots dir
            image_path = sorted(image_path)
    return image_path


def sample(files_list, n=4):  # extract n frames for each shot
    files_num = len(files_list)
    samples = []
    for i in range(n-1):
        samples.append(files_list[(i*files_num)/(n-1)])
    samples.append(files_list[-1])
    return samples


def create_graph(model_dir='/Users/wangzhuoyue/PycharmProject/adDetection/model/inception-2015-12-05'):
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        for ind, image in enumerate(list_images):
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)
    return features  # return feature with n*2048 dimensional


def get_shot_feature(pca_model, matrix):  # dimension reduction and merge by means
    res = pca_model.transform(matrix)
    vec = np.mean(res, axis=0)
    return vec  # return feature with 1*84 dimension


def classify(svm_model, vector):
    res = svm_model.predict(vector)
    return res


def main():
    create_graph()
    images_dir = shot_files()  # all shots including frames in a video
    for shot in images_dir:
        frames = get_file_list(shot)
        shot_ims = sample(frames)
        feas = extract_features(shot_ims)
        pca_res = (get_shot_feature(pca, feas)).reshape(1, -1)
        res = classify(svm, pca_res)  # classify
        print(res)


pca = joblib.load("/Users/wangzhuoyue/PycharmProject/adDetection/model/pca.job")  # read pre-trained pca and svm model
svm = joblib.load("/Users/wangzhuoyue/PycharmProject/adDetection/model/svm.job")

if __name__ == "__main__":
    main()
