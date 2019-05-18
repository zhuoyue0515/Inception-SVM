#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import pickle
import librosa
import librosa.display
from PIL import Image

"""
输入完整镜头及其音频文件路径，使用Inception模型提取特征
输出广告／非广告的图像以及音频的特征pickle文件
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


def shot_files(path):
    """
    get a list of completed sorted path.
    Input : The path include file.
    Output : a sorted list of filename with completed path.
    """
    image_path = []
    for filename in os.listdir(path):  # video dir
        if os.path.isdir(os.path.join(path, filename)):
            path1 = os.path.join(path, filename)
        for filename1 in os.listdir(path1):
            if os.path.isdir(os.path.join(path1, filename1)):
                image_path.append(os.path.join(path1, filename1))  # image shots dir
        image_path = sorted(image_path)
    return image_path


def create_graph():
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images):
    """
    输入图片路径的list
    输出2048维的特征向量
    """
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        for ind, image in enumerate(list_images):
            if ind % 1000 == 0:
                print('Processing %s...' % image)
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)
    return features


def sample(files_list, n=4):  # extract n frames for each shot
    files_num = len(files_list)
    samples = []
    for i in range(n-1):
        samples.append(files_list[(i*files_num)//(n-1)])
    samples.append(files_list[-1])
    return samples


def gener_chroma(names):
    """
    输入音频文件的list
    输出音频文件的色谱图list
    """
    features = []
    for f in names:
        src, sr = librosa.load(f, 48000)
        feature = librosa.feature.chroma_stft(src)
        tmp = [f, feature]
        features.append(tmp)
    return features


def matrix2image(data):  # 将numpy矩阵转换为图片
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


# file direction
model_dir = "/home/lab307/wzy/adDetection/inception-2015-12-05"
images_dir = '/media/lab307/aa9d6aa6-56d4-49e0-80cd-908035f93743/mengyueshuju/Images/cctv120170303'
audio_dir = '/media/lab307/aa9d6aa6-56d4-49e0-80cd-908035f93743/mengyueshuju/shot_wav'
create_graph()

# pre-processing
features_image_ad = []
features_image_not = []
shots = shot_files(images_dir)
for i in range(len(shots)-1, -1, -1):  # 倒序删除列表中的元素
    if shots[i].endswith("not"):
        index = np.random.binomial(1, 0.13)  # 按照比例对非广告镜头进行抽样
        if index == 0:
            shots.pop(i)

# extract audio features
audio = []
audio_list = [0]
features_audio_ad = []
features_audio_not = []
for i in shots:
    audio_shots = audio_dir + i[84:] + '.wav'  # 取得抽样后相应镜头的音频文件
    audio.append(audio_shots)
audio_chroma = gener_chroma(audio)
for i in range(len(audio)):
    new_im = matrix2image(audio_chroma[i][1])
    new_im.save('template.jpg')  # 将色谱图存储便于抽取特征向量
    audio_list[0] = "/home/lab307/wzy/adDetection/template.jpg"
    audio_fes = extract_features(audio_list)
    filename = audio_chroma[i][0].replace(".wav", "")
    if filename.endswith("not"):  # 根据标签判断广告非广告
        features_audio_not.append(audio_fes)
    if filename.endswith("ad"):
        features_audio_ad.append(audio_fes)
pickle.dump(features_audio_ad, open('features-audio' + '-ad.pkl', 'wb'))  # 存储为pickle文件
pickle.dump(features_audio_not, open('features-audio' + '-not.pkl', 'wb'))

# extract image features
for frames in shots:
    images = get_file_list(frames)
    images_sampled = sample(images)
    features_im = extract_features(images_sampled)  # 提取出2048*4维度的特征向量
    features_im = list(features_im)
    if frames.endswith("not"):
        features_image_not.append(features_im)
    if frames.endswith("ad"):
        features_image_ad.append(features_im)
pickle.dump(features_image_ad, open('features-image' + '-ad.pkl', 'wb'))
pickle.dump(features_image_not, open('features-image' + '-not.pkl', 'wb'))
