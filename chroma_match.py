#! /usr/bin/env python
# -*-coding:utf-8-*-

import librosa
import numpy as np
import librosa.display
from skimage.feature import match_template
import os


def audio_template(path):  # 提取目录中的wav音频文件放入list中
    names = []
    for f in os.listdir(path):
        if f.endswith(".wav"):
            names.append(os.path.join(path, f))
    return names


def gener_chroma(names):  # 提取色度特征
    features = []
    for f in names:
        src, sr = librosa.load(f, 48000)  # 采样频率48000HZ
        feature = librosa.feature.chroma_stft(src)
        tmp = [f, feature]
        features.append(tmp)
    return features  # 返回文件路径以及其特征的list


def choice_file(path):
    names = []
    for f in os.listdir(path):
        if f.endswith(".wav"):
            names.append(os.path.join(path, f))
    return np.random.choice(names)  # 随机选取列表中的一个路径


def match(audio_name, feas):
    src, sr = librosa.load(audio_name, 48000)
    feature = librosa.feature.chroma_stft(src)
    res = []
    for name_fea in feas:
        try:
            tmp_res = match_template(name_fea[1], feature).max()  # 色度模版匹配
            print(tmp_res, name_fea[0])
            res.append([tmp_res, name_fea[0]])
        except:
            pass

    _res = 0
    for _tmp in res:
        if _tmp[0] > _res:
            _res = _tmp[0]
            ans = _tmp
    print(ans)  # 选择匹配得分最高的输出


database = audio_template('/Users/wangzhuoyue/PycharmProject/adDetection/test')  # 数据库文件包含所有广告的完整wav文件
feas = gener_chroma(database)
path = '/Users/wangzhuoyue/PycharmProject/adDetection/ad_dataset/baisuishan'  # 进行query的镜头音频文件
audio_name = choice_file(path)
print(audio_name)
match(audio_name, feas)
