#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
import cv2

"""
进行镜头级别的logo匹配，输入广告路径
输出匹配出的广告及其得分
"""


def logo_names(path):
    names = []
    for f in os.listdir(path):
        names.append(os.path.join(path, f))  # 列出所有logo的路径
    return sorted(names)


def tar_images(path):
    names = []
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):  # 如果路径是文件夹
            dir_1 = os.path.join(path, f)  # 将文件夹路径赋值
            for e in os.listdir(dir_1):
                if os.path.isdir(os.path.join(dir_1, e)):
                    dir_2 = os.path.join(dir_1, e)  # 在商标文件夹里添加镜头路径
                    names.extend([os.path.join(dir_2, im) for im in np.random.choice(os.listdir(dir_2), 1)])
                    # 随机在每个镜头中选择一张图片
    return names


def match_sift(im1, im2):

    img1 = cv2.imread(im1, 0)  # queryImage
    img2 = cv2.imread(im2, 0)  # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])
        # print len(good),len(kp1)
    return len(good)*1.0/len(kp1)


im_names = tar_images("/Users/wangzhuoyue/PycharmProject/adDetection/test")  # 所有商标所有镜头的一帧
print(im_names)
logos = logo_names("/Users/wangzhuoyue/PycharmProject/adDetection/ad_dataset/log")  # 所有广告的商标图片
print(logos)

__num = 0
ans = []
for shots in im_names:
    for logo in logos:
        print(logo)
        num = match_sift(shots, logo)
        if num > __num:
            __num = num
            ans = [shots, logo]  # 输出当前得分最高的帧与其匹配的logo
    print(__num)
    print(ans)
