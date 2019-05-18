#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import os
import shutil
import numpy as np
from scipy.io import wavfile

"""
输入包含完整路径的视频文件名
输出镜头图片与镜头音频
"""


def video_to_frames(filename):
    """
    use ffmpeg to translate video to frames
    input: completed path of video
    output: none
    """
    folder = filename.replace(".mp4", "")  # 去掉文件后缀名

    try:
        os.mkdir(folder)  # 创建文件夹
    except:
        os.system("rm -r %s" % folder)
        os.mkdir(folder)
    command = "ffmpeg -i %s -r 2 -f image2 %s//%%4d.jpg" % (filename, folder)
    os.system(command)  # 是用ffmpeg将视频分帧


def get_file_list(path):
    """
    get a list of completed sorted path.
    Input : The path include file.
    Output : a sorted list of filename with completed path.
    """

    names = []
    for _file in os.listdir(path):  # 提取目录下每个帧文件名
        name = os.path.join(path, _file)  # 合并成路径
        names.append(name)
        names = sorted(names)	
    return names


def get_diff_list(diff):  # 曼哈顿距离差分
    """calculate the difference of list """

    diff2 = []
    for i in range(len(diff)-1):
        diff2.append(abs(diff[i] - diff[i+1]))

    return diff2


def get_rgb_diff(frames):
    """
    calculate image's difference.
    Input : a list of frame with completed path.
    Output : a list of image's difference.	
    """

    im2 = cv2.imread(frames[0])  # 读取第一帧的图片
    h = im2.shape[0]  # 图像的长度
    w = im2.shape[1]  # 图像的宽度
    num_pixel = h * w   # 像素个数
    rgb_diff = []
    for i in range(len(frames)-1):
        im1 = im2
        im2 = cv2.imread(frames[i+1])  # 读取下一帧

        diff_r = abs(np.sum(im1[:, :, 0])/num_pixel-np.sum(im2[:, :, 0])/num_pixel)  # rgb三通道的曼哈顿距离
        diff_g = abs(np.sum(im1[:, :, 1])/num_pixel-np.sum(im2[:, :, 1])/num_pixel)
        diff_b = abs(np.sum(im1[:, :, 2])/num_pixel-np.sum(im2[:, :, 2])/num_pixel)

        diff = diff_r+diff_g+diff_b  # OpenCV对img的存储方式其实是BGR
        rgb_diff.append(diff)

    return rgb_diff


def get_shots(rgb_diff2):  # 通过曼哈顿距离差分切分镜头，返回一个list包含n组镜头的第一帧，中间帧，和结束帧
    """
    from RGB_diff list get scenes
    each scene include the first middle last frame.	
    """

    shots = []
    frame = 0
    for i in range(len(rgb_diff2)-1):
        if rgb_diff2[i] > 16 and rgb_diff2[i+1] > 15 and abs(i-frame) > 5:
            temp = [frame, (frame+i+1)/2, i+1]
            shots.append(temp)
            frame = i+2

    i = len(rgb_diff2)
    temp = [frame, (frame+i+1)/2, i+1]
    shots.append(temp)

    return shots


def frames_to_shots(path):
    """
    input: path include images
    """

    images_list = get_file_list(path)  # 得到全部帧的名称
    rgb_diff = get_rgb_diff(images_list)  # 求曼哈顿距离
    rgb_diff2 = get_diff_list(rgb_diff)  # 曼哈顿距离差分
    shots = get_shots(rgb_diff2)  # 通过阈值切分镜头
    i = 0
    for scene in shots:
        i = i+1
        last = "%03d" % i  # 镜头标识符001-999
        scene_folder = path+"/"+"shot-"+str(last)
        os.mkdir(scene_folder)  # 创建该镜头的文件夹
        for j in range(scene[0], scene[-1]+1):
            shutil.move(images_list[j], scene_folder)  # 镜头的所有帧移入该文件夹


def extract_audio(filename):
    des_f = filename.replace("ts", "wav")
    command = "ffmpeg -i {source_file} -f wav -vn {dest_file}".format(source_file=filename, dest_file=des_f)
    os.system(command)


def shot_length(shot_names):
    sec_len = []
    for s in shot_names:
        for a, b, c in os.walk(s):  # 遍历镜头文件夹
            sec_len.append([a, len(c)])  # 输出镜头文件夹的地址和镜头帧的数量
    return sorted(sec_len)


def write_scene(filename, sec_len):
    """separate wav to scene"""

    filename = filename.replace("ts", "wav")
    freq, aud = wavfile.read(filename)  # 得到音频的频率以及时间序列
    aud_len = len(aud)
    img_num = 0
    for item in sec_len:  # 得到总的帧数
        img_num += item[1]
    start = 0
    for item in sec_len:
        des_name = item[0]+".wav"
        sec_length = int(aud_len*item[1]/img_num)  # 按镜头帧的比列截取音频文件得到镜头帧音频
        end = start + sec_length
        wavfile.write(des_name, freq, aud[start:end])
        start = end


if __name__ == "__main__":  # 主函数
    names = ["ET.mp4"]
    for filename in names:
        try:
            filename = os.path.join("/Users/zhuoyuewang/Desktop/其他/adDetection/test", filename)  # 文件名
            video_to_frames(filename)  # 视频分帧
            # path = filename.replace(".ts", "")
            # frames_to_shots(path)  # 帧分镜头
            # extract_audio(filename)  # 抽取音频文件
            # shots = get_file_list(path)  # shot001-shot999 as str
            # wav_len = shot_length(shots)  # 得到镜头中的帧数
            # write_scene(filename, wav_len)  # 切分音频文件得到每个镜头的音频文件
        except:
            pass
