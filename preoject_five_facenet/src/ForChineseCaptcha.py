"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
from src import facenet, lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate


def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Read the file containing the pairs used for testing
            # 1. 读入之前的pairs.txt文件
            # 读入后如[['Abel_Pacheco','1','4']]
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))  # 剪裁好了的图片

            # Get the paths for the corresponding images
            # 获取文件路径和是否匹配关系对
            all_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
            paths=all_paths[0::2]
            print("paths shape =", len(paths))  # 12000
            print("paths=", paths[0:200])
            print('actual_issame shape=', len(actual_issame))  # 6000
            print('actual_issame', actual_issame[0:200])

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]  # 128

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)  # 12000
            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))  # 12000* 128
            for i in range(nrof_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            # # 输出调试信息
            # embeddings1 = emb_array[0::2]  # 6000张图片 是每一个Paris中的第一张
            # embeddings2 = emb_array[1::2]  # 6000张图片 是每一个Paris中的第2张
            # diff = np.subtract(embeddings1, embeddings2)  # 每一个Paris中的两个向量相减
            # dist = np.sum(np.square(diff), 1)  # 向量二范数的平方，两个向量之间距离的平方。 每一个Pari之间的欧几里得距离的平方
            # # #也可以说是两张图片之间特征向量的相似度
            # print('------------------------------------------------------------')
            # print('dist.len=', len(dist))
            need_embeddings=emb_array
            # 使用embedding的信息计算相似度。
            f=open(r'E:\MyPythonProjects\captchaOneChineseTrain/result.txt','w')
            for number_of_tests in range(10000):
                f.write(str(number_of_tests).zfill(4)+',')
                # 每个里面有13张图片 ：9 10 11 12
                for i in range(9,13):
                    emb_i=need_embeddings[(number_of_tests*13)+i]
                    min_index=0
                    min_dist=999999
                    for j in range(9):
                        emb_j=need_embeddings[(number_of_tests*13)+j]
                        # 这里计算相似度使用的是欧式距离
                        diff = np.subtract(emb_i, emb_j)
                        dist = np.sum(np.square(diff))
                        # 使用余弦相似度
                        # dist=np.sum(emb_i*emb_j)/(np.linalg.norm(emb_i)*np.linalg.norm(emb_j))
                        if dist<min_dist:
                            min_dist=dist
                            min_index=j
                    f.write(str(min_index))
                f.write('\n')
            f.close()





def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('lfw_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.',
                        default=r'E:\MyPythonProjects\resizedImage')
    # E:\MyPythonProjects\HWDB1\train1\handledImageForTrain
    # E:\MyPythonProjects/facenet-master\datasets\lfw_mtcnnpy_160
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='E:\\MyPythonProjects\\facenet-master\models\\facenet\\20180317-105208')

    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=90)  # 原来是160
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.',
                        default=r'E:\MyPythonProjects\captchaOneChineseTrain/chineseAll.txt')
    # r'E:\MyPythonProjects\facenet-master\wjx_test/wjxpairs.txt'
    # E:\MyPythonProjects\HWDB1\train1\pairs/chinesepairs.txt

    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    # 需要预测的图片数据的路径  模型参数的路径
    main(parse_arguments([r'E:\MyPythonProjects\resizedImage/',
                          'E:\\MyPythonProjects\\facenet-master\models\\facenet\\20180327-215050']))

