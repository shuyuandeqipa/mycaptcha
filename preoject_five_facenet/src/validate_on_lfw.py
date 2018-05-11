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
from src import facenet,lfw
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
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs)) # 剪裁好了的图片

            # Get the paths for the corresponding images
            # 获取文件路径和是否匹配关系对
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
            print("paths shape =",len(paths)) # 12000
            print("paths=",paths[0:200])
            print('actual_issame shape=',len(actual_issame)) # 6000
            print('actual_issame',actual_issame[0:200])

            # Load the model
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            #image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]# 128
        
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths) # 12000
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))  # 12000* 128
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            # 输出调试信息
            embeddings1 = emb_array[0::2]  # 6000张图片 是每一个Paris中的第一张
            embeddings2 = emb_array[1::2]  # 6000张图片 是每一个Paris中的第2张
            diff = np.subtract(embeddings1, embeddings2)  # 每一个Paris中的两个向量相减
            dist = np.sum(np.square(diff), 1)  # 向量二范数的平方，两个向量之间距离的平方。 每一个Pari之间的欧几里得距离的平方
            # #也可以说是两张图片之间特征向量的相似度
            print('------------------------------------------------------------')
            print('dist.len=', len(dist))

            # 把特征向量保存到文件中去 ,由于这里处理的数据不是很靠谱，所以，这里输出的特征无法直接用于聚类处理
            f=open(r'E:\MyPythonProjects\HWDB1\train1\pairs/embeddingsOfChinesepairs.txt','w')
            for embedding in emb_array:
                for data in embedding:
                    f.write(str(data)+' ')
                f.write('\n')
            f.close()


            # 把误差保存到文件中去
            f=open(r'E:\MyPythonProjects\HWDB1\train1\pairs/distInChinesePairs.txt','w')
            for d in dist:
                f.write(str(d)+'\n')
            f.close()
            # ***************************************************************************

            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
                actual_issame, nrof_folds=args.lfw_nrof_folds)

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.',
                        default=r'E:\MyPythonProjects\captchaOneChineseTrain\dataset') # 数据集所在目录，需要修改
    # E:\MyPythonProjects\HWDB1\train1\handledImageForTrain
    # E:\MyPythonProjects/facenet-master\datasets\lfw_mtcnnpy_160
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='E:\\MyPythonProjects\\facenet-master\models\\facenet\\20180330-090756')# 需要修改

    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=90) # 原来是160 需要修改
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.',
                        default= r'E:\MyPythonProjects\captchaOneChineseTrain/pairs.txt') # 需要修改
    # r'E:\MyPythonProjects\facenet-master\wjx_test/wjxpairs.txt'
    # E:\MyPythonProjects\HWDB1\train1\pairs/chinesepairs.txt


    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    # main(parse_arguments(['E:\MyPythonProjects/facenet-master\datasets\lfw_mtcnnpy_160',
    #                       'E:\\MyPythonProjects\\facenet-master\models\\facenet\\20170512-110547']))

    # main(parse_arguments(['E:\MyPythonProjects/facenet-master\datasets\lfw_mtcnnpy_160',
    #                       'E:\\MyPythonProjects\\facenet-master\models\\facenet\\20180316-075520']))

    main(parse_arguments([r'E:\MyPythonProjects\captchaOneChineseTrain\dataset/',
                          'E:\\MyPythonProjects\\facenet-master\models\\facenet\\20180330-090756']))

# 这将
# a）加载模型，
# b）用图像对加载和解析文本文件，
# c）计算测试集中所有图像的嵌入，
# d）计算精度，验证率（@ FAR = -10e- 3），曲线下面积（AUC）和等误差率（EER）性能指标。
#
# Runnning forward pass on LFW images
# Accuracy: 0.992+-0.003
# Validation rate: 0.97467+-0.01477 @ FAR=0.00133
# Area Under Curve (AUC): 1.000
# Equal Error Rate (EER): 0.007

'''
# 使用自己写的代码生成的pairs文件，由于没有使用全部的训练样本，会让正确率有所偏高
wjxpairs.txt 8 ford ; 200 a ford
Runnning forward pass on LFW images
Accuracy: 0.994+-0.003
Validation rate: 0.98438+-0.00768 @ FAR=0.00063
Area Under Curve (AUC): 1.000
Equal Error Rate (EER): 0.006
'''