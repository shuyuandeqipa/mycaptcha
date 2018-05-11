

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
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
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

            # 输出调试信息
            embeddings1 = emb_array[0::2]  # 6000张图片 是每一个Paris中的第一张
            embeddings2 = emb_array[1::2]  # 6000张图片 是每一个Paris中的第2张
            diff = np.subtract(embeddings1, embeddings2)  # 每一个Paris中的两个向量相减
            dist = np.sum(np.square(diff), 1)  # 向量二范数的平方，两个向量之间距离的平方。 每一个Pari之间的欧几里得距离的平方
            # #也可以说是两张图片之间特征向量的相似度
            print('------------------------------------------------------------')
            print('dist.len=', len(dist))

            # 把特征向量保存到文件中去
            f = open(r'E:\MyPythonProjects\HWDB1\train1\pairs/embeddingsOfCluster.txt', 'w')
            for embedding in emb_array:
                for data in embedding:
                    f.write(str(data) + ' ')
                f.write('\n')
            f.close()

            # 把误差保存到文件中去
            f = open(r'E:\MyPythonProjects\HWDB1\train1\pairs/distForCluster.txt', 'w')
            for d in dist:
                f.write(str(d) + '\n')
            f.close()

            # 对数据做聚类 k-means 10个类别
            from sklearn.cluster import KMeans
            kmeans=KMeans(n_clusters=10,random_state=0).fit(emb_array)
            #cluster_centers_   cluster_centers_kmeans
            f=open(r'E:\MyPythonProjects\HWDB1\train1\pairs/cluster_centers_kmeans.txt','w')
            for centers in kmeans.cluster_centers_:
                for i in centers:
                    f.write(str(i)+' ')
                f.write('\n')
            f.close()

            f=open(r'E:\MyPythonProjects\HWDB1\train1\pairs/labelsForKmeansCluster.txt','w')
            for label in kmeans.labels_:
                f.write(str(label)+'\n') # 从数据中可以看出来，是有一定的聚类的作用。不过效果不是特别好
            f.close()

            print("Done")

            # # 绘制三维图像
            #
            # import matplotlib.pyplot as plt
            # from mpl_toolkits.mplot3d import Axes3D
            # data =emb_array[:,0:3]
            #
            # x, y, z = data[:, 0], data[:, 1], data[:, 2]
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')  # 创建一个三维的绘图工程
            # #  将数据点分成三部分画，在颜色上有区分度
            # # b: blue
            # # g: green
            # # r: red
            # # c: cyan
            # # m: magenta
            # # y: yellow
            # # k: black
            # # w: white
            #
            # ax.scatter(x[0:200], y[0:200], z[0:200], c='b')  # 绘制数据点
            # ax.scatter(x[200:400], y[200:400], z[200:400], c='r')
            # ax.scatter(x[400:600], y[400:600], z[400:600], c='g')
            # ax.scatter(x[600:800], y[600:800], z[600:800], c='k')
            #
            # ax.set_zlabel('Z')  # 坐标轴
            # ax.set_ylabel('Y')
            # ax.set_xlabel('X')
            # plt.show()

            '''
>>> from sklearn.cluster import KMeans
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [4, 2], [4, 4], [4, 0]])
>>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
>>> kmeans.labels_
array([0, 0, 0, 1, 1, 1], dtype=int32)
>>> kmeans.predict([[0, 0], [4, 4]])
array([0, 1], dtype=int32)
>>> kmeans.cluster_centers_
array([[ 1.,  2.],
       [ 4.,  2.]])'''


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('lfw_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.',
                        default=r'E:\MyPythonProjects\HWDB1\train1\handledImageForTrain')
    # E:\MyPythonProjects\HWDB1\train1\handledImageForTrain
    # E:\MyPythonProjects/facenet-master\datasets\lfw_mtcnnpy_160
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='E:\\MyPythonProjects\\facenet-master\models\\facenet\\20180317-172114')

    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=90)  # 原来是160
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.',
                        default=r'E:\MyPythonProjects\HWDB1\train1\pairs/dataForCluster.txt')
    # r'E:\MyPythonProjects\facenet-master\wjx_test/wjxpairs.txt'
    # E:\MyPythonProjects\HWDB1\train1\pairs/chinesepairs.txt

    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=9)
    return parser.parse_args(argv)


if __name__ == '__main__':

    main(parse_arguments([r'E:\MyPythonProjects\HWDB1\train1\handledImageForTrain/',
                          'E:\\MyPythonProjects\\facenet-master\models\\facenet\\20180317-172114']))


# 需要的改进工作：
# 1.使用pca之后再试试：效果是好了一些，在图上显示的效果区分的更好了(对多个类的区分效果更好)
# 2.但是，类间距要是更大就好了
# 3.可以在学到的特征上面使用其他的聚类算法试试效果
# 4.把学到的特征进行降维，然后再使用几种聚类算法试试效果
# 5.有了特征了，其实也可以算是知道了类别了，也就是有监督的学习，那么，就可以使用大量的监督学习算法来进行
#   分类处理。svm可以试试效果。这样做，效果可以提升吗？(比直接使用这些特征来进行分类)如果可以提升，原因？

