import matplotlib.pyplot as plt
import numpy as np
from skimage import io,transform
from skimage.morphology import disk
from skimage import data,filters
import os
from PIL import Image

from project_data_one.utils import convert2gray,text2vec,handle_noise
from project_data_one.cfg import IMAGE_HEIGHT,IMAGE_WIDTH,MAX_CAPTCHA,CHAR_SET_LEN
# 读取图片数据，转换成需要的数据格式
image_path='E:\MyPythonProjects\mycaptcha\project_data_one/test/'
labels_path='E:\MyPythonProjects\mycaptcha\captdata\data-1/train/mappings.txt'

def add_zeros(index):
    index_str = str(index) + ""
    return index_str.zfill(4)+'.jpg'

batch_size=64
# global data_now_start,data_now_end
# data_now_start=0
# data_now_end=data_now_start+batch_size
all_dataset_size=10000

# def get_next_batch():
#     batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
#     batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
#     global data_now_start, data_now_end
#
#     for index in range(data_now_start, data_now_end):
#          index_str = str(index) + ""
#          read_path = image_path + add_zeros(index_str)
#          img = Image.open(read_path)
#          img = np.array(img)
#          gray_img = convert2gray(img)
#          batch_x[index % (batch_size), :] = gray_img.flatten() / 255
#         # 需要从文件中读取需要的labels数据
#
#          # batch_y[index % (batch_size), :] =
#
#     # 下一批数据开始
#          if data_now_end < dataset_size - batch_size:
#              data_now_start += batch_size
#              data_now_end += batch_size
#          else:
#             data_now_start = data_now_end
#             data_now_end = dataset_size - 1

# 这里读取一批数据的方式并不是很好，应该从10000中图片中随机选取64张

# from skimage import data,filters
# import matplotlib.pyplot as plt
# from skimage.morphology import disk
# img = data.camera()
# edges1 = filters.median(img,disk(5))
# edges2= filters.median(img,disk(9))
# plt.figure('median',figsize=(8,8))
# plt.subplot(121)
# plt.imshow(edges1,plt.cm.gray)
# plt.subplot(122)
# plt.imshow(edges2,plt.cm.gray)
# plt.show()


def get_next_batch_image_and_name(batch_size=batch_size,dataset_size=9000,flag=0):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    # all_image=os.listdir(image_path)
    if flag==0:#获取训练数据
        random_file = np.random.randint(0, dataset_size, batch_size)
    else:#测试数据
        random_file = np.random.randint(dataset_size, all_dataset_size, batch_size)
        # print('random_file=',random_file)

    all_labels_data = np.genfromtxt(labels_path,dtype='str')

    for i in range(len(random_file)):
        index=random_file[i]
        read_path=image_path+add_zeros(index)#图片的文件路径
        # 读取图片
        image= io.imread(read_path, as_grey=False)

        # plt.figure()
        # plt.imshow(image)
        # plt.show()

        # print('image.shape=',image.shape)
        image=convert2gray(image)
        # image=handle_noise(image)

        # show a image for test
        # plt.figure()
        # plt.imshow(image)
        # plt.show()
        # print('image.shape=',image.shape)

        batch_x[i,:]=image.flatten()/255

        #读取图片的name
        str=all_labels_data[index]
        # print(str)
        a=str.index(',')
        b=str.index('=')
        # print('label=',str[a+1:])
        batch_y[i,:]=text2vec(str[a+1:b])

    return batch_x,batch_y
#
# batch_x,batch_y=get_next_batch_image_and_name(1,flag=1)
# print(batch_x.shape)
# print(batch_y)

# 需要把图片进行预处理，把所有图片保存好，训练和测试使用

def get_one_predict_image_data(index_start=9000,index_end=all_dataset_size):
    random_index = np.random.randint(index_start, index_end)
    read_path = image_path + add_zeros(random_index)  # 图片的文件路径
    image = io.imread(read_path, as_grey=False)
    return image,random_index

# print(get_one_predict_image_data().shape)

def get_one_predict_image_data_with_index(image_index=9000):
    read_path = image_path + add_zeros(image_index)  # 图片的文件路径
    image = io.imread(read_path, as_grey=False)
    return image
# print('get_one_predict_image_data_with_index',get_one_predict_image_data_with_index().shape)

