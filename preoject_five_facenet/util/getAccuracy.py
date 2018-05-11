# # function: 更改图片尺寸大小
# import os
# import os.path
# from PIL import Image
#
# '''
# filein: 输入图片
# fileout: 输出图片
# width: 输出图片宽度
# height:输出图片高度
# type:输出图片类型（png, gif, jpeg...）
# '''
#
#
# def ResizeImage(filein, fileout, width, height, type):
#     img = Image.open(filein)
#     out = img.resize((width, height), Image.ANTIALIAS)
#     out.save(fileout, type)
#     # if __name__ == "__main__":
#     #   filein = r'image\test.png'
#     #   fileout = r'image\testout.png'
#     #   width = 60
#     #   height = 85
#     #   type = 'png'
#     #   ResizeImage(filein, fileout, width, height, type)
#
# import os
# import numpy as np
#
#
# def get_image_paths(facedir):
#     image_paths = []
#     if os.path.isdir(facedir):
#         images = os.listdir(facedir)
#         image_paths = [os.path.join(facedir, img) for img in images]
#     return image_paths
#
#
# class ImageClass():
#     "Stores the paths to images for a given class"
#
#     def __init__(self, name, image_paths):
#         self.name = name
#         self.image_paths = image_paths
#
#     def __str__(self):
#         return self.name + ', ' + str(len(self.image_paths)) + ' images'
#
#     def __len__(self):
#         return len(self.image_paths)
#
#
# def get_dataset(path, has_class_directories=True):
#     dataset = []
#     path_exp = os.path.expanduser(path)
#     classes = [path for path in os.listdir(path_exp) \
#                if os.path.isdir(os.path.join(path_exp, path))]
#     classes.sort()
#     nrof_classes = len(classes)
#     for i in range(nrof_classes):
#         class_name = classes[i]
#         facedir = os.path.join(path_exp, class_name)
#         image_paths = get_image_paths(facedir)
#         dataset.append(ImageClass(class_name, image_paths))
#
#     return dataset
#
#     #
#     # Aaron_Sorkin, 2 images
#     # 2 文件夹里面有几张图片
#     # ['E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Sorkin\\Aaron_Sorkin_0001.png',
#     #  'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Sorkin\\Aaron_Sorkin_0002.png']
#     # 列出所有的文件夹和图片的路径之后，就可以自己写pairs.txt文件了
#
#
# def get_image_paths_and_labels(dataset):  # 获得所有图片的路径和类别
#     image_paths_flat = []  # 所有图片的路径
#     labels_flat = []  # 所有图片的类别： 0，1，2，3....按照这样的方式来标图片的类别
#     for i in range(len(dataset)):
#         image_paths_flat += dataset[i].image_paths  # 数据集中每一个子文件夹都是一个类比，不论里面有几张图片
#         labels_flat += [i] * len(dataset[i].image_paths)
#     return image_paths_flat, labels_flat
# #
# # # 改变图片的大小
# # # 1.这里应该使用多线程来加速
# # paths = r'E:\MyPythonProjects\captchaResize'
# # dataset = get_dataset(paths)
# # image_paths_flat, labels_flat = get_image_paths_and_labels(dataset)
# # index=1
# # for filein in image_paths_flat:
# #     fileout = filein
# #     ResizeImage(filein, fileout, width=90, height=90, type='png')
# #     if index%1000==0:
# #         print('have Done: ',index)
# #     index=index+1
# # # 一共895000张图片
# '''
# def add_zeros(index):
#     index_str = str(index) + ""
#     return index_str.zfill(4)
#     '''
# def change_image_name(class_name='',index=0):
#     target_path = r'E:\MyPythonProjects\captchaOneChineseTrain\dataset/'+class_name+'/'
#     return target_path+class_name+"_"+str(index).zfill(4)+'.png'
#
# # 建立目录
# '''
# os.mkdir("E://bb")
# '''
#
# # # # 改变图片的名字
# # paths = r'E:\MyPythonProjects\captchaOneChinese/'
# # dataset = get_dataset(paths)
# # for i in range(len(dataset)):
# #     class_name=dataset[i].name
# #     target_path=r'E:\MyPythonProjects\captchaOneChineseTrain\dataset/'+class_name
# #     os.mkdir(target_path)# 建立目录
# #     number_of_images_in_the_class=len(dataset[i].image_paths )
# #     for j in range(0,number_of_images_in_the_class):
# #         image_name=dataset[i].image_paths[j]
# #         img = Image.open(image_name)
# #         new_image_name=change_image_name(class_name=class_name,index=j+1)
# #         img.save(new_image_name, type='png')
# #     print('class number is = ',i)
# #
# # # 输出需要的信息由于测试聚类的效果
# # 文件中用前10类的汉字图片做聚类测试使用
#
# def output_images_for_cluster_test(path,filename):
#     dataset=get_dataset(path) # 读取全部的文件夹是一个很耗时的操作
#     dataset_used=dataset[0:10] # 前10个类
#     f=open(filename,'w')
#     for data in dataset_used:
#         name=data.name
#         # 每个类取200张图片
#         index=1
#         for num in range(100):
#             f.write(name+' '+str(index)+' '+str(index+1)+'\n')
#             index=index+2
#     f.close()
#
# def output_images_for_compute_chinese_embedding(path,filename):
#     dataset=get_dataset(path)
#     f = open(filename, 'w')
#     f.write(str(1)+" "+str(130000)+'\n')
#     for data in dataset:
#         name = data.name
#         # 每一个训练数据有13张图片
#         index = 0
#         for num in range(13):
#             f.write(name + ' ' + str(index) + ' ' + str(index) + '\n')
#             index = index + 1
#     f.close()
# # 所有汉字图片数据的文件夹
# # path=r'E:\MyPythonProjects\resizedImage/'
# # # 将需要的信息写入这个文件
# # filename=r'E:\MyPythonProjects\captchaOneChineseTrain/chineseAll.txt'
# # output_images_for_compute_chinese_embedding(path,filename)
#
# # paths = r'E:\MyPythonProjects\HWDB1\train1\train/'
# # filename=r'E:\MyPythonProjects\HWDB1\train1\pairs/dataForCluster.txt'
# # output_images_for_cluster_test(path=paths,filename=filename)
#
#
# def file2matrix(filename):
#     fr=open(filename)
#     arrayOfLines=fr.readlines()
#     numberOfLines=len(arrayOfLines)
#     data=np.zeros((numberOfLines,3))
#     index=0
#     for line in arrayOfLines:
#         line=line.strip()
#         listFromLine=line.split(' ')
#         data[index,:]=listFromLine[0:3]
#         index+=1
#     return data
#     # 绘制三维图像
#
# #
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # dataFileName=r'E:\MyPythonProjects\HWDB1\train1\pairs/embeddingsOfCluster.txt'
# # data = file2matrix(dataFileName)
# #
# # x, y, z = data[:, 0], data[:, 1], data[:, 2]
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')  # 创建一个三维的绘图工程
# # #  将数据点分成三部分画，在颜色上有区分度
# # # b: blue
# # # g: green
# # # r: red
# # # c: cyan
# # # m: magenta
# # # y: yellow
# # # k: black
# # # w: white
# #
# # ax.scatter(x[0:200], y[0:200], z[0:200], c='b')  # 绘制数据点
# # ax.scatter(x[200:400], y[200:400], z[200:400], c='r')
# # ax.scatter(x[400:600], y[400:600], z[400:600], c='g')
# # ax.scatter(x[600:800], y[600:800], z[600:800], c='k')
# # ax.scatter(x[800:1000], y[800:1000], z[800:1000], c='y')
# # ax.scatter(x[1000:1200], y[1000:1200], z[1000:1200], c='c')
# # # 使用plt的缺点是：无法旋转图像查看效果
# # ax.set_zlabel('Z')  # 坐标轴
# # ax.set_ylabel('Y')
# # ax.set_xlabel('X')
# # plt.show()
#
# # 1.使用多线程是加速最快的方式，实现要简单
# # 2.在使用多线程的同时，使用运算性能更加强的硬件可以很好的起到加速的作用
# # 3.可是使用分布式的平台来进行操作：每一个机器上面再使用多线程的程序。
# # 4.可以使用多个进程，每个进程中有多个线程。
# # 实现从简单到容易： 1，4，2，3

def file2list(filename):
    f=open(filename)
    arrayOfLines=f.readlines()
    data=[]
    for line in arrayOfLines:
        data.append(line.strip())
    return data
    # for i in range(len(arrayOfLines)):
    #     print(arrayOfLines[i])


# 正确的标注结果
filename=r'E:\MyPythonProjects\mycaptcha\captdata\data-5\train/mappings.txt'
list1=file2list(filename)
print(list1)
# 模型预测的结果
filename2=r"E:\MyPythonProjects\captchaOneChineseTrain/newResult.txt"
list2=file2list(filename2)
print(list2)
print(len(list1),len(list2))
num=0
# 统计预测正确的数量，计算正确率
for i in range(min([len(list1),len(list2)])):
    if list1[i]==list2[i]:
        num+=1
print('accuracy= ',num/len(list1))
