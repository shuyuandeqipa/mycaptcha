import os
import numpy as np

# 从一个文件夹中将所有的图片的路径保存下来
# ['E:\\MyPythonProjects\\facenet-master\\data\\images\\Anthony_Hopkins_0001.jpg',
#  'E:\\MyPythonProjects\\facenet-master\\data\\images\\Anthony_Hopkins_0002.jpg']
def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


#
# Aaron_Sorkin, 2 images
# 2 文件夹里面有几张图片
# ['E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Sorkin\\Aaron_Sorkin_0001.png',
#  'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Sorkin\\Aaron_Sorkin_0002.png']
# 列出所有的文件夹和图片的路径之后，就可以自己写pairs.txt文件了


def get_image_paths_and_labels(dataset):  # 获得所有图片的路径和类别
    image_paths_flat = []  # 所有图片的路径
    labels_flat = []  # 所有图片的类别： 0，1，2，3....按照这样的方式来标图片的类别
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths  # 数据集中每一个子文件夹都是一个类比，不论里面有几张图片
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def shuffle_examples(image_paths, labels):  # 把数据集打乱
    shuffle_list = list(zip(image_paths, labels))
    np.random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff

'''
加载所有的类别

一个类别的图片（300对）
    1.把图片数量>=2的过滤出来。这里大于几可以自己设一下
    2.有放回的抽取300个类，这样可以保证有一定的重复率。
    2.每个类从中随机抽取2张图片作为一对
不同类别的图像
    1.随机选择600个类 ，先shuffle，再选择600个即可
    2.每个类中选择一个图片（组成300对）
这里采用的是10Ford交叉验证
要求，每一个Ford选择的图片尽可能的不重复
把每一次抽中的图片从待选择的文件夹中去掉即可
如果图片的类别数量不够，可以修改这里的每一折选择的图片的数量
可以使用remove函数试试
'''
# 这里的写法，对于数据量比较小的情况下，可以凑合使用。但是，也做不到全部测试数据的效果。
def gen_pairs_txt(path,number_of_images_in_a_ford=300,number_of_fords=10,filename=r'E:\MyPythonProjects\captchaOneChineseTrain/pairs.txt'):

    dataset = get_dataset(path)
    number_of_all_classes_in_original_dataset=len(dataset)
    all_image_index = [index for index in range(number_of_all_classes_in_original_dataset)]
    classes_with_images_more_than_two = []
    # 筛选图片数量>=2的class
    for i in range(number_of_all_classes_in_original_dataset):
        if len(dataset[i].image_paths) >= 2:  # 如果图片的数量>=2
            classes_with_images_more_than_two.append(i)
    print('classes_with_images_more_than_two',len(classes_with_images_more_than_two))

    f=open(filename,'w') # 打开文件
    f.write(str(number_of_fords)+' '+str(number_of_images_in_a_ford)+'\n')
    for fords in range(number_of_fords):
        print("*******************************************************************")
        print('************************     ',fords)
        print("*******************************************************************")

        same_used_classes = []
        diff_used_classes = []

        np.random.shuffle(classes_with_images_more_than_two)
        # 从里面选择300个类
        # print('len(classes_with_images_more_than_two)=',len(classes_with_images_more_than_two))
        for i in range(np.min([number_of_images_in_a_ford, len(classes_with_images_more_than_two)])):
            same_class_index = classes_with_images_more_than_two[i]
            same_used_classes.append(same_class_index)
            # 从每个类中随机选择2张图片
            class_name = dataset[same_class_index].name
            number_of_images_in_a_class = len(dataset[same_class_index])
            images_in_a_class = [index for index in range(number_of_images_in_a_class)]
            np.random.shuffle(images_in_a_class)
            # 在lfw数据集中图片的下标是从1开始的
            first_image = images_in_a_class[0] + 1
            second_image = images_in_a_class[1] + 1
            # print(class_name, first_image, second_image)  # 把这个保存到文件中去
            f.write(class_name+' '+str(first_image)+' '+str(second_image)+'\n')
        # 把用过的类从有>=2张图片的类的集合中去掉
        # for i in range(len(same_used_classes)):
        #     classes_with_images_more_than_two.remove(same_used_classes[i])

        # 选择不同类别的图片
        # 从全部的图片中随机选择600个类

        np.random.shuffle(all_image_index)
        for i in range(np.min([number_of_images_in_a_ford * 2, len(all_image_index)])):
            diff_used_classes.append(all_image_index[i])
        diff_first_image_indexs = diff_used_classes[0::2]
        diff_second_image_indexs = diff_used_classes[1::2]
        for i in range(len(diff_first_image_indexs)):
            # 在lfw数据集中图片的下标是从1开始的
            diff_first_image_name = dataset[diff_first_image_indexs[i]].name
            diff_first_image = np.random.randint(0, len(dataset[diff_first_image_indexs[i]].image_paths)) + 1

            diff_second_image_name = dataset[diff_second_image_indexs[i]].name
            diff_second_image = np.random.randint(0, len(dataset[diff_second_image_indexs[i]].image_paths)) + 1
            # print(diff_first_image_name, diff_first_image, diff_second_image_name, diff_second_image)  # 写到文件中去
            f.write(diff_first_image_name + ' ' + str(diff_first_image) + ' '+ diff_second_image_name+' '+str(diff_second_image)+'\n')
        # 将用过的从全部的里面去掉
        # for i in diff_used_classes:
        #     all_image_index.remove(i)
        # for i in same_used_classes:
        #     if all_image_index.count(i) > 0:
        #         all_image_index.remove(i)
        # 这里的写法好像和pairs.txt中的规律不一样，但是，应该不影响正确率的判断
    f.close()# 关闭文件


# 下面这段代码是针对手写汉字识别这样的具有大量样本的数据集
# 从里面得到可以充分利用数据集数据的抽样获得pairs.txt文件的方法
def gen_pairs_txt_from_chinese_handwrite_dataset(path, number_of_images_in_a_fold=370, number_of_folds=10, filename=''):
    dataset = get_dataset(path)
    number_of_all_classes_in_original_dataset = len(dataset)  # 一共拥有的类别的数量
    # 所有的类别的下标
    all_image_index = [index for index in range(number_of_all_classes_in_original_dataset)]
    for fords in range(number_of_folds):
        print("*******************************************************************")
        print('************************       ', fords)
        print("*******************************************************************")

        same_used_classes = []
        diff_used_classes = []

        # 先随机选择400个类别
        np.random.shuffle(all_image_index)
        selected_class_index=all_image_index[0:int(number_of_images_in_a_fold)]
        # 需要再想一想，要使用到 5% 左右的数据来进行测试。而且是使用 10 折交叉验证




#     same_used_classes=[]
#     diff_used_classes=[]
#
#
#
#     np.random.shuffle(classes_with_images_more_than_two)
#     # 从里面选择300个类
#     # print('len(classes_with_images_more_than_two)=',len(classes_with_images_more_than_two))
#     for i in range(np.min([number_of_images_in_a_ford,len(classes_with_images_more_than_two)])):
#         same_class_index=classes_with_images_more_than_two[i]
#         same_used_classes.append(same_class_index)
#         # 从每个类中随机选择2张图片
#         class_name=dataset[same_class_index].name
#         number_of_images_in_a_class=len(dataset[same_class_index])
#         images_in_a_class=[index for index in range(number_of_images_in_a_class)]
#         np.random.shuffle(images_in_a_class)
#         # 在lfw数据集中图片的下标是从1开始的
#         first_image=images_in_a_class[0]+1
#         second_image=images_in_a_class[1]+1
#         print(class_name,first_image,second_image)# 把这个保存到文件中去
#     # 把用过的类从有>=2张图片的类的集合中去掉
#     for i in range(len(same_used_classes)):
#         classes_with_images_more_than_two.remove(same_used_classes[i])
#
#     # 选择不同类别的图片
#     # 从全部的图片中随机选择600个类
#
#     np.random.shuffle(all_image_index)
#     for i in range(np.min([number_of_images_in_a_ford*2,number_of_all_classes_in_original_dataset])):
#         diff_used_classes.append(all_image_index[i])
#     diff_first_image_indexs =diff_used_classes[0::2]
#     diff_second_image_indexs=diff_used_classes[1::2]
#     for i in range(len(diff_first_image_indexs)):
#         # 在lfw数据集中图片的下标是从1开始的
#         diff_first_image_name=dataset[diff_first_image_indexs[i]].name
#         diff_first_image=np.random.randint(0,len(dataset[diff_first_image_indexs[i]].image_paths))+1
#
#         diff_second_image_name=dataset[diff_second_image_indexs[i]].name
#         diff_second_image=np.random.randint(0,len(dataset[diff_second_image_indexs[i]].image_paths))+1
#         print(diff_first_image_name,diff_first_image,diff_second_image_name,diff_second_image) # 写到文件中去
#     # 将用过的从全部的里面去掉
#     for i in diff_used_classes:
#         all_image_index.remove(i)
#     for i in same_used_classes:
#         if all_image_index.count(i)>0:
#             all_image_index.remove(i)
#     # 这里的写法好像和pairs.txt中的规律不一样，但是，应该不影响正确率的判断
#
#
### 测试代码
#
# path=r'E:\MyPythonProjects\facenet-master\datasets\lfw_mtcnnpy_160/'
# dataset=get_dataset(path)
# print(len(dataset))
# print(dataset[5])
# print(len(dataset[5]))
# print(dataset[5].image_paths)
# print("-------------------------------------------")
# image_paths_flat, labels_flat=get_image_paths_and_labels(dataset)
# print(image_paths_flat)
# print(labels_flat)
'''
['E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Eckhart\\Aaron_Eckhart_0001.png',
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Guiel\\Aaron_Guiel_0001.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Patterson\\Aaron_Patterson_0001.png',
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Peirsol\\Aaron_Peirsol_0001.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Peirsol\\Aaron_Peirsol_0002.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Peirsol\\Aaron_Peirsol_0003.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Peirsol\\Aaron_Peirsol_0004.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Pena\\Aaron_Pena_0001.png',
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Sorkin\\Aaron_Sorkin_0001.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Sorkin\\Aaron_Sorkin_0002.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Tippin\\Aaron_Tippin_0001.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abba_Eban\\Abba_Eban_0001.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abbas_Kiarostami\\Abbas_Kiarostami_0001.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abdel_Aziz_Al-Hakim\\Abdel_Aziz_Al-Hakim_0001.png',
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abdel_Madi_Shabneh\\Abdel_Madi_Shabneh_0001.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abdel_Nasser_Assidi\\Abdel_Nasser_Assidi_0001.png',
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abdel_Nasser_Assidi\\Abdel_Nasser_Assidi_0002.png']
[0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 11] 
'''
# image_paths_shuff, labels_shuff=shuffle_examples(image_paths_flat,labels_flat)
# print(image_paths_shuff)
# print(labels_shuff)
'''
('E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Patterson\\Aaron_Patterson_0001.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Eckhart\\Aaron_Eckhart_0001.png', 
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Sorkin\\Aaron_Sorkin_0001.png',
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Guiel\\Aaron_Guiel_0001.png', 
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Peirsol\\Aaron_Peirsol_0004.png',
'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abdel_Nasser_Assidi\\Abdel_Nasser_Assidi_0001.png',
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Pena\\Aaron_Pena_0001.png', 
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Tippin\\Aaron_Tippin_0001.png', 
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abdel_Madi_Shabneh\\Abdel_Madi_Shabneh_0001.png', 
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Sorkin\\Aaron_Sorkin_0002.png', 
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Peirsol\\Aaron_Peirsol_0001.png', 
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abbas_Kiarostami\\Abbas_Kiarostami_0001.png', 
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Peirsol\\Aaron_Peirsol_0002.png', 
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Aaron_Peirsol\\Aaron_Peirsol_0003.png', 
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abba_Eban\\Abba_Eban_0001.png', 
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abdel_Aziz_Al-Hakim\\Abdel_Aziz_Al-Hakim_0001.png', 
 'E:\\MyPythonProjects\\facenet-master\\datasets\\val_data/Abdel_Nasser_Assidi\\Abdel_Nasser_Assidi_0002.png')
(2, 0, 5, 1, 3, 11, 4, 6, 10, 5, 3, 8, 3, 3, 7, 9, 11)
'''
print('*********************************************************************************')
print('*********************************************************************************')
print('*********************************************************************************')

# path=r'E:\MyPythonProjects\facenet-master\datasets\lfw_mtcnnpy_160/'
# 手写汉字图片的路径
path=r'E:\MyPythonProjects\captchaOneChineseTrain\dataset/'
dataset=get_dataset(path)
print(len(dataset))
# 写到的目标文件 E:\MyPythonProjects\captchaOneChineseTrain/pairs.txt
filename=r'E:\MyPythonProjects\captchaOneChineseTrain/pairs.txt'
#
gen_pairs_txt(path,number_of_images_in_a_ford=600,number_of_fords=10,filename=filename)

# 这里的代码解释都是从实际应用的概念出发进行解释的。但是，只要数据准备是靠谱的，那么，这里面数据的处理逻辑是一样的。

