"""
专门做预测的
"""
import time

import numpy as np
import tensorflow as tf

from project_data_one.cfg import MAX_CAPTCHA, CHAR_SET_LEN, model_path,predicted_result_file_path,needed_labels_length_path
from project_data_one.cnn_sys import crack_captcha_cnn, X, keep_prob
from project_data_one.utils import convert2gray, vec2text,change_text_to_mathematical_formula_compute_result_text
from project_data_one.get_data import labels_path,get_one_predict_image_data,get_one_predict_image_data_with_index

def hack_function(sess, predict, captcha_image):
    """
    装载完成识别内容后，
    :param sess:
    :param predict:
    :param captcha_image:
    :return:
    """
    text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector)



def batch_hack_captcha():

    # 定义预测计算图
    output = crack_captcha_cnn()
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)

    saver = tf.train.Saver()

    # windows环境下，tf不会分配所有可用的内存，因此需要手动设置，允许动态内存分配增长。
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # 允许内存动态分配



    with tf.Session(config=config) as sess:
        # saver = tf.train.import_meta_graph(save_model + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint(model_path))  # 加载刚才训练好的模型的训练参数

        stime = time.time() # 用来计算时间，单位是秒
        task_cnt = 5000  #需要预测的图片的数量
        right_cnt = 0   # 统计预测正确的数量
        # 加载mappings.txt中的所有labels；验证正确率需要这个标记好了的数据
        # labels_path目前在get_data.py中
        all_labels_data = np.genfromtxt(labels_path, dtype='str')
        # 这个文件是project_data_one_processing输出的结果。预测的是图片中的字符的数量。
        # 是用0,1,2,3分别代表图片中字符数量为5,6,7,8
        all_needed_labels_length=np.genfromtxt(needed_labels_length_path,dtype='str')

        # 打开文件
        fp= open(predicted_result_file_path, 'w')

        for i in range(task_cnt):
            # image,random_index= get_one_predict_image_data()  #产生一张新的测试图片
            image_index=i # 需要预测的图片的序号(在数据集中的序号，比如：9000-9999)
            image=get_one_predict_image_data_with_index(image_index) #加载一张新的测试图片
            image = convert2gray(image) #把三通道图片转为灰度图片
            image = image.flatten() / 255 #把数据进行归一化
            string = all_labels_data[image_index]
            a = string.index(',')
            b = string.index('=')
            text=string[a+1:b]
            predict_text = hack_function(sess, predict, image) #预测图像的标记
            # 这里需要+5.因为txt文件中5,6,7,8分别用0,1,2,3代替
            needed_length=int(all_needed_labels_length[image_index][-1])+5
            predict_text=predict_text[0:needed_length]
            label_expression = string[a + 1:]
            # 转换成完整的表达式(string类型)，比如：13*10-16=114
            predict_expression = change_text_to_mathematical_formula_compute_result_text(predict_text)
            # 写入文件 ：比如：0002,13*10-16=114 为了和mappings.txt中文件格式一模一样。
            fp.writelines(""+str(image_index).zfill(4)+','+predict_expression+'\n')
            if text == predict_text:
                right_cnt += 1 # 统计正确的数量
                # print("true images: 标记: {}  预测: {}".format(text, predict_text))
                if label_expression==predict_expression:
                    print("true images: 标记: {}  预测: {}".format(label_expression, predict_expression))
                # 需要将预测的写到文件中去
            else:
                print("error images: 标记: {}  预测: {}".format(text, predict_text))
                pass
                # print("标记: {}  预测: {}".format(text, predict_text))
        #关闭文件
        fp.close()
        print('task:', task_cnt, ' cost time:', (time.time() - stime), 's')
        print('right/total-----', right_cnt, '/', task_cnt)

# 一定要注意需要使用到的文件路径的配置
# 如果没有配置标注好的正确的mappings.txt文件，那么预测过程中在输出文件中写入的预测结果是合理的
# 通过阅读代码可以知道，控制台的输出是没用的。所以，最好配置正确的mappings.txt这个文件，用来统计正确率。
if __name__ == '__main__':
    batch_hack_captcha()
    print('end...')

#----------------------------------------------------------------

#-----------------------------------------------------------------
#
# true images: 标记: 20*9-13  预测: 20*9-13
# true images: 标记: 17*8-11  预测: 17*8-11
# true images: 标记: 4-10+1  预测: 4-10+1
# true images: 标记: 4+6+19  预测: 4+6+19
# true images: 标记: 18*13*6  预测: 18*13*6
# true images: 标记: 13*7+14  预测: 13*7+14
# true images: 标记: 15+8-13  预测: 15+8-13
# true images: 标记: 7+3-5  预测: 7+3-5
# task: 1000  cost time: 9.495219945907593 s
# right/total----- 1000 / 1000 正确率 100% 没问题