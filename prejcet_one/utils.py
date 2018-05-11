import numpy as np

from project_data_one.cfg import MAX_CAPTCHA,CHAR_SET_LEN
from project_data_one.cfg import predicted_result_file_path

def char2pos(c):
    """
    字符验证码，字符串转成位置信息
    :param c:
    :return:
    """
    if c=='+':
        k=10
    if c=='-':
        k=11
    if c=='*':
        k=12
    if ord(c)>47:
         k = ord(c) - 48 # 0 -> 9
    return k

def pos2char(char_idx):
    """
    根据位置信息转化为索引信息
    :param char_idx:
    :return:
    """
    #
    # if not isinstance(char_idx, int64):
    #     raise ValueError('error')

    if char_idx < 10:
        char_code = char_idx + ord('0')
    elif char_idx ==10:
        char_code=43
    elif char_idx==11:
        char_code=45
    elif char_idx==12:
        char_code=42

    else:
        raise ValueError('error')
    # chr(43)
    # Out[5]: '+'
    # chr(45)
    # Out[6]: '-'
    # chr(42)
    # Out[7]: '*'

    return chr(char_code)

# 传入灰度图片
# 返回用中值滤波去除椒盐噪声后的灰度图片
def handle_noise(im):
    im_copy = np.zeros((im.shape[0], im.shape[1]))
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            im_copy[i][j] = im[i][j]
    # 用3*3的中值滤波器
    step = 3

    def m_filter(x, y):
        sum_s = []
        for k in range(-int(step / 2), int(step / 2) + 1):
            for m in range(-int(step / 2), int(step / 2) + 1):
                sum_s.append(im[x + k][y + m])
        sum_s.sort()
        return sum_s[(int(step * step / 2) + 1)]

    for i in range(int(step / 2), im.shape[0] - int(step / 2)):
        for j in range(int(step / 2), im.shape[1] - int(step / 2)):
            im_copy[i][j] = m_filter(i, j)
    return im_copy

def convert2gray(img):
    """
    把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
    :param img:
    :return:
    """
    if len(img.shape) > 2:
       # gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长%d个字符'%MAX_CAPTCHA)

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    #大写字母的数据
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    # 小写字母的数据也放进去
    # text_lower=text.lower()
    # for i, c in enumerate(text_lower):
    #     idx = i * CHAR_SET_LEN + char2pos(c)
    #     vector[idx] = 1


    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        char_code = pos2char(char_idx)
        text.append(char_code)
    return "".join(text)



# 解析算式，输出数学运算结果的表达式
def change_text_to_mathematical_formula_compute_result_text(text):
    add_op='+'
    minus_op='-'
    multiply_op='*'
    # 找到两个运算符的位置;一个字符一个字符的去找
    op_index=[]
    ops=[]
    for i in range(len(text)):
        c=text[i]
        if not c.isdigit():
            if c==add_op:
                op_index.append(i)
                ops.append(add_op)
            if c==minus_op:
                op_index.append(i)
                ops.append(minus_op)
            if c == multiply_op:
                op_index.append(i)
                ops.append(multiply_op)

    op_index_first=op_index[0]
    op_index_second=op_index[1]
    first_op=ops[0]
    second_op=ops[1]
    # 拆出来了三个运算数字
    first_num=int(text[0:op_index_first])
    second_num=int(text[op_index_first+1:op_index_second])
    third_num=int(text[op_index_second+1:])
    # 进行运算公式的运算
    if first_op=='+' and second_op=='+':
        result = first_num+second_num + third_num
    if first_op == '+' and second_op == '-':
        result = first_num + second_num - third_num
    if first_op == '+' and second_op == '*':
        result = first_num + second_num * third_num

    if first_op == '-' and second_op == '+':
        result = first_num - second_num + third_num
    if first_op == '-' and second_op == '-':
        result = first_num - second_num - third_num
    if first_op == '-' and second_op == '*':
        result = first_num - second_num * third_num

    if first_op == '*' and second_op == '+':
        result = first_num * second_num + third_num
    if first_op == '*' and second_op == '-':
        result = first_num * second_num - third_num
    if first_op == '*' and second_op == '*':
        result = first_num * second_num * third_num
    result_text=str(first_num)+first_op+str(second_num)+second_op+str(third_num)+'='+str(result)
    return result_text


# # 把预测的东西写到一个文本文件中去
# def write_predict_info_to_file():
#     text='12346789'
#     with open(predicted_result_file_path,'w') as fp:
#         fp.writelines(text+'\n')
#         fp.writelines(text + '\n')
#         fp.writelines(text+'\n')
#     fp.close()
#




#
# if __name__ == '__main__':
#     text = '13*6-18'
#     vec=text2vec(text)
#     print(vec)
#     t=vec2text(vec)
#     print(t)

# text=change_text_to_mathematical_formula_compute_result_text('2-55*9')
# print(text)
# write_predict_info_to_file()

#
# labels_path='E:\MyPythonProjects\mycaptcha\captdata\data-1/train/mappings.txt'
# all_labels_data = np.genfromtxt(labels_path,dtype='str')
# my_predited_data=np.genfromtxt(predicted_result_file_path,dtype='str')
# right_num=0
# for i in range(len(all_labels_data)):
#     if all_labels_data[i]==my_predited_data[i]:
#         right_num+=1
# print('the number of right labels is = ',right_num)