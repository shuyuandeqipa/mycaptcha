from os.path import join
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
math_op=['+','-','*'] # 43 45 42
gen_char_set = number + math_op
# 有先后的顺序的
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 350
MAX_CAPTCHA = 8
print("验证码文本最长字符数", MAX_CAPTCHA)
CHAR_SET_LEN = len(gen_char_set)
print('CHAR_SET_LEN:', CHAR_SET_LEN)

home_root = 'E:\MyPythonProjects\mycaptcha\project_data_one/'
workspace = join(home_root, 'work/')  # 用于工作的训练数据集
model_path = join(home_root, 'work/model')
model_tag = 'crack_capcha.model'
save_model = join(model_path, model_tag)
print('model_path:', save_model)

# 输出日志 tensorboard监控的内容
tb_log_path='E:\MyPythonProjects\mycaptcha\project_data_one\logs' # 修改tensorboard的日志路径
# 把预测的结果写入这个文件中
predicted_result_file_path='E:\MyPythonProjects\mycaptcha\project_data_one/predicted.txt'

#通过预处理项目project_data_one_preprocessing处理的关于所有图片中每张图片拥有的字符数量的文件路径
# 这个文件和mappings相对应。
needed_labels_length_path='E:\MyPythonProjects\mycaptcha\project_data_one_preprocessing/predicted.txt'