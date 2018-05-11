#
#
# def file2list(filename):
#     f=open(filename)
#     arrayOfLines=f.readlines()
#     data=[]
#     for line in arrayOfLines:
#         data.append(line.strip())
#     return data
#     # for i in range(len(arrayOfLines)):
#     #     print(arrayOfLines[i])
#
# # 正确的标注结果
# filename=r'E:\MyPythonProjects\mycaptcha\captdata\data-1\train/mappings.txt'
# list1=file2list(filename)
#
# # 模型预测的结果
# filename2=r"E:\MyPythonProjects\mycaptcha\project_data_one/predicted.txt"
# list2=file2list(filename2)
#
# print(len(list1),len(list2))
# num=0
# # 统计预测正确的数量，计算正确率
# for i in range(min([len(list1),len(list2)])):
#     if list1[i]==list2[i]:
#         num+=1
# print('accuracy= ',num/len(list1))
