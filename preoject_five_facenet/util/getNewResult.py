"""
这里的代码是用来做模型集成的
"""
import os
import numpy as np
# 从文件中加载数据
def file2list(filename):
    f=open(filename)
    arrayOfLines=f.readlines()
    data=[]
    for line in arrayOfLines:
        data.append(line.strip())
    return data
    # for i in range(len(arrayOfLines)):
    #     print(arrayOfLines[i])

def getMaxNum(v):
    num=[]
    for item in v:
        num.append(v.count(item))
    max_num=max(num)
    for i in range(len(num)):
        if num[i]==max_num:
            return v[i]
def getNewChar(v):
    v_different_len=len(set(v))
    if v_different_len==1 or v_different_len==4:
        return v[0]
    if v_different_len==2:
        max_num=getMaxNum(v)
        return max_num
    if v_different_len==3:
        if v[0]==v[1] or v[0]==v[2] or v[0]==v[3]:
            return v[0]
        if v[1]==v[2] or v[1]==v[3]:
            return v[1]
        if v[2]==v[3]:
            return v[2]
# 如果生成的字符串中有重复的数字，那么就把这个字符串换为预测效果最好的模型的结果。
# 在本项目中，最好的模型结果是第一个：20180327-215050.txt
def handleNewString(string,replace):
    v=[]
    for i in range(len(string)):
        v.append(string[i])
    if len(set(v))!=4:
        string =replace
    return string


# 在四个模型的结果中使用投票法选出新的结果
def getNewItem(str1,str2,str3,str4):
    str1=str1[5:len(str1)];str2=str2[5:len(str2)];
    str3 = str3[5:len(str3)];str4=str4[5:len(str4)];
    length_of_string=len(str1)
    newString=""
    for i in range(length_of_string):
        tmp = []
        tmp.append(str1[i]);
        tmp.append(str2[i]);
        tmp.append(str3[i]);
        tmp.append(str4[i]);
        newString+=getNewChar(tmp)
    newString=handleNewString(newString,str1)
    return newString
def getNewResult(list1,list2,list3,list4,newResultFileName):
    number_of_items=len(list1)#一共拥有的测试数据的数量
    fp=open(newResultFileName,'w')
    for item in range(number_of_items):
        newItem=getNewItem(list1[item],list2[item],list3[item],list4[item])
        fp.writelines(str(item).zfill(4)+','+newItem+'\n')
    fp.close()


list1=file2list(r"E:\MyPythonProjects\captchaOneChineseTrain/20180327-215050.txt");print(list1)
list2=file2list(r"E:\MyPythonProjects\captchaOneChineseTrain/20180328-142024.txt")
list3=file2list(r"E:\MyPythonProjects\captchaOneChineseTrain/20180329-210826.txt")
list4=file2list(r"E:\MyPythonProjects\captchaOneChineseTrain/20180330-121755.txt")
getNewResult(list1,list2,list3,list4,r"E:\MyPythonProjects\captchaOneChineseTrain/newResult.txt")

