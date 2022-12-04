# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 20:44:04 2019

@author: Moran Xu
"""


import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
 
 
"""
功能：完后对三次样条函数求解方程参数的输入
参数：要进行三次样条曲线计算的自变量
返回值：方程的参数
"""
def calculateEquationParameters(x):
    #parameter为二维数组，用来存放参数，sizeOfInterval是用来存放区间的个数
    parameter = []
    sizeOfInterval=len(x)-1;
    i = 1
    #首先输入方程两边相邻节点处函数值相等的方程为2n-2个方程
    while i < len(x)-1:
        data = init(sizeOfInterval*4)
        data[(i-1)*4] = x[i]*x[i]*x[i]
        data[(i-1)*4+1] = x[i]*x[i]
        data[(i-1)*4+2] = x[i]
        data[(i-1)*4+3] = 1
        data1 =init(sizeOfInterval*4)
        data1[i*4] =x[i]*x[i]*x[i]
        data1[i*4+1] =x[i]*x[i]
        data1[i*4+2] =x[i]
        data1[i*4+3] = 1
        temp = data[2:]
        parameter.append(temp)
        temp = data1[2:]
        parameter.append(temp)
        i += 1
   # 输入端点处的函数值。为两个方程, 加上前面的2n - 2个方程，一共2n个方程
    data = init(sizeOfInterval * 4 - 2)
    data[0] = x[0]
    data[1] = 1
    parameter.append(data)
    data = init(sizeOfInterval * 4)
    data[(sizeOfInterval - 1) * 4 ] = x[-1] * x[-1] * x[-1]
    data[(sizeOfInterval - 1) * 4 + 1] = x[-1] * x[-1]
    data[(sizeOfInterval - 1) * 4 + 2] = x[-1]
    data[(sizeOfInterval - 1) * 4 + 3] = 1
    temp = data[2:]
    parameter.append(temp)
    # 端点函数一阶导数值相等为n-1个方程。加上前面的方程为3n-1个方程。
    i=1
    while i < sizeOfInterval:
        data = init(sizeOfInterval * 4)
        data[(i - 1) * 4] = 3 * x[i] * x[i]
        data[(i - 1) * 4 + 1] = 2 * x[i]
        data[(i - 1) * 4 + 2] = 1
        data[i * 4] = -3 * x[i] * x[i]
        data[i * 4 + 1] = -2 * x[i]
        data[i * 4 + 2] = -1
        temp = data[2:]
        parameter.append(temp)
        i += 1
    # 端点函数二阶导数值相等为n-1个方程。加上前面的方程为4n-2个方程。且端点处的函数值的二阶导数为零，为两个方程。总共为4n个方程。
    i = 1
    while i < len(x) - 1:
        data = init(sizeOfInterval * 4)
        data[(i - 1) * 4] = 6 * x[i]
        data[(i - 1) * 4 + 1] = 2
        data[i * 4] = -6 * x[i]
        data[i * 4 + 1] = -2
        temp = data[2:]
        parameter.append(temp)
        i += 1
    return parameter
 
 
 
"""
对一个size大小的元组初始化为0
"""
def init(size):
    j = 0;
    data = []
    while j < size:
        data.append(0)
        j += 1
    return data
 
"""
功能：计算样条函数的系数。
参数：parametes为方程的系数，y为要插值函数的因变量。
返回值：三次插值函数的系数。
"""
 
def solutionOfEquation(parametes, x, y):
    sizeOfInterval = len(x) - 1;
    result = init(sizeOfInterval*4-2)
    i=1
    while i<sizeOfInterval:
        result[(i-1)*2]=y[i]
        result[(i-1)*2+1]=y[i]
        i+=1
    result[(sizeOfInterval-1)*2]=y[0]
    result[(sizeOfInterval-1)*2+1]=y[-1]
    a = np.array(calculateEquationParameters(x))
    b = np.array(result)
#    for data_x in b:
#        print(data_x)
    return np.linalg.solve(a,b)
 
"""
功能：根据所给参数，计算三次函数的函数值：
参数:parameters为二次函数的系数，x为自变量
返回值：为函数的因变量
"""
def calculate(paremeters,x):
    result=[]
    for data_x in x:
        result.append(paremeters[0]*data_x*data_x*data_x+paremeters[1]*data_x*data_x+paremeters[2]*data_x+paremeters[3])
    return  result
 
 
"""
功能：将函数绘制成图像
参数：data_x,data_y为离散的点.new_data_x,new_data_y为由拉格朗日插值函数计算的值。x为函数的预测值。
返回值：空
"""
def  Draw(data_x,data_y,new_data_x,new_data_y):
        plt.plot(new_data_x, new_data_y, label="fitttingCurve", color="black")
        plt.scatter(data_x,data_y, label="discretedata",color="red")
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        plt.title("3 spline function")
        plt.legend(loc="upper left")
        plt.show()
 
if __name__ == '__main__':
    """
三次样条实现：
函数x的自变量为:3,   4.5, 7,    9
      因变量为：2.5, 1   2.5,  0.5
"""
    x = [3, 4.5, 7, 9]
    y = [2.5, 1, 2.5, 0.5]
    result=solutionOfEquation(calculateEquationParameters(x), x, y)
    new_data_x1=np.arange(3, 4.5, 0.1)
    new_data_y1=calculate([0,0,result[0],result[1]],new_data_x1)
    new_data_x2=np.arange(4.5, 7, 0.1)
    new_data_y2=calculate([result[2],result[3],result[4],result[5]],new_data_x2)
    new_data_x3=np.arange(7, 9.5, 0.1)
    new_data_y3=calculate([result[6],result[7],result[8],result[9]],new_data_x3)
    new_data_x=[]
    new_data_y=[]
    new_data_x.extend(new_data_x1)
    new_data_x.extend(new_data_x2)
    new_data_x.extend(new_data_x3)
    new_data_y.extend(new_data_y1)
    new_data_y.extend(new_data_y2)
    new_data_y.extend(new_data_y3)
    Draw(x,y,new_data_x,new_data_y)
