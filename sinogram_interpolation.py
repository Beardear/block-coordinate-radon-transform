# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:47:49 2019

@author: Moran Xu
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
#import astra
import time

'''
1-order linear interpolation
'''
def sinogram_linear_interpolation(sinogram, shifted_sinogram_interp, column_num, position_start, T_length):
    row_start = int(np.round(position_start))
    row_end = row_start + sinogram.shape[0]    
    
    shift_sinogram = np.zeros(sinogram.shape[0])
    
    length = sinogram.shape[0]-1
    if round(position_start) - position_start >=0:
        step = round(position_start) - position_start
        shift_sinogram[:length] = (sinogram[1:, column_num] - sinogram[0:length, column_num]) * step + sinogram[0:length, column_num]
        shift_sinogram[length] = sinogram[length, column_num]
    else:
        step = 1 - position_start + round(position_start)
        shift_sinogram[0] = sinogram[0, column_num]
        shift_sinogram[1:] = (sinogram[1:, column_num] - sinogram[0:length, column_num]) * step + sinogram[0:length, column_num]
    shifted_sinogram_interp[row_start: row_end, column_num] = shift_sinogram[:]
    return shifted_sinogram_interp


 
 
"""
功能：完后对二次样条函数求解方程参数的输入
参数：要进行二次样条曲线计算的自变量
返回值：方程的参数
"""
def calculateEquationParameters(x):
    #parameter为二维数组，用来存放参数，sizeOfInterval是用来存放区间的个数
    parameter = []
    sizeOfInterval=len(x)-1;
    i = 1
    #首先输入方程两边相邻节点处函数值相等的方程为2n-2个方程
    while i < len(x)-1:
        data = init(sizeOfInterval*3)
        data[(i-1)*3]=x[i]*x[i]
        data[(i-1)*3+1]=x[i]
        data[(i-1)*3+2]=1
        data1 =init(sizeOfInterval*3)
        data1[i * 3] = x[i] * x[i]
        data1[i * 3 + 1] = x[i]
        data1[i * 3 + 2] = 1
        temp=data[1:]
        parameter.append(temp)
        temp=data1[1:]
        parameter.append(temp)
        i += 1
    #输入端点处的函数值。为两个方程,加上前面的2n-2个方程，一共2n个方程
    data = init(sizeOfInterval*3-1)
    data[0] = x[0]
    data[1] = 1
    parameter.append(data)
    data = init(sizeOfInterval *3)
    data[(sizeOfInterval-1)*3+0] = x[-1] * x[-1]
    data[(sizeOfInterval-1)*3+1] = x[-1]
    data[(sizeOfInterval-1)*3+2] = 1
    temp=data[1:]
    parameter.append(temp)
    #端点函数值相等为n-1个方程。加上前面的方程为3n-1个方程,最后一个方程为a1=0总共为3n个方程
    i=1
    while i < len(x) - 1:
        data = init(sizeOfInterval * 3)
        data[(i - 1) * 3] =2*x[i]
        data[(i - 1) * 3 + 1] =1
        data[i*3]=-2*x[i]
        data[i*3+1]=-1
        temp=data[1:]
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
返回值：二次插值函数的系数。
"""

 
def solutionOfEquation(parametes, x, y):
    sizeOfInterval = len(x) - 1;
    result = init(sizeOfInterval*3-1)
    i=1
    while i<sizeOfInterval:
        result[(i-1)*2]=y[i]
        result[(i-1)*2+1]=y[i]
        i+=1
    result[(sizeOfInterval-1)*2]=y[0]
    result[(sizeOfInterval-1)*2+1]=y[-1]
    a = np.array(calculateEquationParameters(x))
    b = np.array(result)
    return np.linalg.solve(a,b)
 
    
"""
功能：根据所给参数，计算二次函数的函数值：
参数:parameters为二次函数的系数，x为自变量
返回值：为函数的因变量
"""
def calculate(paremeters,x):
    result=[]
    for data_x in x:
        result.append(paremeters[0]*data_x*data_x+paremeters[1]*data_x+paremeters[2])
    return  result


"""
功能：将函数绘制成图像
参数：data_x,data_y为离散的点.new_data_x,new_data_y为由拉格朗日插值函数计算的值。x为函数的预测值。
返回值：空
"""
def Draw(data_x,data_y,new_data_x,new_data_y):
    plt.plot(new_data_x, new_data_y, label="fitting_curve", color="black")
    plt.scatter(data_x,data_y, label="discrete data",color="red")
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.title("2 order spline function")
    plt.legend(loc="upper left")
    plt.show()

    
#def SART_projection(image, detector_num, num_of_projections):
#    distance_source_origin = 300  # [mm]
#    distance_origin_detector = 100  # [mm]
#    detector_pixel_size = 1.05 # [mm]
##    detector_rows = 16 # Vertical size of detector [pixels].
##    detector_cols = int(128 * np.sqrt(2)) + 3  # Horizontal size of detector [pixels].
##    num_of_projections = 180
#    detector_num1 = int(detector_num * np.sqrt(2))+1
#    num_of_projections = num_of_projections
##    output_dir = 'projections'
#    print("projection started...")
#    start = time.time()
#    vol_geom = astra.create_vol_geom(image.shape)
##384 is num of detectors
#    proj_geom = astra.create_proj_geom('parallel', 1.0, detector_num1, np.linspace(0,np.pi,num_of_projections,False))
#    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
#    sinogram_id, sinogram = astra.create_sino(image, proj_id)
#    
#    end = time.time()
#    print("projection completed. Total time is " + str(end - start))
##    scipy.io.savemat('projections.mat', mdict={'projections': projections})
#    return np.transpose(sinogram)
#   
#     
#def SART_reconstruction(projections, detector_num, num_of_projections):
#    detector_num1 = int(detector_num * np.sqrt(2))+1
#    angles = np.linspace(0, np.pi, num=num_of_projections, endpoint=False)
#    
#    proj_geom = astra.create_proj_geom('parallel',1, detector_num1, angles)
#    projections_id = astra.data2d.create('-sino', proj_geom, np.transpose(projections))
#    print("projections loaded.")
#    
#    print("reconstruction started...")
#    vol_geom = astra.creators.create_vol_geom(detector_num)
#    
#    # Create a data object for the reconstruction
#    rec_id = astra.data2d.create('-vol', vol_geom)
#    
#    cfg = astra.astra_dict('SART_CUDA')
#    cfg['ReconstructionDataId'] = rec_id
#    cfg['ProjectionDataId'] = projections_id
#    
#    # Create the algorithm object from the configuration structure
#    alg_id = astra.algorithm.create(cfg)
#    
#    # Run 150 iterations of the algorithm
#    astra.algorithm.run(alg_id, 150)
#    
#    # Get the result
#    rec = astra.data2d.get(rec_id)
##    scipy.io.savemat('recon_SART.mat', mdict={'img': rec})
#    
#    astra.algorithm.delete(alg_id)
#    astra.data3d.delete(rec_id)
#    astra.data3d.delete(projections_id)
#    return rec
    
        



if __name__ == '__main__':
    """
    二次样条实现：
    函数x的自变量为:3,   4.5, 7,    9
          因变量为：2.5, 1   2.5,  0.5
    """
    x = [3, 4.5, 7, 9]
    y = [2.5, 1, 2.5, 0.5]
     
    """一共有三个区间，用二次样条求解，需要有9个方程"""
    result=solutionOfEquation(calculateEquationParameters(x), x, y)
    new_data_x1=np.arange(3, 4.5, 0.1)
    new_data_y1=calculate([0,result[0],result[1]],new_data_x1)
    new_data_x2=np.arange(4.5, 7, 0.1)
    new_data_y2=calculate([result[2],result[3],result[4]],new_data_x2)
    new_data_x3=np.arange(7, 9.5, 0.1)
    new_data_y3=calculate([result[5],result[6],result[7]],new_data_x3)
    new_data_x=[]
    new_data_y=[]
    new_data_x.extend(new_data_x1)
    new_data_x.extend(new_data_x2)
    new_data_x.extend(new_data_x3)
    new_data_y.extend(new_data_y1)
    new_data_y.extend(new_data_y2)
    new_data_y.extend(new_data_y3)
    Draw(x,y,new_data_x,new_data_y)