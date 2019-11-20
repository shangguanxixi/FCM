import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl   #显示中文
import math
from numpy import random


def MSE(A1, data2):
    dist_temp = (np.power((A1 - data2), 2)) / Num1
    dist = dist_temp.sum(axis=0)
    # for i in range(0, Num2):
    #
    #     dist[i] = np.sum(dist_temp[:, i])
    return dist


def RMSE(M):
    dist = np.power(M, 0.5)
    return dist


def jdwucha(A1, data2):
    # A1 = cala(W, data, Num1)  # 更新后的矩阵
    # A1 = np.array(A1)
    # dist = [0] * Num2
    dist_temp = np.abs(A1 - data2)
    dist = dist_temp.sum(axis=0) / (Num1)
    return dist


def xdwucha(data1, data2):  # data1是预测的矩阵；data2是要做比较的真实矩阵

    # dist = [0] * Num2
    dist_temp = (np.abs((data1 - data2) / data2)) / (Num1)
    dist = dist_temp.sum(axis=0)
    return dist

def f(x):
    y = 1/(1+np.exp(-x))
    return y

def caa(W,data):
    a = data.shape[0]
    A2 = [0]*a
    for o in range (0,a):
        temp = np.dot(data,W)
        A2 = f(temp)


    return A2

def wucha(A1,A2):
    dist_temp = A1 - A2
    Bm = dist_temp.sum(axis=0) / 12
    return Bm
def wucha2(A1,A2):
    dist_temp = abs(A1 - A2)
    # Bm = dist_temp.sum(axis=0) / 12
    return dist_temp

def bijiao(A1,A2):
    b = A1.shape[0]
    for i in range(0,b):
        s = A1[i]
        q = A2[i]
        if s >= q:
            return False
        else:
            continue
    return True


jd = [0] * 6
xd = [0] * 6
mse = [0] * 6
arr1 = pd.read_csv('1zhaoqing.csv',header=None)
data1 = np.array(arr1)
w = 504
size = 50
data_real = data1[w:w + size, :]                  # 测试集真实数据
Num1=data_real.shape[0]                             #测试集行数
Num2=data_real.shape[1]                             #列数12

data_pred = [0] * size
n = 0.04
m = 0.1
E = [0.2]*6
k = 0.09      #修正率

for i in range(0, size):
    data_test = data1[w + i]
    first_w = random.random(size=(6,6))
    first_w = pd.DataFrame(first_w)

    X = data1[i:i + w, :]
    for j in range(0,w-1):
        x_test = X[j]     #输入这一时刻的值
        y_pred = caa(first_w,x_test)  #得到预测的下一时刻节点状态值
        y_real = X[j + 1]  #下一时刻真实值

        pwucha = wucha(y_real,y_pred)
        abpwucha = wucha2(y_real,y_pred)

        if bijiao(abpwucha,E):
            print(abpwucha)
            break
        else:
            # 权重矩阵更新

            for l in range(0, first_w.shape[0]):
                for p in range(0, first_w.shape[1]):
                    part1 = (n + k * pwucha * (1 - pwucha)) * y_pred[l]*y_pred[p]
                    part2 = k * pwucha * (1 - pwucha) * first_w.iloc[l][p] * y_pred[l]
                    part3 = n * first_w.iloc[l][p] * math.pow(y_pred[l], 2)

                    first_w.iloc[l][p] = (1-m)*first_w.iloc[l][p] + part1 + part2 - part3

    data_pred[i] = caa(first_w, data_test)

A1 = np.array(data_pred)  # 预测出的数据
Mwucha = MSE(A1, data_real)
Rwucha = RMSE(Mwucha)

jd = jdwucha(A1, data_real)
xd = xdwucha(A1, data_real)

# print(bestw)
print('绝对误差：{}'.format(jd))
print('相对误差：{}'.format(xd))
# print('欧式距离：{}'.format(swucha))
print('MSE：{}'.format(Mwucha))
print('RMSE:{}'.format(RMSE(Mwucha)))


temp = [0]*5
temp[0] = ["CO","NO2","SO2","O3","PM25","PM10"]
temp[1] = jd
temp[2] = xd
temp[3] = Mwucha
temp[4] = Rwucha
temp = np.array(temp)
temp = pd.DataFrame(temp)
temp.to_csv('NHL_9yue_zhaoqing.csv' , header=None)














