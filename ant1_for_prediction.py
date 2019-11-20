import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

from numpy import random
arr1 = pd.read_csv('1zhaoqing.csv',header=None)
data1 = np.array(arr1)
# data2 = pd.read_csv('ceshi5.csv',header=None)
# data2 = np.array(data2)

w = 504
size = 50
data_real = data1[w:w + size, :]  # 测试集真实数据
Num1=data_real.shape[0]                             #测试集行数189
Num2=data_real.shape[1]                             #列数12
MAXIT = 20                                         #最大循环次数
N = 11
rho = 0.5  # 挥发系数

alpha = 1  # 残留信息相对重要度
beta = 5  # 预见值的相对重要度
Q = 10  # 蚁环常数
NumAnt = 20  # 蚂蚁数量
p0 = 0.3                                        #以一定的几率随机选择路径
e = 0.4                                         #局部信息素更新事信息素的衰减率
r = 0.0004                                          #全局信息素更新时的强度
t0 = 0.02                                        #初始信息素

a = 1                                      #预测未来一小时的

data_pred = [0]*size









# h = np.ones(1,10)
# h[4] = 2                                          #增大选择权重取0的概率；但我不知道在这个问题中有没有必要



def code(w):
    w = np.array(w).reshape(1,NC*NC)
    route = np.ceil((np.array(w)+1)*5)
    # route = pd.DataFrame(route,dtype=int)
    return route

def decode(route):
    w = np.array(route) / math.floor(N / 2) - 1
    w = list(w)
    return w


def mul(t,w):                                                #非均匀变异算子
    for i in range(0, NC*NC):
        m = random.randint(0,2,1)
        if m == [0]:
            w[i] = w[i] + tf(t,(1-w[i]))
        else:
            w[i] = w[i] - tf(t, (w[i]+1))
    return w

def tf(t,y):                #t为当前迭代次数
    b = 3
    T = 60000               #最大迭代次数，为了和别的实验做对比
    m = random.random()
    k = math.pow((1-t/T),b)
    a = y * (1-math.pow(m,k))
    return  a



def choose(daltatau):
    #a = random.random()
    a = np.random.random()
    if a <= p0:
        b = np.random.random()
        m = 0
        dsum = sum(daltatau)

        for j in range(0, N):
            m += daltatau[j] / dsum
            # m += daltatau[j]*h[j]/sum
            if b <= m:
                return j
    else:
        k = daltatau.idxmax()
        #daltatau = daltatau.tolist()
        #k = daltatau.index(max(daltatau))
        return k


def path(swucha,daltatau,start):
    #kwucha = 0

    route = [0]*(NC*NC)
    route[0] = int(start)
    daltatau.iloc[route[0], 0] = (1 - e) * daltatau.iloc[route[0], 0] + e * t0  # 局部信息素更新

    for i in range(1, NC*NC):
        route[i] = choose(daltatau.loc[:, i])
        #for j in range(NC*NC):
        b = daltatau.iloc[route[i], i]
        daltatau.iloc[route[i],i] = (1-e)*b + e*t0                #局部信息素更新
    #print(daltatau)

    #exit()
    #评价蚂蚁路径
    ew = decode(route)

    eW = np.array(ew).reshape(NC,NC)
    kwucha = wucha(eW,data)

    return route,kwucha



def f(x):
    y = 1/(1+np.exp(-x))
    return y

def cala(W,data):
    A2 = [0] * ItemNum1
    A2[0] = data[0]
    for i in range(0, ItemNum1-1):
        temp1 = np.dot(data[i], W)
        A2[i + 1] = f(temp1)
    return A2

def wucha(W,data):
    A1 = cala(W,data)                 #更新后的矩阵
    A1 = np.array(A1)

    # data1 = np.ones((6068, 12))                # 预测矩阵与真实矩阵的logistic误差
    # dist = 0
    #
    # dist1 = data * (np.log(A1)) + (data1 - data) * (np.log(data1 - A1))
    # for i in range(0, ItemNum2):
    #     dist += sum(dist1[:, i])
    #
    # dist = -dist / (ItemNum2 * ItemNum2)

    dist = np.linalg.norm(A1 - data)/ItemNum1         #预测矩阵与真实矩阵的欧式距离
    return dist

def jdwucha(A1,data2):
    # A1 = cala(W, data, Num1)  # 更新后的矩阵
    # A1 = np.array(A1)
    # dist = [0] * Num2
    dist_temp = np.abs(A1 - data2) / (Num1)
    dist = dist_temp.sum(axis=0)
    return dist

def xdwucha(data1,data2):                 #data1是预测的矩阵；data2是要做比较的真实矩阵


    # dist = [0] * Num2
    dist_temp = (np.abs((data1 - data2)/data2))/(Num1)
    dist = dist_temp.sum(axis=0)
    return dist

def caa(W,data):
    A2 = [0]*a
    for o in range (0,a):
        temp = np.dot(data,W)
        A2 = f(temp)


    return A2


def MSE(A1,data2):
    dist = [0] * Num2

    dist_temp = (np.power((A1 - data2),2))/Num1
    dist = dist_temp.sum(axis=0)
    # for i in range(0, Num2):
    #
    #     dist[i] = np.sum(dist_temp[:, i])
    return dist


def RMSE(M):
    dist = np.power(M,0.5)
    return dist





for i in range(0, size):


    data = data1[i:i + w, :]                 # 训练集
    ItemNum1 = data.shape[0]                       # 训练集行数
    ItemNum2 = data.shape[1]                        # 列数12
    NC = ItemNum2
    tau = np.ones((NC, NC))                         # 初始时刻各边上的信息痕迹为1

    data_test = data1[i + w - 1,:]         #测试输入
    # data_real = data1.iloc[i + w : i + w + 1,:]        #测试真实值


    start = np.random.randint(0, 11, NumAnt)           # 随机产生蚂蚁起点编号
    swucha = float('inf')                           # 用来记录当前找到的最优权重矩阵误差
    global daltatau                                           # 蚂蚁移动前各边上的信息素为0。01

    # daltatau = np.ones((N, NC * NC)) * 0.01
    daltatau = pd.DataFrame(np.ones((N, NC * NC)) * 0.01)

    for op in range(0, MAXIT):
        for j in range(0, NumAnt):                          # 考察第j只蚂蚁

            route, kwucha = path(swucha, daltatau, start[j])  # 第j只蚂蚁的路径以及误差
            if kwucha <= swucha:                           # 更新最优误差的蚂蚁路径
                bestroute = route
                swucha = kwucha
        for o in range(0, NC * NC):                  # 全局信息素的更新
            l = int(bestroute[o])
            #       print(l.type())
            daltatau.iloc[l][o] = (1 - r) * daltatau.iloc[l][o] + r / swucha

        bestw = decode(bestroute)
        W = mul(i, bestw)                                   # 最优矩阵非均匀变异
        W = np.array(bestw).reshape(NC, NC)

        mwucha = wucha(W, data)                          # 评价非均匀变异之后的最优权重
        if mwucha <= swucha:                         # 更新最优误差的蚂蚁路径
            bestroute = code(W)
            bestroute = bestroute.tolist()
            bestroute = bestroute[0]
            # print(bestroute.shape)
            # for i in range(len(bestroute)):
            #     print(bestroute[i])
            #     print(type(bestroute[i]))
            # exit()
            swucha = mwucha
        for mo in range(0, NC * NC):  # 全局信息素的更新

            d = int(bestroute[mo])
            # print(o)
            # print(d)
            # print(bestroute)
            daltatau.iloc[d][mo] = (1 - r) * daltatau.iloc[d][mo] + r / swucha
    bestw = decode(bestroute)
    bestw = np.array(bestw).reshape(NC, NC)

    # jd = [0]*ItemNum1
    # for i in range(0,Num1):
    #     jd = jdwucha(bestw,data2)
    # test = data.iloc[data.shape[0] - 1:data.shape[0], :]

    data_pred[i] = caa(bestw,data_test)



A1 = np.array(data_pred)  # 预测出的数据
Mwucha = MSE(A1, data_real)
Rwucha = RMSE(Mwucha)

jd = jdwucha(A1, data_real)
xd = xdwucha(A1, data_real)

# print(bestw)
print('绝对误差：{}'.format(jd))
print('相对误差：{}'.format(xd))
print('欧式距离：{}'.format(swucha))
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
temp.to_csv('ACO_9yue_zhaoqing.csv' , header=None)


# x = np.arange(Num1)
# for i in range (Num2):
#     y1 = data2[:,i]
#     y2 = A1[:,i]
#     plt.plot(x, y1, label='R')
#     plt.plot(x, y2, label='P')
#     plt.title(i, loc='center')
#     plt.show()
#
# x = np.arange(ItemNum2)
# for i in range (0,185,30):
#     y1 = data2[i]
#     y2 = A1[i]
#     plt.plot(x, y1, label='R')
#     plt.plot(x, y2, label='P')
#     plt.title(i, loc='center')
#     plt.show()