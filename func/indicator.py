#encoding:utf8
import numpy as np
import pandas as pd

#取序列的绝对值
def ABS(series):
    return np.fabs(series)

#获取N日最大值序列
def HHV(series,N):
    return pd.Series.rolling(series,N).max()

#获取N日最小值序列
def LLV(series,N):
    return pd.Series.rolling(series,N).min()

#获取N日之和序列
def SUM(series,N):
    return pd.Series.rolling(series,N).sum()

#获取标准差序列
def STD(series,N):
    return pd.Series.rolling(series,N).std()

#IF函数，s1,s2数据格式为series
def IF(cond,s1,s2):
    return pd.Series(np.where(cond,s1,s2))

#返回N之前的值序列
def REF(series,N):
    return series.shift(N)

#两个序列逐位比较，取其大者
def MAX(s1,s2):
    return np.maximum(s1,s2)

#两个序列逐位比较，取其小者
def MIN(s1,s2):
    return np.minimum(s1,s2)

#cross函数
def CROSS(s1,s2):
    con1 = np.where(REF(s1,1)<REF(s2,1),1,0)
    con2 = np.where(s1>s2,1,0)
    return pd.Series(con1*con2)
