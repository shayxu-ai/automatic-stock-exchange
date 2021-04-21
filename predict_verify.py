#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date: 2021/03/11 Thu
# @Author: ShayXU
# @Filename: predict.py


"""
    0、重构下代码，优化结构
    
    0.1 *pandas rolling聚合 是否有信息泄露 （大概不会吧）

    1、结果不准确 =》 再加特征，多维CNN
        # volume 成交量
        # amount 成交额
        # turn 换手率

    2、结果不稳定 =》 多次训练。

    3、使用tensorboard进行超参优化

    4、模型集成

    5、模型量化
"""


import os

import datetime
from tqdm import tqdm

import csv
import numpy as np
import pandas as pd
import matplotlib. pyplot as plt 

import baostock as bs   # 股票宝，获取股票数据

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Reshape,Dropout,Activation
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Conv1D,MaxPooling1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping


def download():
    # 从股票宝下载股票数据
    bs.login()
    for stock_code in tqdm(stock_code_list):
        stock_info_path = "stock_info/" + stock_code + ".csv"
        if not os.path.exists(stock_info_path) or re_download:
            rs = bs.query_history_k_data(stock_code, "date, open, close, volume, amount, turn, pctChg", start_date=start_date, end_date=to_date, frequency="d", adjustflag="3")
            # volume 成交量
            # amount 成交额
            # turn 换手率

            data_list = []
            while (rs.error_code == '0') & rs.next():  # 获取一条记录，将记录合并在一起
                data_list.append(rs.get_row_data())
            result = pd.DataFrame(data_list, columns=rs.fields)
            result.to_csv(stock_info_path, index=False)
    bs.logout()

def preprocess():
    stock_info_path = "stock_info/" + stock_code + ".csv"       # 文件路径
    # 读取csv文件
    stock = pd.read_csv(stock_info_path, parse_dates=['date'])
    
    if i == 0:
        pass
    else:
        stock = stock[:-i]

    # 准备数据
    stock['close_nomalized'] = (stock['close']-stock['close'].min())/(stock['close'].max()-stock['close'].min())        # 收盘价 归一化
    stock['volume_nomalized'] = (stock['volume']-stock['volume'].min())/(stock['volume'].max()-stock['volume'].min())   # 交易量 归一化
    stock['avg_price'] = stock['close'].rolling(predict_period).mean()                                                  # 最近周期内的平均股价
    stock = stock[predict_period-1:]
    stock['future_price'] = stock['close'].rolling(predict_period).mean().shift(-predict_period)                        # 未来股价均值(不包含当日收盘价)
    # stock = stock.dropna(axis=0)                                                                                        # 去除空值

    def flat_or_not(x):
        if x >= threshold_flat:
            return 2       # 涨
        elif x <= -threshold_flat:
            return 1       # 跌
        else:
            return 0       # 持平

    stock['label'] = ((stock['future_price'] - stock['avg_price']) / stock['avg_price']).apply(flat_or_not)

    n = len(stock)

    if not cnn_3d_flag:
        x = np.array([stock['close_nomalized'][i:i+history_period] for i in range(n-history_period-predict_period+1)]).reshape(-1, 20, 20) # 输入 400天 （0:400）~（n-400-predict~n）
        x = x[:, :, :, np.newaxis]
    else:
        x = np.array([stock[['close_nomalized', 'volume_nomalized']][i:i+history_period] for i in range(n-history_period-predict_period+1)]).reshape(-1, 20, 20, 2) # 输入 400天 + 交易量
        x = x[:, :, :, :, np.newaxis]

    y = stock['label'][history_period-1:].values[:-predict_period]                                               # 标签 

    print(pd.DataFrame(y)[0].value_counts())    # 打印三种类别样本的个数。
    return stock, x, y

def train():
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, 3, activation='relu', input_shape=(20, 20, 1)))         # 卷积核的个数 => 输出的维度
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(3))
        model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        # model = tf.keras.models.load_model('saved_model.h5')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=6, verbose=1, mode='auto')

        model.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test), callbacks = [monitor])
        # tf.saved_model.save(model, 'saved_model/')
        model.save('saved_model.h5')

def train_3d():
        model = keras.models.Sequential()
        model.add(keras.layers.Conv3D(32, (3, 3, 1), activation='relu', input_shape=(20, 20, 2, 1)))         # 卷积核的个数 => 输出的维度
        model.add(keras.layers.MaxPool3D((2, 2, 1)))
        model.add(keras.layers.Conv3D(64, (3, 3, 1), activation='relu'))
        model.add(keras.layers.MaxPool3D((2, 2, 1)))
        model.add(keras.layers.Conv3D(64, (3, 3, 1), activation='relu'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(3))
        model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        # model = tf.keras.models.load_model('saved_model.h5')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=6, verbose=1, mode='auto')

        model.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test), callbacks = [monitor])
        # tf.saved_model.save(model, 'saved_model/')
        model.save('saved_model.h5')


def predict():
    """
        preprocess中已经根据i，缩短了stock从而x,y都无需额外处理
    """
    # 读取模型
    model = tf.keras.models.load_model('saved_model.h5')
    # model = tf.saved_model.load('saved_model/')
    xi = tf.convert_to_tensor(x[[-1]], tf.float32, name='inputs')
    predictions = model(xi)
    score = tf.nn.softmax(predictions[0])
    class_names = {
        0: "持平",
        1: "跌",
        2: "涨"
    }
    print("Price: {}".format(stock['close'].values[-1]))
    print(
        "Stock {} most likely {} with a {:.2f} percent confidence."
        .format(stock_code, class_names[np.argmax(score)], 100 * np.max(score))
    )
    with open("predict_output.csv", 'a', newline='') as f:
        tmp = list(stock.loc[-1])
        for item in [class_names[np.argmax(score)], 100 * np.max(score)]:
            tmp.append(item)
        csv.writer(f).writerow(tmp)
    return [stock['close'].values[-1], np.argmax(score)]



if __name__ == '__main__':
    # 清空验证结果文件
    with open("predict_output.csv", 'w') as f:
        pass

    # 一些超参
    to_date = datetime.datetime.now().strftime("%Y-%m-%d")      # 今日日期       
    re_download = False              # 重新下载数据
    re_train = False                 # 重新训练
    predict_period = 6             # 预测天数
    history_period = 400            # 分析天数
    epoch = 200
    start_date = '2010-01-01'       # 最早数据
    threshold_flat = 0.007           # 股价持平的阈值
    stock_code_list = pd.read_csv('stock_codes.csv')['code']    # 获取需要预测的股票代码

    # 一些设置
    verify_period = 90              # 验证周期
    simulation = {
        'bought': False,
        'price': 0,
        'asset': 1

    }

    cnn_3d_flag = True

    download()

    for i in range(verify_period, -1, -1):              # 验证天数， n ~ 0
        for stock_code in tqdm(stock_code_list):        # 股票代码

            stock, x, y = preprocess()
        
            try:
                t = 1           # 模型集成
                while t > 0:
                    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,shuffle=True)
                    if re_train:
                        if not cnn_3d_flag:
                            train()
                        else:
                            train_3d()
                    close, trend = predict()
                    if not simulation['bought'] and trend == 1:
                        simulation['bought'] = True
                        simulation['price'] = close
                    elif simulation['bought'] and trend == 2:
                        simulation['bought'] = False
                        simulation['asset'] = simulation['asset']/simulation['price']*close
                    t -= 1

            except Exception as e:
                with open("predict_output.csv", 'a') as f:
                    wri = csv.writer(f)
                    wri.writerow([stock_code, e, e.__traceback__.tb_lineno])
            
            print(simulation['asset'])

        


