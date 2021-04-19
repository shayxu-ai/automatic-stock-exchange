#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date: 2021/03/11 Thu
# @Author: ShayXU
# @Filename: predict.py


"""
    1、结果不准确 =》 再加特征，多维CNN
        # volume 成交量
        # amount 成交额
        # turn 换手率

    2、结果不稳定 =》 多次训练。

    3、使用tensorboard进行超参优化

"""

from genericpath import exists
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import datetime

import baostock as bs
import matplotlib. pyplot as plt 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Reshape,Dropout,Activation
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Conv1D,MaxPooling1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping


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
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

        model.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test), initial_epoch=10, callbacks = [monitor])
        # tf.saved_model.save(model, 'saved_model/')
        model.save('saved_model.h5')


def predict():
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
        wri = csv.writer(f)
        wri.writerow([stock_code, stock['date'].values[-1], stock['close'].values[-1], class_names[np.argmax(score)], 100 * np.max(score)])



if __name__ == '__main__':

    verify_period = 20      # 验证周期，自然日
    now = datetime.datetime.now()
    now = now - datetime.timedelta(days=verify_period)

    stock_code_list = pd.read_csv('stock_codes.csv')['code']

    with open("predict_output.csv", 'w') as f:
        wri = csv.writer(f)

    
    to_date = datetime.datetime.now().strftime("%Y-%m-%d")      # 今日日期       
    re_download = True              # 重新下载
    re_train = True                 # 是否训练
    predict_period = 20             # 预测天数
    history_period = 400            # 分析天数
    epoch = 200
    start_date = '2010-01-01'       # 最早数据

    # 从股票宝下载
    bs.login()
    print("数据下载")
    for stock_code in tqdm(stock_code_list):
        stock_info_path = "stock_info/" + stock_code + ".csv"
        if not os.path.exists(stock_info_path) or re_download:
            rs = bs.query_history_k_data(stock_code, "date, open, close, volume, amount, turn", start_date=start_date, end_date=to_date, frequency="d", adjustflag="3")
            # volume 成交量
            # amount 成交额
            # turn 换手率

            data_list = []
            while (rs.error_code == '0') & rs.next():  # 获取一条记录，将记录合并在一起
                data_list.append(rs.get_row_data())
            result = pd.DataFrame(data_list, columns=rs.fields)
            result.to_csv(stock_info_path, index=False)
    bs.logout()

    # 预测
    for i in range(verify_period, -1, -1):
        for stock_code in tqdm(stock_code_list):
            stock_info_path = "stock_info/" + stock_code + ".csv"       # 文件路径
            # 读取csv文件
            stock = pd.read_csv(stock_info_path, parse_dates=['date'])
            
            if i == 0:
                pass
            else:
                stock = stock[:-i]

            # 准备数据
            stock['close_nomalized'] = (stock['close']-stock['close'].min())/(stock['close'].max()-stock['close'].min())
            stock['future_price'] = stock['close_nomalized'].rolling(predict_period).mean()
            
            n = len(stock)
            x = np.array([stock['close_nomalized'][i:i+history_period] for i in range(n-history_period+1)])[20:].reshape(-1, 20, 20)
            x = x[:, :, :, np.newaxis]

            open_ = ((stock['open']-stock['close'].min())/(stock['close'].max()-stock['close'].min())).values[history_period-1:-predict_period]
            y = stock['future_price'][history_period-1:].values[:-predict_period]

            y_decrase = y - open_ <= -0.01
            y_increase = y - open_ >= 0.01
            y_equal = np.logical_not(y_decrase+y_increase)
            y[y_increase] = 2
            y[y_decrase] = 1
            y[y_equal] = 0
            print(pd.DataFrame(y)[0].value_counts())
            
            try:
                t = 10
                while t > 0:
                    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,shuffle=True)
                    if re_train:
                        train()
                    predict()
                    t -= 1

            except Exception as e:
                with open("predict_output.csv", 'a') as f:
                    wri = csv.writer(f)
                    wri.writerow([stock_code, e, e.__traceback__.tb_lineno])

        


