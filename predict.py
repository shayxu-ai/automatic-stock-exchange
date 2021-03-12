#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date: 2021/03/11 Thu
# @Author: ShayXU
# @Filename: predict.py


import os
import numpy as np
import pandas as pd

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


def train():
        # model = keras.models.Sequential()
        # model.add(keras.layers.Conv2D(32, 3, activation='relu', input_shape=(20, 20, 1)))         # 卷积核的个数 => 输出的维度
        # model.add(keras.layers.MaxPooling2D((2, 2)))
        # model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(keras.layers.MaxPooling2D((2, 2)))
        # model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

        # model.add(keras.layers.Flatten())
        # model.add(keras.layers.Dense(64, activation='relu'))
        # model.add(keras.layers.Dense(2))
        # model.compile(optimizer='adam', 
        #         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #         metrics=['accuracy'])
        model = tf.keras.models.load_model('saved_model.h5')

        model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
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
        0: "跌",
        1: "涨"
    }
    print("Price: {}".format(stock['close'].values[-1]))
    print(
        "Stock {} most likely {} with a {:.2f} percent confidence."
        .format(stock_code, class_names[np.argmax(score)], 100 * np.max(score))
    )


if __name__ == '__main__':
    stock_code = "002738.sz"        # 股票代码
    to_date = '2021-03-11'          # 今日日期
    re_download = False             # 重新下载
    re_train = True                 # 是否训练
    predict_period = 20             # 预测天数
    history_period = 400            # 分析天数

    start_date = '2010-01-01'
    
    stock_info_path = "stock_info/" + stock_code + ".csv"
    if not os.path.exists(stock_info_path) or re_download:
        # 从股票宝下载
        bs.login()
        rs = bs.query_history_k_data(stock_code, "date, code, open, high, low, close, preclose, volume, amount, adjustflag, turn", start_date=start_date, end_date=to_date, frequency="d", adjustflag="3")

        data_list = []
        while (rs.error_code == '0') & rs.next():  # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        result.to_csv(stock_info_path, index=False)
        bs.logout()

    # 读取csv文件
    stock = pd.read_csv(stock_info_path)

    # 准备数据
    stock['close_nomalized'] = (stock['close']-stock['close'].min())/(stock['close'].max()-stock['close'].min())
    stock['future_price'] = stock['close_nomalized'].rolling(predict_period).mean()
    
    n = len(stock)
    x = np.array([stock['close_nomalized'][i:i+history_period] for i in range(n-history_period+1)])[20:].reshape(-1, 20, 20)
    x = x[:, :, :, np.newaxis]

    open_ = ((stock['open']-stock['close'].min())/(stock['close'].max()-stock['close'].min())).values[history_period-1:-predict_period]
    y = stock['future_price'][history_period-1:].values[:-predict_period]
    y[y - open_ > 0] = 1
    y[y - open_ <= 0] = 0

    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,shuffle=True)

    if re_train:
        train()

    predict()


