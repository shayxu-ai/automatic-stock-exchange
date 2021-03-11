#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date: 2021/03/09 Tue
# @Author: ShayXU
# @Filename: stock_prediction.py


"""
    1、先把流程走通
    2、对参数进行网格搜索。
        开始结束时间
        周期
        参数
    3、保存参数
    
"""

import baostock as bs
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

def preprocess():
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+ lg.error_code)
    print('login respond  error_msg:'+ lg.error_msg)

    rs = bs.query_history_k_data("600100", "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST", start_date='2010-01-01', end_date='2021-03-09', frequency="d", adjustflag="3")
    print('query_history_k_data respond error_code:'+rs.error_code)
    print('query_history_k_data respond  error_msg:'+rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():  # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    print(result)

    # #### 结果集输出到csv文件 ####   
    # result.to_csv("history_A_stock_k_data.csv", index=False)
    # print(result)

    #### 登出系统 ####
    bs.logout()


def train_model():
    # 读取图片 tf.data.Dataset 默认双线性插值 bilinear
    train_ds = keras.preprocessing.image_dataset_from_directory(
        "img_output/", image_size=(40, 40), subset="training", validation_split=0.2,
        seed=123
    )

    val_ds  = keras.preprocessing.image_dataset_from_directory(
        "img_output/", labels='inferred', image_size=(40, 40), subset="validation", validation_split=0.2,
        seed=123
    )
    # train_ds.class_names

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(40, 40, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(len(train_ds.class_names)))

    # model.summary()

    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=1)
    tf.saved_model.save(model, 'saved_model/')
    # pretrained_model = tf.saved_model.load('saved_model/')
