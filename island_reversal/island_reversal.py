#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date: 2021/05/27 Thu
# @Author: ShayXU
# @Filename: 60日均线策略.py


"""
    运行时间: ~01:20, min
"""

import baostock as bs
import pandas as pd

import datetime
from tqdm import tqdm

# 参数
history_days = 7    # 历史股票天数
set_date = "2021-05-27"         # 设置日期

# 计算日期
if not set_date:
    now = datetime.datetime.now()
    if now.hour <= 17 and now.minute <= 30:
        now -= datetime.timedelta(days=1)
else:
    now = datetime.datetime.strptime(set_date, '%Y-%m-%d')


end_date = now.strftime('%Y-%m-%d')
start_date = (now - datetime.timedelta(days=history_days)).strftime('%Y-%m-%d')
print(start_date, end_date)

# 登陆系统
lg = bs.login()
if lg.error_code != '0':
    print("错误信息:", lg.error_code, lg.error_msg)

# 查询
data_list = []
stock_rs = bs.query_all_stock(end_date)     # 查询全量股票，含指数
stock_df  = stock_rs.get_data()
stock_df  =  stock_df [stock_df ['tradeStatus'] == '1'].reset_index(drop =  True)

data_df = pd.DataFrame()
islands_reversal = []       # 计算方差。
for row in tqdm(stock_df.itertuples()):
    code = row[1]
    code_name = row[3]
    k_rs = bs.query_history_k_data_plus(code, "high,low,date,close", start_date, end_date)
    data_df = k_rs.get_data()
    data_types_dict = {'high': float, 'low': float}
    data_df = data_df.astype(data_types_dict)

    # print(code, data_df.iloc[-2][0] - data_df.iloc[-3][1], data_df.iloc[-2][0] - data_df.iloc[-1][1])
    if len(data_df) >= 3 and data_df.iloc[-2][0] < data_df.iloc[-3][1] and data_df.iloc[-2][0] < data_df.iloc[-1][1]:
        date = data_df.iloc[-1][2]
        close = data_df.iloc[-1][3]
        islands_reversal.append([code, code_name, date, close])
        
bs.logout()
result = pd.DataFrame(islands_reversal, columns=['code', 'code_name', 'date', 'close'])
# result = result.reset_index(drop=True)
result.to_csv("island_reversal/岛型反转策略.csv", index=False)