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
history_days = 120    # 历史股票天数
mean_days = 60        # x日均线
set_date = ""         # 设置日期

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
stock_df  =  stock_df[stock_df ['tradeStatus'] == '1'][~stock_df['code_name'].str.contains('ST')].reset_index(drop =  True)

data_df = pd.DataFrame()
variance = []       # 计算方差。
for row in tqdm(stock_df.itertuples()):
    code = row[1]
    code_name = row[3]

    # 检测是否收盘超过3个点
    k_rs = bs.query_history_k_data_plus(code, "close,pctChg", end_date, end_date)
    
    pct_chg = float(k_rs.get_row_data()[1])    # percentage change
    if  pct_chg >= 3:
        # print(code, pct_chg)
        k_rs = bs.query_history_k_data_plus(code, "close", start_date, end_date)
        data_df = k_rs.get_data()
        if  len(data_df) > 60:
            mean_df = data_df['close'].rolling(mean_days).mean()
            var_tmp = mean_df.dropna().var()
            close = float(data_df['close'].iloc[-1])
            if mean_df.iloc[-1] >= mean_df.iloc[-2]:
                variance.append([code, code_name, end_date, pct_chg, var_tmp, close])
        
bs.logout()
result = pd.DataFrame(variance, columns=['code', 'code_name', 'date', 'pct_chg', 'var', 'close'])
result = result.dropna().sort_values(by=['var']).reset_index(drop=True)
result.to_csv("moving_averages_strategy/" + str(mean_days) + "日均线策略.csv", index=False)
