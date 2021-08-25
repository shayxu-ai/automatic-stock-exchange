#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date: 2021/05/27 Thu
# @Author: ShayXU
# @Filename: validate.py


"""
    
"""


import baostock as bs
import pandas as pd

import datetime
from tqdm import tqdm


stock_df = pd.read_csv("moving_averages_strategy/60日均线策略.csv")
start_date = "2021-05-01"
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
print(start_date, end_date)

lg = bs.login()
if lg.error_code != '0':
    print("错误信息:", lg.error_code, lg.error_msg)

mean_close = []
for row in tqdm(stock_df.itertuples()):
    code = row[1]
    k_rs = bs.query_history_k_data_plus(code, "close", start_date, end_date)
    data_df = k_rs.get_data()
    
    mean_tmp = data_df['close'].rolling(5).mean().dropna()
    tmp = list(row[1:])
    tmp.extend(list(mean_tmp))
    mean_close.append(tmp)

# mean_close = pd.DataFrame(mean_close, columns=['code', 'code_name', 'date', 'pct_chg', 'var', 'close'])
mean_close = pd.DataFrame(mean_close)
mean_close = mean_close.dropna().reset_index(drop=True)
mean_close.to_csv("moving_averages_strategy/" + 'validate.csv', index=False)
bs.logout()