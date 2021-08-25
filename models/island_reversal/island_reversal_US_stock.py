#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date: 2021/07/08 Thu
# @Author: ShayXU
# @Filename: island_reversal_US_stock.py


"""
    
"""


import pandas as pd
import datetime
from tqdm import tqdm

import yfinance as yf


# 参数
history_days = 14    # 历史股票天数

now = datetime.datetime.now()
end_date = now.strftime('%Y-%m-%d')
start_date = (now - datetime.timedelta(days=history_days)).strftime('%Y-%m-%d')
print(start_date, end_date)

stock_df = pd.read_csv('../US_stock_symbols/美股.csv')

data_df = pd.DataFrame()
islands_reversal = []       # 
for row in tqdm(stock_df.itertuples()):
# for row in tqdm([[0, 'BABA']]):
    # Pandas(Index=0, _1='SOHU')
    code = row[1]
    data_df = yf.Ticker(code).history(start=start_date, end=end_date)
    high = data_df['High']
    low = data_df['Low']
    upward_break = 0
    downward_break = 0

    if len(data_df) >= 3:
        for i in range(len(data_df)-1):        
            if low[i+1] > high[i]:
                upward_break += 1
            if high[i+1] < low[i]:
                downward_break += 1
            
            if upward_break > 0 and downward_break > 0:
                print(code)
                islands_reversal.append([code])
                break
        
result = pd.DataFrame(islands_reversal, columns=['code'])
# result = result.reset_index(drop=True)
result.to_csv("岛型反转策略.csv", index=False)

