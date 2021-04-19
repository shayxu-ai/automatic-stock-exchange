# automatic-stock-exchange
人工智障。


|项目进度|进度|说明|
| :-: | :-: | :-: |
| 训练买入模型 | Done | 没人能够预测市场 |
| 训练卖出模型 | not start |预测跌就可以卖|
| 自动交易 | not start ||

这个股票信息api可以 http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5 (证券宝)

## 所以什么是股价
1. 价格实际就是上一笔交易的成交价格
2. A股不允许下市价单(market order)只可以下限价单(limit order)，当买入限价单价格高于现价也就相当于是市价单了
3. 限价单/市价单/条件单 https://www.528btc.com/tk/158591349462418.html
4. 集合竞价的交易量最大原则 => 开盘价 https://www.zhihu.com/question/19805529/answer/157315243
5. 连续竞价”，直至收盘。
6. stock dividends 股息
7. stock splits 股票拆分成更多的小份
8. adj close 调整后的收盘价，计算股息和股票拆分后的收盘价
9. 股票代码
000: 代表深市A股。
002: 代表中小板。
300: 代表创业板。
600: 代表沪市A股，其中601或603也代表沪市A股。
10. 

## 求打赏[支付宝]
<img src="https://github.com/shayxu-ai/shayxu-ai.github.io/blob/master/images/alipay.jpg?raw=true" width="150">