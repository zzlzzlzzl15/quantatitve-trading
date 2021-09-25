# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 01:30:19 2021

@author: rushy
"""
#学习tuple、？
#float,int, str->datetime
#how to get out of while loop?

import time
import requests
from datetime import  datetime, time, timedelta
from dateutil import parser #解析器
# Tab-> shift+Tab <-

page = requests.get("http://hq.sinajs.cn/?format=text&list=sh600519")
stock_info = page.text
mt_info = stock_info
list1=mt_info.split(',')



last = float(list1[1])
trade_datatime = list1[30]+" "+list1[31]

tick = (last,trade_datatime)

#tuple中的值都是不可修改的，好处：tick可能会在系统中被传来穿去，可以不被修改

order_number = 1
order1 = 'order' + str(order_number)
order_number = order_number + 1
order2 = 'order' + str(order_number)

i = 1 
while i<5:
    print("i is :",i)
    i += 1
print("job done")