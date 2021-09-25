#stiring
#float, int ,str->datatime?
#tuple
#how to get out of while loop
import time as TIME
import requests
from datetime import  datetime, time, timedelta
from dateutil import parser #解析器

def getTick():  #def function
    #go to sina to get last tick info
    page = requests.get("http://hq.sinajs.cn/?format=text&list=sh600519")
    stock_info = page.text
    mt_info = stock_info
    list1=mt_info.split(',')



    last = float(list1[1])
    trade_datatime = list1[30]+" "+list1[31]

    tick = (last,trade_datatime)
    
    return tick

trade_time = time(9,30)
while time(9)<trade_time<time(15):
    last_tick = getTick()
    dt = parser.parse(last_tick[1]).time()
    print(last_tick)
    trade_time = parser.parse(last_tick[1]).time()
    #wait for 1 second
    TIME.sleep(3)
print("job done")