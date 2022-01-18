import numpy as np
import datetime
import pandas as pd

class backtest_system():
    def __init__(self, backtest_data,coly):
        self.backtest_data = backtest_data
        self.coly = coly

    def factor_signal_histogram_distribution(self):
        self.backtest_data['pnl_signal'].hist(bins=100)

    def rank_histogram_distribution(self):
        self.backtest_data['rank'].hist(bins=100)

    def backtest_parameter_and_return(self):

        #################
        df_p = self.backtest_data[(self.backtest_data['rank'] < 0.1)].copy(deep=True).reset_index(drop=True)[
            ['TRADE_DATE', 'TICKER_SYMBOL', self.coly]]  # 提取预测值排序前10%的数据
        df_p['TICKER_SYMBOL'] = df_p['TICKER_SYMBOL'].apply(
            lambda s: str(s) if len(str(s)) == 6 else '0' * (6 - len(str(s))) + str(s))
        df_p['cnt'] = 1
        df_pro = df_p.groupby('TRADE_DATE')['cnt', self.coly].sum().reset_index()  # 变成收益的百分之多少
        # 平均买入50只股票，买入1000万
        df_pro['ret'] = round(df_pro[self.coly] / df_pro['cnt'], 5) * 1000  # 本金交易后变成了多少
        df_pro['ret'] = df_pro['ret'] - 1000 * 0.002  # 扣除手续费后本金变为多少
        df_pro.index = df_pro['TRADE_DATE'].apply(str)
        df_pro.index = [datetime.datetime.strptime(x, '%Y%m%d') for x in df_pro.index]

        df_sharp = df_pro['ret']
        df_sharp.columns = (['pro'])  # 添加列名pro
        returns = [x / abs(10000000) for x in df_sharp.values]  # 计算回报率
        sharp = np.average(returns) * 250 / (np.std(returns) * np.sqrt(250))  # 夏普率 回报风险比
        sharp = round(sharp, 3)
        # 胜率 win_rate
        win_rate = round(sum(df_sharp > 0) / sum(df_sharp != 0), 3) if sum(df_sharp != 0) != 0 else 0
        # 盈亏比 wl_rate
        win_pro = int(sum(df_sharp[df_sharp > 0]) / sum(df_sharp > 0)) if sum(df_sharp > 0) else 0
        loss_pro = abs(int(sum(df_sharp[df_sharp < 0]) / sum(df_sharp < 0))) if sum(df_sharp < 0) else 0
        wl_rate = round(win_pro / loss_pro, 3) if loss_pro != 0 else 0
        print("夏普：{}  #  胜率：{} # 盈亏比：{}".format(sharp, win_rate, wl_rate))
        x = np.array(returns).cumsum()
        i = np.argmax(np.maximum.accumulate(x) - x)
        j = np.argmax(x[:i])
        max_dd = (x[j] - x[i]) / x[j]
        max_dd = round(max_dd, 3)
        print("最大回撤:", max_dd, '手续费：千2')
        df_pro['ret'].cumsum().plot(figsize=(12, 6))