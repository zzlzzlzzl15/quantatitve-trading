import numpy as np
import pandas as pd
import datetime
import csv
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt


#file = pd.read_csv('D:/quantum_learning/backtest_by_gong/data_for_model_multi_pct_standard.csv',chunksize= 50000, iterator= True)

class processed_data_factor_engineer():
    def __init__(self, Ticker_symbol_extracted_address, Ticker_file_address, extracted_attributes, start_date, end_date):
        self.Ticker_symbol_extracted_address = Ticker_symbol_extracted_address #获取股票名称地址
        self.Ticker_file_address = Ticker_file_address#股票数据的地址
        self.Ticker_symbol = pd.read_csv(
            filepath_or_buffer = self.Ticker_symbol_extracted_address, header = 0)#股票的参数，list类型

        self.extracted_attributes = extracted_attributes#手动设置需要提取出数据的属性

        self.attributes = pd.read_csv(filepath_or_buffer='D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/ticker_attributes.csv',
                                       nrows=1)
        factor_names = pd.read_csv(filepath_or_buffer='D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/data_for_model_multi_pct_standard.csv', nrows = 1)
        self.dataset = factor_names.iloc[[0]]
        self.dataset.drop(self.dataset.index, inplace=True)
        self.df_model = self.dataset

        self.start_date = start_date
        self.end_date = end_date

    def ticker_symbols_extraction(self):

        return self.Ticker_symbol

    def attribute_extraction(self):
        for ticker in self.Ticker_symbol['0']:
            read_path = self.Ticker_file_address + '/' + str(ticker) + '.csv'
            ticker_data = pd.read_csv(filepath_or_buffer=read_path, usecols=self.extracted_attributes)
            ticker_data.index = ticker_data['TICKER_SYMBOL_x'].apply(str)
            print(ticker)
            '''
            if ticker == 1:
                self.dataset = ticker_data
            else:
                pd.concat([self.dataset, ticker_data])
            '''
            ticker_data = self.timeseries_extraction(ticker_data)
            self.dataset = self.dataset.append(ticker_data)
        return self.dataset

    def attribute_check(self):
        for extracted_attribute in self.extracted_attributes:
            if extracted_attribute not in self.attributes:
                print(extracted_attribute+' is not a useful attribute.\n')
            print('all the attribute is inside the table')

    def close_ret_extraction(self):
        self.df_model = self.dataset[~np.isnan(self.dataset['close_ret'])].copy(deep=True)
        print('the closed ticker has been removed from all the tickers ')
        return self.df_model.head(5)



    def timeseries_extraction(self, ticker_data):
        #self.dataset.index = self.dataset['TRADE_DATE'].apply(str)
        #self.dataset.index = [datetime.datetime.strptime(x, '%Y%m%d') for x in self.dataset.index]
        #self.dataset.head(5)
        ticker_data = ticker_data[~np.isnan(ticker_data['close_ret'])].copy(deep=True)
        ticker_data = ticker_data[ticker_data['TRADE_DATE'] > self.start_date]
        ticker_data = ticker_data[ticker_data['TRADE_DATE'] < self.end_date]
        return ticker_data

    def save_datasets_to_csv(self, save_path):
        self.dataset.to_csv(path_or_buf=save_path, index=False, encoding='utf_8_sig')


'''
class processed_data_machine_learning:
    def __init__(self, Ticker_data_address):
        self.Ticker_data_address = Ticker_data_address
        self.dataset = self.dataset = pd.read_csv(filepath_or_buffer=self.Ticker_data_address)
'''


#Ticker_symbol_extracted_address = 'D:/quantum_learning/classified_data/TICKER_SYMBOL_x_new_example'
#extracted_attributes = ['TRADE_DATE', 'CLOSE_PRICE_1','OPEN_PRICE_1','HIGHEST_PRICE_1','LOWEST_PRICE_1','VWAP','TICKER_SYMBOL_x']
