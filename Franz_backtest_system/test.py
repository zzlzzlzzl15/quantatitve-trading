import numpy as np
import pandas as pd
import datetime
import csv
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from init_data import processed_data_factor_engineer
from factor_method import factor_generator
from factor_method import factor_pnl_signal_generator
from backtest_system import backtest_system
from factor_method import factor_selection


def main():
    Ticker_symbol_extracted_address = 'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/TICKER_SYMBOL_x.csv'
    Ticker_file_address = 'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/ticker'
    extracted_attributes = pd.read_csv(filepath_or_buffer = 'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/ticker_attributes.csv')
    start_date = 20180601
    end_date = 20200601

    data_processor = processed_data_factor_engineer(Ticker_symbol_extracted_address, Ticker_file_address, extracted_attributes,
                                                    start_date, end_date)
    processed_data = data_processor.attribute_extraction()
    #data_processor.attribute_check()
    #processed_data = data_processor.timeseries_extraction()
    processed_data.head(5)
    processed_data_save_path = 'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/processed_data.csv'
    data_processor.save_datasets_to_csv(processed_data_save_path)

#main()
def calculator():
    df_model = pd.read_csv(filepath_or_buffer = 'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/processed_data.csv')
    '''
    df_model.index = df_model['TRADE_DATE'].apply(str)
    df_model.index = [datetime.datetime.strptime(x, '%Y%m%d') for x in df_model.index]
    df_model['month'] = df_model['TRADE_DATE'].apply(lambda s: int(str(s)[4:6]))
    print(df_model.shape)
    df_model.head()
    '''
    df_model = df_model[df_model['TRADE_DATE'] > 20180601]
    df_model = df_model[df_model['TRADE_DATE'] < 20200601]
    df_model.index = df_model['TICKER_SYMBOL_x']

    factor_array_save_path = 'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/factor_table_20180601_20200601.csv'
    ticker_data_path = 'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/ticker'
    factor_calculator = factor_generator(df_model, factor_array_save_path)
    factor_array = factor_calculator.factor_calculation()
    factor_array.head(5)

#calculator()
'''这一部分用于测试当在时间区间内筛选出来的股票长度为0或者1,2的时候相关性的计算就会出现的问题
#calculator()
def datetime_extraction_test():
    start_date = 20180601
    end_date = 20200601
    ticker_data = pd.read_csv(filepath_or_buffer='D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/ticker/300836.csv')
    ticker_data.index = ticker_data['TICKER_SYMBOL_x'].apply(str)

    ticker_data = ticker_data[~np.isnan(ticker_data['close_ret'])].copy(deep=True)
    ticker_data = ticker_data[ticker_data['TRADE_DATE'] > start_date]
    ticker_data = ticker_data[ticker_data['TRADE_DATE'] < end_date]
    factor_array_save_path = 'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/ticker_300836.csv'
    ticker_data.to_csv(path_or_buf=factor_array_save_path, index=False, encoding='utf_8_sig')
    return ticker_data


def datetime_varification():
    df_model = datetime_extraction_test()

    factor_array_save_path = 'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/ticker_300836_factor_20180601_20200601.csv'
    factor_calculator = factor_generator(df_model, factor_array_save_path)
    factor_array = factor_calculator.factor_calculation()
    factor_array.head(5)
    
datetime_varification()
'''

def factor_rank_table_generation():
    ticker_data = pd.read_csv(filepath_or_buffer = \
                              'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/processed_data.csv')
    factors_table = pd.read_csv(filepath_or_buffer = \
                                'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/factor_table_20180601_20200601.csv')
    factor_rank_table_save_path = 'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/factor_rank_table_20180601_20200601.csv'
    ticker_data = ticker_data[ticker_data['TRADE_DATE'] > 20180601]
    ticker_data = ticker_data[ticker_data['TRADE_DATE'] < 20200601]
    normalization = 'z_score_norm'
    factor_pnl_signal_generation = factor_pnl_signal_generator(ticker_data, factors_table, factor_rank_table_save_path, normalization)
    factor_pnl_signal_generation.pnl_signal_rank_table_save()
    pass

#factor_rank_table_generation()

def factor_selection_method():
    factors_table = pd.read_csv(filepath_or_buffer= \
                                    'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/factor_table_20180601_20200601.csv')

    factor_selection_result = factor_selection(factors_table)
    factor_selection_result.mean_std_calculation()
    factor_selection_result.factors_selection_method()
    factor_selection_result.factors_selection_mean_std_table_plot()
    pass
#factor_selection_method()


def backtest_result():
    backtest_data = pd.read_csv(filepath_or_buffer = \
                              'D:/quantum_learning/Franz_backtest_system/franz_system/classified_data/factor_rank_table_20180601_20200601.csv')
    coly = 'close_ret'
    backtest = backtest_system(backtest_data, coly)
    backtest.factor_signal_histogram_distribution()
    backtest.rank_histogram_distribution()
    backtest.backtest_parameter_and_return()

backtest_result()

