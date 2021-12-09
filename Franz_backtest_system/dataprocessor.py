import numpy as np
import pandas as pd
import csv
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import re
import abc
import copy
import queue
import bisect
from multiprocessing import Pool
from typing import Iterable, Union
from typing import List, Union

import numpy as np
import pandas as pd
import csv
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt


class data_processing:

    def Ticker_symbol_extraction(extract_filename, chunk_number, output_filename):
        file = pd.read_csv(
            filepath_or_buffer=extract_filename,
            chunksize=chunk_number,
            iterator=True)
        Ticker_symbol_list = list(NULL)
        for chunk in file:
            for i in chunk['TICKER_SYMBOL_x']:
                if i not in Ticker_symbol_list:
                    Ticker_symbol_list.append(i)
                    print('append ' + str(i) + '.SXHE stock successfully')
        print('we have totally ', len(Ticker_symbol_list), 'stocks')
        Ticker_symbol = pd.DataFrame(Ticker_symbol_list)

        Ticker_symbol.to_csv(output_filename, index=False,
                             encoding='utf_8_sig')
        pass

    def Ticker_division(extract_file_name, ticker_file_name, chunk_num, output_file_path):



        Ticker_symbol = pd.read_csv(
            filepath_or_buffer=ticker_file_name)
        '''for extract in Ticker_symbol:
            pass'''

        file_index_extractor = pd.read_csv(
            filepath_or_buffer=extract_file_name,
            nrows=1)



        column_index_extractor = file_index_extractor
        symbol= Ticker_symbol['0']


        for ticker_name in Ticker_symbol['0']:
            file = pd.read_csv(
                filepath_or_buffer=extract_file_name,
                chunksize=chunk_num,
                iterator=True)

            save_dataframe = column_index_extractor.iloc[[0]]
            save_dataframe.drop(save_dataframe.index, inplace=True)
            for chunk in file:
                for i, x in enumerate(chunk['TICKER_SYMBOL_x']):
                    if x == ticker_name:
                        middle_data = chunk.iloc[[i]]
                        save_dataframe = save_dataframe.append(middle_data)
            save_path = output_file_path + '/' + str(ticker_name) + '.csv'
            save_dataframe.to_csv(path_or_buf=save_path, index=False,
                                  encoding='utf_8_sig')
            print('save ' + save_path + 'successfully')
            save_dataframe.drop(save_dataframe.index, inplace=True)
            pass


extract_file_name = 'D:/quantum_learning/backtest_by_gong/data_for_model_multi_pct_standard.csv'
ticker_file_name = 'D:/quantum_learning/classified_data/TICKER_SYMBOL_x.csv'
chunk_num = 30000
output_file_path = 'D:/quantum_learning/classified_data/ticker'
data_processing.Ticker_division(extract_file_name, ticker_file_name, chunk_num, output_file_path)
