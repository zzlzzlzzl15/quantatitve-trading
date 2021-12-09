import numpy as np
import pandas as pd
import csv
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt


file = pd.read_csv('D:/quantum_learning/backtest_by_gong/data_for_model_multi_pct_standard.csv',chunksize= 50000, iterator= True)

class Dataset(Serializable):
