import numpy as np
import datetime
import pandas as pd
from init_data import processed_data_factor_engineer
from sklearn  import  preprocessing
import matplotlib.pyplot as plt
import math


class factor_generator: #对于所有股票求取因子矩阵,这一个函数使用了未来数据进行因子计算
    def __init__(self, df_model, factor_array_save_path):
        self.df_model = df_model
        self.ticker_symbols = self.df_model['TICKER_SYMBOL_x'].unique().tolist()

        self.factors_array = pd.DataFrame(columns=self.df_model.columns, index = self.ticker_symbols)
        self.factors_array['TICKER_SYMBOL_x'] = self.ticker_symbols# 设置输出的相关系数矩阵的行名称为ticker_symbol
        self.factor_array_save_path = factor_array_save_path

        #self.ticker_file_read_path = ticker_file_read_path
        alpha_p1 = ['ACD20', 'BBIC', 'MA10Close', 'BBIC', 'BIAS10',
                    'BIAS20', 'BIAS60', 'ROC20', 'REVS5', 'REVS10', 'REVS20',
                    'REVS5Indu1', 'REVS20Indu1', 'Price1M', 'Price3M', 'RC24',
                    'Alpha20', 'RealizedVolatility', 'plusDI', 'PLRC6', 'VOL10',
                    'VOL5']  #
        alpha_p2 = ['ACD6', 'BIAS5', 'ROC6', 'SRMI', 'Volatility', 'Volumn1M',
                    'Rank1M', 'BearPower', 'BullPower', 'RC12', 'CoppockCurve',
                    'ASI', 'MACD', 'MTM', 'TRIX5', 'TRIX10', 'PLRC12',
                    'SwingIndex', 'TVSTD6', 'VOL20', 'STOM', 'PEHist20']
        alpha_p3 = [
            'CCI10', 'CCI20', 'CCI5', 'CMO', 'DBCD', 'ILLIQUIDITY',
            'REVS60', 'Volumn3M', 'AR', 'CR20', 'Elder', 'Variance20',
            'SharpeRatio20', 'InformationRatio20', 'GainVariance20',
            'GainVariance60', 'ChaikinVolatility', 'EMV14', 'minusDI',
            'DIFF', 'MTMMA', 'UOS', 'MA10RegressCoeff12',
            'MA10RegressCoeff6', 'TVMA6', 'TVSTD20', 'VEMA10', 'VEMA12',
            'VEMA5', 'VOL60', 'VROC12', 'VROC6', 'VSTD10', 'VSTD20',
            'DAVOL5', 'DAVOL10', 'STOQ', 'LCAP']
        alpha_p4 = [
            'ADTM', 'KDJ_K', 'KDJ_J', 'RSI', 'Skewness', 'MFI',
            'WVAD', 'REVS5m20', 'Price1Y', 'JDQS20', 'Alpha60',
            'SharpeRatio60', 'InformationRatio60', 'GainLossVarianceRatio20',
            'DASTD', 'HsigmaCNE5', 'AroonUp', 'DDI', 'DIZ', 'DIF',
            'VDIFF', 'VEMA26', 'VOL120', 'VR', 'DAVOL20', 'STOA',
            'PEHist60']
        alpha_p5 = [
            'FY12P',
            'EPIBS', 'InvestCashGrowRate', 'OperCashGrowRate', 'CCI88',
            'KDJ_D', 'RVI', 'HSIGMA', 'DDNSR', 'DDNCR', 'MAWVAD',
            'REVS120', 'REVS5m60', 'BR', 'ARBR', 'MassIndex', 'TOBT',
            'PSY', 'RSTR504', 'TaxRatio', 'ROECut', 'EPS',
            'OperatingProfitPSLatest', 'Variance60', 'Variance120',
            'Beta20', 'TreynorRatio20', 'TreynorRatio60', 'GainVariance120',
            'LossVariance60', 'LossVariance120', 'GainLossVarianceRatio60',
            'ChaikinOscillator', 'EMV6', 'Aroon', 'DEA', 'TVMA20',
            'VDEA', 'VMACD', 'VOL240', 'VOSC', 'MoneyFlow20', 'PEIndu',
            'PEHist120', 'NLSIZE']
        alpha_p6 = ['CTOP',
                    'CTP5', 'ACCA', 'OperCashInToAsset', 'OperatingProfitGrowRate',
                    'RetainedEarnings', 'NRProfitLoss', 'NIAPCut', 'TRevenueTTM',
                    'TCostTTM', 'RevenueTTM', 'CostTTM', 'NonOperatingNPTTM',
                    'TProfitTTM', 'NetProfitTTM', 'NetProfitAPTTM',
                    'NetInvestCFTTM', 'BollDown', 'SBM', 'STM', 'DownRVI',
                    'ChandeSD', 'ChandeSU', 'REVS750', 'PVI', 'RSTR24',
                    'RSTR12', 'ETOP', 'ROIC', 'OperatingNIToTP', 'DilutedEPS',
                    'CashEquivalentPS', 'DividendPS', 'EPSTTM', 'NetAssetPS',
                    'TORPS', 'TORPSLatest', 'OperatingRevenuePS',
                    'OperatingRevenuePSLatest', 'OperatingProfitPS',
                    'SurplusReserveFundPS', 'UndividedProfitPS', 'RetainedEarningsPS',
                    'OperCashFlowPS', 'Alpha120', 'Beta60', 'SharpeRatio120',
                    'TreynorRatio120', 'InformationRatio120', 'LossVariance20',
                    'GainLossVarianceRatio120', 'Beta252', 'AroonDown', 'Ulcer10',
                    'Ulcer5', 'DHILO', 'PE', 'PCF', 'ASSI', 'LFLO', 'TA2EV',
                    'PBIndu', 'PSIndu', 'PCFIndu', 'PEHist250', 'ForwardPE',
                    'MktValue', 'NegMktValue', 'TEAP', 'NIAP', 'CETOP']
        alpha_p7 = ['NetProfitCashCover', 'NPParentCompanyCutYOY', 'NetAssetGrowRate',
                    'TotalProfitGrowRate', 'NetProfitGrowRate', 'EARNMOM', 'EMA120',
                    'EMA20', 'EMA26', 'MA10', 'MA20', 'MA120',
                    'NetTangibleAssets', 'OperateNetIncome', 'ValueChgProfit',
                    'NPFromOperatingTTM', 'OperateProfitTTM',
                    'SaleServiceRenderCashTTM', 'NetOperateCFTTM', 'NetFinanceCFTTM',
                    'ATR6', 'BollUp', 'UpRVI', 'FiftyTwoWeekHigh', 'REVS250',
                    'NetProfitRatio', 'OperatingProfitRatio', 'NPToTOR',
                    'OperatingProfitToTOR', 'GrossIncomeRatio', 'SalesCostRatio',
                    'ROA5', 'ROE', 'ROE5', 'ROEDiluted', 'ROEWeighted',
                    'InvestRAssociatesToTPLatest', 'DividendPaidRatio',
                    'RetainedEarningRatio', 'CapitalSurplusFundPS', 'Beta120',
                    'StaticPE', 'TotalAssets']
        self.factors = alpha_p1+alpha_p2+alpha_p3+alpha_p4+alpha_p5+alpha_p6+alpha_p7
        pass

    def factor_calculation(self):
        for ticker in self.ticker_symbols:
            '''read_path = output_file_path + '/' + str(ticker) + '.csv'
            pd.read_csv
            '''
            ticker_array = self.df_model[self.df_model['TICKER_SYMBOL_x'].isin([ticker])].copy()
            for factor in self.factors:
                a = ticker_array['close_ret']
                b = ticker_array[str(factor)]
                c = ticker_array['close_ret'].corr(ticker_array[str(factor)])
                table_depth = ticker_array.shape[0]
                if table_depth > 10:
                    self.factors_array.loc[self.factors_array['TICKER_SYMBOL_x'] == ticker, str(factor)] = \
                        ticker_array['close_ret'].corr(ticker_array[str(factor)])
                else:
                    self.factors_array.loc[self.factors_array['TICKER_SYMBOL_x'] == ticker, str(factor)] = 0
            print(ticker)
        self.factors_array_save()
        return self.factors_array

    def factors_array_save(self):
        self.factors_array.to_csv(path_or_buf=self.factor_array_save_path, index=False,
                              encoding='utf_8_sig')
        print('the array has been saved successfully at' + self.factor_array_save_path)

    def __del__(self):
        print("__del__factor_generator")


class factor_pnl_signal_generator():
    def __init__(self, ticker_data, factors_table, factor_rank_table_save_path, normalization):
        pnl_cols = ['pnl_signal', 'rank']
        self.ticker_data = ticker_data
        self.factors_table = factors_table
        self.total_colunms_name = self.ticker_data.columns.tolist() + pnl_cols
        self.dft_ticker_factor_rank_table = self.ticker_data
        self.dft_ticker_factor_rank_table[pnl_cols] = 0
        self.weekiter_list = self.ticker_data['weekiter'].unique().tolist()
        self.factor_rank_table_save_path = factor_rank_table_save_path
        self.normalization = normalization
        alpha_p1 = ['ACD20', 'BBIC', 'MA10Close', 'BBIC', 'BIAS10',
                    'BIAS20', 'BIAS60', 'ROC20', 'REVS5', 'REVS10', 'REVS20',
                    'REVS5Indu1', 'REVS20Indu1', 'Price1M', 'Price3M', 'RC24',
                    'Alpha20', 'RealizedVolatility', 'plusDI', 'PLRC6', 'VOL10',
                    'VOL5']  #
        alpha_p2 = ['ACD6', 'BIAS5', 'ROC6', 'SRMI', 'Volatility', 'Volumn1M',
                    'Rank1M', 'BearPower', 'BullPower', 'RC12', 'CoppockCurve',
                    'ASI', 'MACD', 'MTM', 'TRIX5', 'TRIX10', 'PLRC12',
                    'SwingIndex', 'TVSTD6', 'VOL20', 'STOM', 'PEHist20']
        alpha_p3 = [
            'CCI10', 'CCI20', 'CCI5', 'CMO', 'DBCD', 'ILLIQUIDITY',
            'REVS60', 'Volumn3M', 'AR', 'CR20', 'Elder', 'Variance20',
            'SharpeRatio20', 'InformationRatio20', 'GainVariance20',
            'GainVariance60', 'ChaikinVolatility', 'EMV14', 'minusDI',
            'DIFF', 'MTMMA', 'UOS', 'MA10RegressCoeff12',
            'MA10RegressCoeff6', 'TVMA6', 'TVSTD20', 'VEMA10', 'VEMA12',
            'VEMA5', 'VOL60', 'VROC12', 'VROC6', 'VSTD10', 'VSTD20',
            'DAVOL5', 'DAVOL10', 'STOQ', 'LCAP']
        alpha_p4 = [
            'ADTM', 'KDJ_K', 'KDJ_J', 'RSI', 'Skewness', 'MFI',
            'WVAD', 'REVS5m20', 'Price1Y', 'JDQS20', 'Alpha60',
            'SharpeRatio60', 'InformationRatio60', 'GainLossVarianceRatio20',
            'DASTD', 'HsigmaCNE5', 'AroonUp', 'DDI', 'DIZ', 'DIF',
            'VDIFF', 'VEMA26', 'VOL120', 'VR', 'DAVOL20', 'STOA',
            'PEHist60']
        alpha_p5 = [
            'FY12P',
            'EPIBS', 'InvestCashGrowRate', 'OperCashGrowRate', 'CCI88',
            'KDJ_D', 'RVI', 'HSIGMA', 'DDNSR', 'DDNCR', 'MAWVAD',
            'REVS120', 'REVS5m60', 'BR', 'ARBR', 'MassIndex', 'TOBT',
            'PSY', 'RSTR504', 'TaxRatio', 'ROECut', 'EPS',
            'OperatingProfitPSLatest', 'Variance60', 'Variance120',
            'Beta20', 'TreynorRatio20', 'TreynorRatio60', 'GainVariance120',
            'LossVariance60', 'LossVariance120', 'GainLossVarianceRatio60',
            'ChaikinOscillator', 'EMV6', 'Aroon', 'DEA', 'TVMA20',
            'VDEA', 'VMACD', 'VOL240', 'VOSC', 'MoneyFlow20', 'PEIndu',
            'PEHist120', 'NLSIZE']
        alpha_p6 = ['CTOP',
                    'CTP5', 'ACCA', 'OperCashInToAsset', 'OperatingProfitGrowRate',
                    'RetainedEarnings', 'NRProfitLoss', 'NIAPCut', 'TRevenueTTM',
                    'TCostTTM', 'RevenueTTM', 'CostTTM', 'NonOperatingNPTTM',
                    'TProfitTTM', 'NetProfitTTM', 'NetProfitAPTTM',
                    'NetInvestCFTTM', 'BollDown', 'SBM', 'STM', 'DownRVI',
                    'ChandeSD', 'ChandeSU', 'REVS750', 'PVI', 'RSTR24',
                    'RSTR12', 'ETOP', 'ROIC', 'OperatingNIToTP', 'DilutedEPS',
                    'CashEquivalentPS', 'DividendPS', 'EPSTTM', 'NetAssetPS',
                    'TORPS', 'TORPSLatest', 'OperatingRevenuePS',
                    'OperatingRevenuePSLatest', 'OperatingProfitPS',
                    'SurplusReserveFundPS', 'UndividedProfitPS', 'RetainedEarningsPS',
                    'OperCashFlowPS', 'Alpha120', 'Beta60', 'SharpeRatio120',
                    'TreynorRatio120', 'InformationRatio120', 'LossVariance20',
                    'GainLossVarianceRatio120', 'Beta252', 'AroonDown', 'Ulcer10',
                    'Ulcer5', 'DHILO', 'PE', 'PCF', 'ASSI', 'LFLO', 'TA2EV',
                    'PBIndu', 'PSIndu', 'PCFIndu', 'PEHist250', 'ForwardPE',
                    'MktValue', 'NegMktValue', 'TEAP', 'NIAP', 'CETOP']
        alpha_p7 = ['NetProfitCashCover', 'NPParentCompanyCutYOY', 'NetAssetGrowRate',
                    'TotalProfitGrowRate', 'NetProfitGrowRate', 'EARNMOM', 'EMA120',
                    'EMA20', 'EMA26', 'MA10', 'MA20', 'MA120',
                    'NetTangibleAssets', 'OperateNetIncome', 'ValueChgProfit',
                    'NPFromOperatingTTM', 'OperateProfitTTM',
                    'SaleServiceRenderCashTTM', 'NetOperateCFTTM', 'NetFinanceCFTTM',
                    'ATR6', 'BollUp', 'UpRVI', 'FiftyTwoWeekHigh', 'REVS250',
                    'NetProfitRatio', 'OperatingProfitRatio', 'NPToTOR',
                    'OperatingProfitToTOR', 'GrossIncomeRatio', 'SalesCostRatio',
                    'ROA5', 'ROE', 'ROE5', 'ROEDiluted', 'ROEWeighted',
                    'InvestRAssociatesToTPLatest', 'DividendPaidRatio',
                    'RetainedEarningRatio', 'CapitalSurplusFundPS', 'Beta120',
                    'StaticPE', 'TotalAssets']
        self.factors = ['ValueChgProfit', 'HsigmaCNE5', 'ATR6', 'HSIGMA', 'Alpha120']
        #self.factors = alpha_p1 + alpha_p2 + alpha_p3 + alpha_p4 + alpha_p5 + alpha_p6 + alpha_p7

    def pnl_signal_generation(self):
        for index, value in enumerate(self.factors_table['TICKER_SYMBOL_x']):
            for jindex, jvalue in enumerate(self.dft_ticker_factor_rank_table['TICKER_SYMBOL_x']):
                if value == jvalue:
                    index_table = self.ticker_data.index[0:5]
                    a = self.ticker_data.iloc[jindex][self.factors]
                    b = self.factors_table.iloc[index][self.factors]
                    if self.normalization == 'std_norm':
                        a_normalized = self.min_max_normalization(a)
                    elif self.normalization == 'z_score_norm':
                        a_normalized = self.z_score_normalization(a)
                    else:
                        print('choose normalization error')
                        exit()


                    c = np.dot(a_normalized, b)
                    self.dft_ticker_factor_rank_table.loc[self.dft_ticker_factor_rank_table.index==jindex, 'pnl_signal']= c
                    # d = self.dft_ticker_factor_rank_table[jindex, 'pnl_signal']
            print(value)
        print(1)
        return self.dft_ticker_factor_rank_table

    def pnl_signal_rank_generation(self):
        self.pnl_signal_generation()
        for weekiter in self.weekiter_list:
            self.dft_ticker_factor_rank_table[self.dft_ticker_factor_rank_table['weekiter'] == weekiter]=\
                self.dft_ticker_factor_rank_table[self.dft_ticker_factor_rank_table['weekiter'] == weekiter].sort_values(
                by='pnl_signal', ascending=False)
            self.dft_ticker_factor_rank_table.loc[self.dft_ticker_factor_rank_table['weekiter'] == weekiter, 'rank'] = \
                [(i+1)/len(self.dft_ticker_factor_rank_table[self.dft_ticker_factor_rank_table['weekiter']== weekiter]) for i in \
                 range(len(self.dft_ticker_factor_rank_table[self.dft_ticker_factor_rank_table['weekiter']== weekiter]))]
            print(weekiter)
        print('The pnl signal rank table has been created successfully')
        return self.dft_ticker_factor_rank_table

    def pnl_signal_rank_table_save(self):
        self.pnl_signal_rank_generation()
        self.dft_ticker_factor_rank_table.to_csv(path_or_buf=self.factor_rank_table_save_path, index=False,
                              encoding='utf_8_sig')
        print('the array has been saved successfully at' + self.factor_rank_table_save_path)
        pass

    def min_max_normalization(self, list):
        list.apply(lambda x: (x - list.min(x)) / (list.max(x) - list.min(x)))
        return list

    def z_score_normalization(self, list):
        list_scaled = preprocessing.scale(list)
        return list_scaled


class factor_selection():
    def __init__(self, factor_data_table):
        self.factor_data_table = factor_data_table
        self.mean_std_table = pd.DataFrame(columns=['factors','mean','mean_abs','std'])
        self.selection_list = []


        alpha_p1 = ['ACD20', 'BBIC', 'MA10Close', 'BBIC', 'BIAS10',
                    'BIAS20', 'BIAS60', 'ROC20', 'REVS5', 'REVS10', 'REVS20',
                    'REVS5Indu1', 'REVS20Indu1', 'Price1M', 'Price3M', 'RC24',
                    'Alpha20', 'RealizedVolatility', 'plusDI', 'PLRC6', 'VOL10',
                    'VOL5']  #
        alpha_p2 = ['ACD6', 'BIAS5', 'ROC6', 'SRMI', 'Volatility', 'Volumn1M',
                    'Rank1M', 'BearPower', 'BullPower', 'RC12', 'CoppockCurve',
                    'ASI', 'MACD', 'MTM', 'TRIX5', 'TRIX10', 'PLRC12',
                    'SwingIndex', 'TVSTD6', 'VOL20', 'STOM', 'PEHist20']
        alpha_p3 = [
            'CCI10', 'CCI20', 'CCI5', 'CMO', 'DBCD', 'ILLIQUIDITY',
            'REVS60', 'Volumn3M', 'AR', 'CR20', 'Elder', 'Variance20',
            'SharpeRatio20', 'InformationRatio20', 'GainVariance20',
            'GainVariance60', 'ChaikinVolatility', 'EMV14', 'minusDI',
            'DIFF', 'MTMMA', 'UOS', 'MA10RegressCoeff12',
            'MA10RegressCoeff6', 'TVMA6', 'TVSTD20', 'VEMA10', 'VEMA12',
            'VEMA5', 'VOL60', 'VROC12', 'VROC6', 'VSTD10', 'VSTD20',
            'DAVOL5', 'DAVOL10', 'STOQ', 'LCAP']
        alpha_p4 = [
            'ADTM', 'KDJ_K', 'KDJ_J', 'RSI', 'Skewness', 'MFI',
            'WVAD', 'REVS5m20', 'Price1Y', 'JDQS20', 'Alpha60',
            'SharpeRatio60', 'InformationRatio60', 'GainLossVarianceRatio20',
            'DASTD', 'HsigmaCNE5', 'AroonUp', 'DDI', 'DIZ', 'DIF',
            'VDIFF', 'VEMA26', 'VOL120', 'VR', 'DAVOL20', 'STOA',
            'PEHist60']
        alpha_p5 = [
            'FY12P',
            'EPIBS', 'InvestCashGrowRate', 'OperCashGrowRate', 'CCI88',
            'KDJ_D', 'RVI', 'HSIGMA', 'DDNSR', 'DDNCR', 'MAWVAD',
            'REVS120', 'REVS5m60', 'BR', 'ARBR', 'MassIndex', 'TOBT',
            'PSY', 'RSTR504', 'TaxRatio', 'ROECut', 'EPS',
            'OperatingProfitPSLatest', 'Variance60', 'Variance120',
            'Beta20', 'TreynorRatio20', 'TreynorRatio60', 'GainVariance120',
            'LossVariance60', 'LossVariance120', 'GainLossVarianceRatio60',
            'ChaikinOscillator', 'EMV6', 'Aroon', 'DEA', 'TVMA20',
            'VDEA', 'VMACD', 'VOL240', 'VOSC', 'MoneyFlow20', 'PEIndu',
            'PEHist120', 'NLSIZE']
        alpha_p6 = ['CTOP',
                    'CTP5', 'ACCA', 'OperCashInToAsset', 'OperatingProfitGrowRate',
                    'RetainedEarnings', 'NRProfitLoss', 'NIAPCut', 'TRevenueTTM',
                    'TCostTTM', 'RevenueTTM', 'CostTTM', 'NonOperatingNPTTM',
                    'TProfitTTM', 'NetProfitTTM', 'NetProfitAPTTM',
                    'NetInvestCFTTM', 'BollDown', 'SBM', 'STM', 'DownRVI',
                    'ChandeSD', 'ChandeSU', 'REVS750', 'PVI', 'RSTR24',
                    'RSTR12', 'ETOP', 'ROIC', 'OperatingNIToTP', 'DilutedEPS',
                    'CashEquivalentPS', 'DividendPS', 'EPSTTM', 'NetAssetPS',
                    'TORPS', 'TORPSLatest', 'OperatingRevenuePS',
                    'OperatingRevenuePSLatest', 'OperatingProfitPS',
                    'SurplusReserveFundPS', 'UndividedProfitPS', 'RetainedEarningsPS',
                    'OperCashFlowPS', 'Alpha120', 'Beta60', 'SharpeRatio120',
                    'TreynorRatio120', 'InformationRatio120', 'LossVariance20',
                    'GainLossVarianceRatio120', 'Beta252', 'AroonDown', 'Ulcer10',
                    'Ulcer5', 'DHILO', 'PE', 'PCF', 'ASSI', 'LFLO', 'TA2EV',
                    'PBIndu', 'PSIndu', 'PCFIndu', 'PEHist250', 'ForwardPE',
                    'MktValue', 'NegMktValue', 'TEAP', 'NIAP', 'CETOP']
        alpha_p7 = ['NetProfitCashCover', 'NPParentCompanyCutYOY', 'NetAssetGrowRate',
                    'TotalProfitGrowRate', 'NetProfitGrowRate', 'EARNMOM', 'EMA120',
                    'EMA20', 'EMA26', 'MA10', 'MA20', 'MA120',
                    'NetTangibleAssets', 'OperateNetIncome', 'ValueChgProfit',
                    'NPFromOperatingTTM', 'OperateProfitTTM',
                    'SaleServiceRenderCashTTM', 'NetOperateCFTTM', 'NetFinanceCFTTM',
                    'ATR6', 'BollUp', 'UpRVI', 'FiftyTwoWeekHigh', 'REVS250',
                    'NetProfitRatio', 'OperatingProfitRatio', 'NPToTOR',
                    'OperatingProfitToTOR', 'GrossIncomeRatio', 'SalesCostRatio',
                    'ROA5', 'ROE', 'ROE5', 'ROEDiluted', 'ROEWeighted',
                    'InvestRAssociatesToTPLatest', 'DividendPaidRatio',
                    'RetainedEarningRatio', 'CapitalSurplusFundPS', 'Beta120',
                    'StaticPE', 'TotalAssets']
        self.factors = alpha_p1 + alpha_p2 + alpha_p3 + alpha_p4 + alpha_p5 + alpha_p6 + alpha_p7

    def mean_std_calculation(self):
        self.mean_std_table['factors'] = self.factors
        for factor in self.factors:
            self.mean_std_table.loc[self.mean_std_table['factors']==factor, 'mean'] = self.factor_data_table.loc[:, factor].mean()
            self.mean_std_table.loc[self.mean_std_table['factors']==factor, 'mean_abs'] = abs(self.factor_data_table.loc[:, factor].mean())
            self.mean_std_table.loc[self.mean_std_table['factors']==factor, 'std'] = self.factor_data_table.loc[:, factor].std()
        return self.mean_std_table




    def factors_selection_method(self):
        rank_mean_abs_table = self.mean_std_table.sort_values(
                by='mean_abs', ascending=False)
        rank_mean_abs_table = rank_mean_abs_table.iloc[:150]
        rank_std_table = self.mean_std_table.sort_values(
                by='std', ascending=True)
        rank_std_table = rank_std_table.iloc[:150]

        for rank_mean_abs in rank_mean_abs_table['factors']:
            for rank_std in rank_std_table['factors']:
                if rank_mean_abs == rank_std:
                    self.selection_list.append(rank_mean_abs)

        print('the suggested factors would be')
        print(self.selection_list)
        return self.selection_list

    def factors_selection_mean_std_table_plot(self):
        selection_factors_table = pd.DataFrame(columns=['factors','mean','mean_abs','std'])
        for factor in self.selection_list:
            a = self.mean_std_table.loc[self.mean_std_table['factors'] == factor]
            selection_factors_table.append(self.mean_std_table.loc[self.mean_std_table['factors'] == factor])

        x = selection_factors_table['mean_abs']
        y = selection_factors_table['std']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y, color='darkgreen', marker='^')

        for index, factor in enumerate(selection_factors_table['factors']):
            ax.text(x[index], y[index], factor)




















