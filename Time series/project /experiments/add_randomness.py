import sys
sys.path.append('./../')
from forecaster import forecaster
import numpy as np
from ABBA import ABBA
from util import myfigure
from util import dtw as DTW
from batchless_VanillaLSTM_pytorch import batchless_VanillaLSTM_pytorch
import os
import csv
import time
import matplotlib.pyplot as plt

def test_adding_randomness():

    def sMAPE(A, F):
        return 100/len(A) * np.sum(2 * np.abs(A - F) / (np.abs(A) + np.abs(F)))


    datadir = './../UCRArchive_2018'

    lag = 10
    patience = 100
    fcast_len = 50
    max_epoch = 1000
    ###############################################################################
    # Add header to csv file
    header = ['Dataset', 'Length', 'Time', 'sMAPE', 'Euclidean', 'diff_Euclidean', 'DTW', 'diff_DTW', 'DTW_reconstructed', 'diff_DTW_reconstructed', 'Euclidean_reconstructed', 'diff_Euclidean_reconstructed']

    if not os.path.isfile('./stochastic/ABBA_LSTM_results_M3.csv'):
        with open('./stochastic/ABBA_LSTM_results_M3.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    if not os.path.isfile('./stochastic/LSTM_results_M3.csv'):
        with open('./stochastic/LSTM_results_M3.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    import pandas as pd
    xls = pd.ExcelFile('./../paper/M3_competition/M3C.xls')

    # Select time series which are sampled monthly, should run from N1402 - N2829
    Monthly = xls.parse(2)

    def sMAPE(A, F):
        return 100/len(A) * np.sum(2 * np.abs(A - F) / (np.abs(A) + np.abs(F)))

    lag = 10
    patience = 100
    fcast_len = 18


    # Run through dataset
    for root, dirs, files in os.walk(datadir):
        if dirs != []:
            for dataset in dirs:
                if dataset == 'Lightning7'  or dataset =='Earthquakes' or dataset =='HouseTwenty':

                    print('Dataset:', dataset)

                    # Import time series
                    with open(datadir+'/'+dataset+'/'+dataset+'_TEST.tsv') as tsvfile:
                        tsvfile = csv.reader(tsvfile, delimiter='\t')
                        for i in range(20):
                            col = next(tsvfile)
                            ts = [float(i) for i in col]
                            if dataset == 'HouseTwenty':
                                ts = ts[:400]

    for index, row in Monthly[0:].iterrows():
        if index > 20:
            break

        # remove class information
        ts = np.array(ts[1:])
        # remove NaN from time series
        ts = ts[~np.isnan(ts)]

        # Normalise time series
        ts -= np.mean(ts)
        ts /= np.std(ts)

        train = ts[:-fcast_len]
        test = ts[-fcast_len:]

        best_hyperparameters = [10, 20, 0.05]

        # Build ABBA constructor
        abba = ABBA(tol=best_hyperparameters[2], max_k = best_hyperparameters[1] , verbose=0)
        string, centers = abba.transform(train)
        abba_numerical = abba.inverse_transform(string, centers, train[0])

        # LSTM model with ABBA and stochasticity
        t0 = time.time()
        f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=best_hyperparameters[0], cells_per_layer=20), abba=abba)
        f.train(patience=patience, max_epoch=max_epoch)
        forecast0 = f.forecast(len(test), randomize=True).tolist()
        t1 = time.time()
                                    
        with open('./stochastic/ABBA_LSTM_results_M3_sto.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            t = t1 - t0
            smape = sMAPE(test, forecast0)
            euclid = np.linalg.norm(test - forecast0)
            diff_euclid = np.linalg.norm(np.diff(test)-np.diff(forecast0))
            dtw = DTW(test, forecast0)
            diff_dtw = DTW(np.diff(test), np.diff(forecast0))
            dtw_reconstructed = DTW(abba_numerical, train)
            diff_dtw_reconstructed = DTW(np.diff(abba_numerical), np.diff(train))
            euclid_reconstructed = np.linalg.norm(abba_numerical - train)
            diff_euclid_reconstructed = np.linalg.norm(np.diff(abba_numerical)-np.diff(train))

            row = [index, len(train), t, smape, euclid, diff_euclid, dtw, diff_dtw, dtw_reconstructed, diff_dtw_reconstructed, euclid_reconstructed, diff_euclid_reconstructed]
            writer.writerow(row)
                            
        # Build ABBA constructor
        abba = ABBA(tol=best_hyperparameters[2], max_k = best_hyperparameters[1] , verbose=0)
        string, centers = abba.transform(train)
        abba_numerical = abba.inverse_transform(string, centers, train[0])

        # LSTM model with ABBA
        t0 = time.time()
        f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=best_hyperparameters[0], cells_per_layer=20), abba=abba)
        f.train(patience=patience, max_epoch=max_epoch)
        forecast1 = f.forecast(len(test)).tolist()
        t1 = time.time()
                                    
        with open('./stochastic/ABBA_LSTM_results_M3.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            t = t1 - t0
            smape = sMAPE(test, forecast1)
            euclid = np.linalg.norm(test - forecast1)
            diff_euclid = np.linalg.norm(np.diff(test)-np.diff(forecast1))
            dtw = DTW(test, forecast1)
            diff_dtw = DTW(np.diff(test), np.diff(forecast1))
            dtw_reconstructed = DTW(abba_numerical, train)
            diff_dtw_reconstructed = DTW(np.diff(abba_numerical), np.diff(train))
            euclid_reconstructed = np.linalg.norm(abba_numerical - train)
            diff_euclid_reconstructed = np.linalg.norm(np.diff(abba_numerical)-np.diff(train))

            row = [index, len(train), t, smape, euclid, diff_euclid, dtw, diff_dtw, dtw_reconstructed, diff_dtw_reconstructed, euclid_reconstructed, diff_euclid_reconstructed]
            writer.writerow(row)

                            
        best_hyperparameter = 10 

        # LSTM model without ABBA
        t0 = time.time()
        f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=best_hyperparameter, cells_per_layer=20), abba=None)
        f.train(patience=patience, max_epoch=max_epoch)
        forecast2 = f.forecast(len(test)).tolist()
        t1 = time.time()
                                    
        with open('./stochastic/LSTM_results_M3.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            t = t1 - t0
            smape = sMAPE(test, forecast2)
            euclid = np.linalg.norm(test - forecast2)
            diff_euclid = np.linalg.norm(np.diff(test)-np.diff(forecast2))
            dtw = DTW(test, forecast2)
            diff_dtw = DTW(np.diff(test), np.diff(forecast2))
            dtw_reconstructed = DTW(abba_numerical, train)
            diff_dtw_reconstructed = DTW(np.diff(abba_numerical), np.diff(train))
            euclid_reconstructed = np.linalg.norm(abba_numerical - train)
            diff_euclid_reconstructed = np.linalg.norm(np.diff(abba_numerical)-np.diff(train))


            row = [index, len(train), t, smape, euclid, diff_euclid, dtw, diff_dtw, dtw_reconstructed, diff_dtw_reconstructed, euclid_reconstructed, diff_euclid_reconstructed]
            writer.writerow(row)
                            
            # Produce plot
            fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1)
            plt.subplots_adjust(left=0.125, bottom=None, right=0.95, top=None, wspace=None, hspace=None)
            # ax1.plot(train, 'k')
            # ax1.plot(abba_numerical, 'k', alpha=0.5, label='ABBA representation')
            # ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast0), 'g', label='ABBA_LSTM_sto')
            # ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast1), 'r', label='ABBA_LSTM')
            # ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast2), 'b', label='LSTM')
            # ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(test), 'y', label='truth')
            ax1.plot(range(-1, len(test)), [train[-1]] + list(forecast0), 'g', label='ABBA_LSTM_sto')
            ax1.plot(range(-1, len(test)), [train[-1]] + list(forecast1), 'r', label='ABBA_LSTM')
            ax1.plot(range(-1, len(test)), [train[-1]] + list(forecast2), 'b', label='LSTM')
            ax1.plot(range(-1, len(test)), [train[-1]] + list(test), 'y', label='truth')
            plt.legend(loc=6)
            plt.savefig('./plots/M3_' +str(index)+'.pdf')
            plt.close()

if __name__ == '__main__':
    test_adding_randomness()