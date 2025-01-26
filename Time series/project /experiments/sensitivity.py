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

def test_sensitivity():

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

    if not os.path.isfile('./sensitivity/ABBA_LSTM_results_Lightning7.csv'):
        with open('./sensitivity/ABBA_LSTM_results_Lightning7.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    if not os.path.isfile('./sensitivity/LSTM_results_Lightning7.csv'):
        with open('./sensitivity/LSTM_results_Lightning7.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    if not os.path.isfile('./sensitivity/ABBA_LSTM_results_Earthquakes.csv'):
        with open('./sensitivity/ABBA_LSTM_results_Earthquakes.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    if not os.path.isfile('./sensitivity/LSTM_results_Earthquakes.csv'):
        with open('./sensitivity/LSTM_results_Earthquakes.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    if not os.path.isfile('./sensitivity/ABBA_LSTM_results_HouseTwenty.csv'):
        with open('./sensitivity/ABBA_LSTM_results_HouseTwenty.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    if not os.path.isfile('./sensitivity/LSTM_results_HouseTwenty.csv'):
        with open('./sensitivity/LSTM_results_HouseTwenty.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    ###############################################################################
    # Run through dataset
    for root, dirs, files in os.walk(datadir):
        if dirs != []:
            for dataset in dirs:
                if dataset == 'Lightning7'  or dataset =='Earthquakes' or dataset =='HouseTwenty':
                    try:
                        print('Dataset:', dataset)

                        # Import time series
                        with open(datadir+'/'+dataset+'/'+dataset+'_TEST.tsv') as tsvfile:
                            tsvfile = csv.reader(tsvfile, delimiter='\t')
                            for i in range(20):
                                col = next(tsvfile)
                                ts = [float(i) for i in col]
                                if dataset == 'HouseTwenty':
                                    ts = ts[:400]
                                print('i', i)

                                # remove class information
                                ts = np.array(ts[1:])
                                # remove NaN from time series
                                ts = ts[~np.isnan(ts)]

                                # Normalise time series
                                ts -= np.mean(ts)
                                ts /= np.std(ts)

                                train = ts[:-fcast_len]
                                test = ts[-fcast_len:]

                                # GRID SEARCH 
                                if i == 0:
                                    min_smape = np.inf

                                    hyperparameters = {}
                                    hyperparameters['lag'] = [5, 10, 15]
                                    hyperparameters['max_k'] = [15, 20, 25]
                                    hyperparameters['tol'] = [0.01, 0.03, 0.05]

                                    for lag in hyperparameters['lag']:
                                        for max_k in hyperparameters['max_k']:
                                            for tol in hyperparameters['tol']:

                                                # Build ABBA constructor
                                                abba = ABBA(tol=tol, max_k = max_k, verbose=0)
                                                string, centers = abba.transform(train)
                                                abba_numerical = abba.inverse_transform(string, centers, train[0])
                                                
                                                # LSTM model with ABBA
                                                t0 = time.time()
                                                f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=lag), abba=abba)
                                                f.train(patience=patience, max_epoch=max_epoch)
                                                forecast1 = f.forecast(len(test)).tolist()
                                                t1 = time.time()
                                        
                                                smape = sMAPE(test, forecast1)

                                                if smape < min_smape:
                                                    min_smape = smape
                                                    best_hyperparameters = [lag, max_k, tol]

                                    print('best hyperparameters', best_hyperparameters)
                                    # Build ABBA constructor
                                    abba = ABBA(tol=best_hyperparameters[2], max_k =best_hyperparameters[1] , verbose=0)
                                    string, centers = abba.transform(train)
                                    abba_numerical = abba.inverse_transform(string, centers, train[0])

                                    # LSTM model with ABBA
                                    t0 = time.time()
                                    f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=best_hyperparameters[0]), abba=abba)
                                    f.train(patience=patience, max_epoch=max_epoch)
                                    forecast1 = f.forecast(len(test)).tolist()
                                    t1 = time.time()
                                        
                                    with open('./sensitivity/ABBA_LSTM_results_' + dataset+'.csv', 'a', newline='') as csvfile:
                                        writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
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

                                        row = [i, len(train), t, smape, euclid, diff_euclid, dtw, diff_dtw, dtw_reconstructed, diff_dtw_reconstructed, euclid_reconstructed, diff_euclid_reconstructed]
                                        writer.writerow(row)

                                
                                else:
                                    abba = ABBA(tol=best_hyperparameters[2], max_k =best_hyperparameters[1] , verbose=0)
                                    string, centers = abba.transform(train)
                                    abba_numerical = abba.inverse_transform(string, centers, train[0])

                                    # LSTM model with ABBA
                                    t0 = time.time()
                                    f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=best_hyperparameters[0]), abba=abba)
                                    f.train(patience=patience, max_epoch=max_epoch)
                                    forecast1 = f.forecast(len(test)).tolist()
                                    t1 = time.time()
                                        
                                    with open('./sensitivity/ABBA_LSTM_results_' + dataset+'.csv', 'a', newline='') as csvfile:
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

                                        row = [i, len(train), t, smape, euclid, diff_euclid, dtw, diff_dtw, dtw_reconstructed, diff_dtw_reconstructed, euclid_reconstructed, diff_euclid_reconstructed]
                                        writer.writerow(row)

                                # GRID SEARCH without ABBA 
                                if i == 0:
                                    min_smape = np.inf

                                    hyperparameters = {}
                                    hyperparameters['lag'] = [5, 10, 20, 30]

                                    for lag in hyperparameters['lag']:
                
                                        # LSTM model with ABBA
                                        t0 = time.time()
                                        f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=lag), abba=None)
                                        f.train(patience=patience, max_epoch=max_epoch)
                                        forecast2 = f.forecast(len(test)).tolist()
                                        t1 = time.time()
                                        
                                        smape = sMAPE(test, forecast2)
                                        if smape < min_smape:
                                            min_smape = smape
                                            best_hyperparameter = lag

                                    print('best hyperparameter', best_hyperparameter)


                                    # LSTM model without ABBA
                                    t0 = time.time()
                                    f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=best_hyperparameter), abba=None)
                                    f.train(patience=patience, max_epoch=max_epoch)
                                    forecast2 = f.forecast(len(test)).tolist()
                                    t1 = time.time()
                                        
                                    with open('./sensitivity/LSTM_results_' + dataset+'.csv', 'a', newline='') as csvfile:
                                        writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)

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

                                        row = [i, len(train), t, smape, euclid, diff_euclid, dtw, diff_dtw, dtw_reconstructed, diff_dtw_reconstructed, euclid_reconstructed, diff_euclid_reconstructed]
                                        writer.writerow(row)

                                
                                else:
                                    # LSTM model without ABBA
                                    t0 = time.time()
                                    f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=best_hyperparameter), abba=None)
                                    f.train(patience=patience, max_epoch=max_epoch)
                                    forecast2 = f.forecast(len(test)).tolist()
                                    t1 = time.time()
                                        
                                    with open('./sensitivity/LSTM_results_' + dataset+'.csv', 'a', newline='') as csvfile:
                                        writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
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

                                        row = [i, len(train), t, smape, euclid, diff_euclid, dtw, diff_dtw, dtw_reconstructed, diff_dtw_reconstructed, euclid_reconstructed, diff_euclid_reconstructed]
                                        writer.writerow(row)

                                
                                # Produce plot
                                fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1)
                                plt.subplots_adjust(left=0.125, bottom=None, right=0.95, top=None, wspace=None, hspace=None)
                                ax1.plot(train, 'k')
                                ax1.plot(abba_numerical, 'k', alpha=0.5, label='ABBA representation')
                                ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast1), 'r', label='ABBA_LSTM')
                                ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast2), 'b', label='LSTM')
                                ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(test), 'y', label='truth')
                                plt.legend(loc=6)
                                plt.savefig('./sensitivity/plots/' + dataset + '_' +str(i)+'.pdf')
                                plt.close()

                    except Exception as e:
                        print(e)



    ###############################################################################
    # Import excel document into pandas dataframe

    if not os.path.isfile('./sensitivity/ABBA_LSTM_results_M3.csv'):
        with open('./sensitivity/ABBA_LSTM_results_M3.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    if not os.path.isfile('./sensitivity/LSTM_results_M3.csv'):
        with open('./sensitivity/LSTM_results_M3.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    import pandas as pd
    xls = pd.ExcelFile('../paper/M3_competition/M3C.xls')

    # Select time series which are sampled monthly, should run from N1402 - N2829
    Monthly = xls.parse(2)

    def sMAPE(A, F):
        return 100/len(A) * np.sum(2 * np.abs(A - F) / (np.abs(A) + np.abs(F)))

    lag = 10
    patience = 100
    fcast_len = 18

    for index, row in Monthly[0:].iterrows():
        if index > 20:
            break
        if True:
            # import row and remove NaN padding
            ts = row.array[6:].to_numpy(dtype=np.float64)
            ts = ts[~np.isnan(ts)]

            train = ts[:-fcast_len]
            test = ts[-fcast_len:]

            # remove class information
            ts = np.array(ts[1:])
            # remove NaN from time series
            ts = ts[~np.isnan(ts)]

            # Normalise time series
            ts -= np.mean(ts)
            ts /= np.std(ts)

            train = ts[:-fcast_len]
            test = ts[-fcast_len:]

            # GRID SEARCH 
            if index == 0:
                min_smape = np.inf

                hyperparameters = {}
                hyperparameters['lag'] = [5, 10, 15]
                hyperparameters['max_k'] = [15, 20, 25]
                hyperparameters['tol'] = [0.01, 0.03, 0.05]

                for lag in hyperparameters['lag']:
                    for max_k in hyperparameters['max_k']:
                        for tol in hyperparameters['tol']:

                            # Build ABBA constructor
                            abba = ABBA(tol=tol, max_k = max_k, verbose=0)
                            string, centers = abba.transform(train)
                            abba_numerical = abba.inverse_transform(string, centers, train[0])
                            
                            # LSTM model with ABBA
                            t0 = time.time()
                            f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=lag, cells_per_layer=20), abba=abba)
                            f.train(patience=patience, max_epoch=max_epoch)
                            forecast1 = f.forecast(len(test)).tolist()
                            t1 = time.time()
                    
                            smape = sMAPE(test, forecast1)

                            if smape < min_smape:
                                min_smape = smape
                                best_hyperparameters = [lag, max_k, tol]

                print('best hyperparameters', best_hyperparameters)
                # Build ABBA constructor
                abba = ABBA(tol=best_hyperparameters[2], max_k =best_hyperparameters[1] , verbose=0)
                string, centers = abba.transform(train)
                abba_numerical = abba.inverse_transform(string, centers, train[0])

                # LSTM model with ABBA
                t0 = time.time()
                f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=best_hyperparameters[0], cells_per_layer=20), abba=abba)
                f.train(patience=patience, max_epoch=max_epoch)
                forecast1 = f.forecast(len(test)).tolist()
                t1 = time.time()
                    
                with open('./sensitivity/ABBA_LSTM_results_M3.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
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

            
            else:
                abba = ABBA(tol=best_hyperparameters[2], max_k =best_hyperparameters[1] , verbose=0)
                string, centers = abba.transform(train)
                abba_numerical = abba.inverse_transform(string, centers, train[0])

                # LSTM model with ABBA
                t0 = time.time()
                f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=best_hyperparameters[0], cells_per_layer=20), abba=abba)
                f.train(patience=patience, max_epoch=max_epoch)
                forecast1 = f.forecast(len(test)).tolist()
                t1 = time.time()
                    
                with open('./sensitivity/ABBA_LSTM_results_M3.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    
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

            # GRID SEARCH without ABBA 
            if index == 0:
                min_smape = np.inf

                hyperparameters = {}
                hyperparameters['lag'] = [5, 10, 20, 30]

                for lag in hyperparameters['lag']:

                    # LSTM model with ABBA
                    t0 = time.time()
                    f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=lag, cells_per_layer=20), abba=None)
                    f.train(patience=patience, max_epoch=max_epoch)
                    forecast2 = f.forecast(len(test)).tolist()
                    t1 = time.time()
                    
                    smape = sMAPE(test, forecast1)
                    if smape < min_smape:
                        min_smape = smape
                        best_hyperparameter = lag

                print('best hyperparameter', best_hyperparameter)


                # LSTM model without ABBA
                t0 = time.time()
                f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=best_hyperparameter, cells_per_layer=20), abba=None)
                f.train(patience=patience, max_epoch=max_epoch)
                forecast2 = f.forecast(len(test)).tolist()
                t1 = time.time()
                    
                with open('./sensitivity/LSTM_results_M3.csv', 'a', newline='') as csvfile:
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

            
            else:
                # LSTM model without ABBA
                t0 = time.time()
                f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=best_hyperparameter, cells_per_layer=20), abba=None)
                f.train(patience=patience, max_epoch=max_epoch)
                forecast2 = f.forecast(len(test)).tolist()
                t1 = time.time()
                    
                with open('./sensitivity/LSTM_results_M3.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
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
            ax1.plot(train, 'k')
            ax1.plot(abba_numerical, 'k', alpha=0.5, label='ABBA representation')
            ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast1), 'r', label='ABBA_LSTM')
            ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast2), 'b', label='LSTM')
            ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(test), 'y', label='truth')
            plt.legend(loc=6)
            plt.savefig('./sensitivity/plots/M3_' +str(index)+'.pdf')
            plt.close()


if __name__ == '__main__':
    test_sensitity()