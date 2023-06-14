import os
import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import inspect 
from scipy.optimize import curve_fit
from tabulate import tabulate
from simple_colors import *

class results:

    def __init__(self, folder, name='results'):
        self.folder = folder
        self.name = name
        self.file = os.path.join(self.folder, self.name) + '.npy'
        if os.path.exists(self.file):
            print('Result file exists: ' + self.file)
            self.analyzedData = np.load(self.file, allow_pickle=True).item()
        else:
            self.analyzedData = {}

    def addResult(self, run, params, x, y, yerr=np.nan):
        res = {run: {'params': params, 'x': x, 'y': y, 'yerr': yerr}}
        self.analyzedData.update(res)
        self.saveResults()

    def getResult(self, run, key_list=None):
        """
        Default return values: x, y, yerr
        """
        if key_list is None:
            key_list = ['x', 'y', 'yerr']
        result_list = [self.analyzedData[run][key] for key in key_list]
        return result_list

    def saveResults(self):
        np.save(self.file, self.analyzedData)

    def getSingleParameterData(self, run, paramValues, includeError=True):
        if not isinstance(paramValues, list):
            paramValues = [paramValues]
        n = len(paramValues)
        dat = self.analyzedData[run]
        x, y = dat['x'], dat['y']
        idx = np.all(x[:, 1:(n + 1)] == paramValues, axis=1)
        if includeError:
            yerr = dat['yerr']
            return x[idx, 0], y[idx], yerr[idx]
        else:
            return x[idx, 0], y[idx]

    def plot(self, runs, plotOptions=None, xScale=1):
        if plotOptions is None:
            plotOptions = {'marker': 'o', 'linestyle': ''}
        plotArgs = {}
        plotArgs.update(plotOptions)

        for run in runs:
            res = self.analyzedData[run]

            if len(res['params']) == 1:
                plt.errorbar(res['x'] * xScale, res['y'].squeeze(), yerr=res['yerr'], label=run, **plotArgs)
            elif len(res['params']) == 2:
                x2_list = np.unique(res['x'][:, 1])
                for x2 in x2_list:
                    idx = (res['x'][:, 1] == x2)
                    if (np.array(res['yerr']) == np.nan).any():
                        plt.plot(res['x'][idx, 0] * xScale, res['y'][idx], label=str(x2), **plotArgs)
                    else:
                        plt.errorbar(res['x'][idx, 0] * xScale, res['y'][idx], yerr=res['yerr'][idx], label=str(x2),
                                     **plotArgs)
            elif len(res['params']) == 3:
                x2_list = np.unique(res['x'][:, 1])
                x3_list = np.unique(res['x'][:, 2])
                for x2 in x2_list:
                    for x3 in x3_list:
                        idx = ([res['x'][:, 1]] == x2) * (res['x'][:, 2] == x3)
                        if (np.array(res['yerr']) == np.nan).any():
                            plt.plot(res['x'][idx[0], 0] * xScale, res['y'][idx[0]], label=str(x2) + ', ' + str(x3),
                                     **plotArgs)
                        else:
                            plt.errorbar(res['x'][idx, 0] * xScale, res['y'][idx], yerr=res['yerr'][idx],
                                         label=str(x2) + ', ' + str(x3), **plotArgs)
        plt.xlabel(res['params'][0])
        plt.minorticks_on()
        plt.grid(b=True, which='major')

    def plot2d(self, runs, plotOptions=None, xScale=1):
        if plotOptions is None:
            plotOptions = {'marker': 'o', 'linestyle': ''}
        plotArgs = {}
        plotArgs.update(plotOptions)
        for run in runs:
            res = self.analyzedData[run]

            if len(res['params']) != 2:
                raise ValueError('Need precisely two parameters to plot2d! Parameters given currently: ' +
                                 str(res['params']))
            else:
                x1, x2 = res['x'].T
                y = res['y']
                plt.tripcolor(x1, x2, y)
        plt.xlabel(res['params'][0])
        plt.ylabel(res['params'][1])
        plt.colorbar()

    def plotVsSite(self, run, xScale=1, cbarLabel='occuppation', vLims=None, cmap='viridis'):
        if vLims is None:
            vLims = [0, 1]
        res = self.analyzedData[run]
        x, y = xScale * res['x'], res['y']

        if xScale < 1:
            y = np.flip(y, axis=0)

        plt.figure(figsize=(20, 5))

        xticks = np.sort(np.linspace(x.min(), x.max(), 10))

        ax = sns.heatmap(y.T, xticklabels=np.round(xticks, 2), cmap=cmap, cbar_kws={'label': cbarLabel}, vmin=vLims[0],
                         vmax=vLims[1])
        ax.set_xticks(np.abs((xticks - xticks[0]) / (x.max() - x.min())) * len(x))
        ax.set_xticklabels(np.round(xticks, 2), rotation=0, fontsize=15);
        ax.tick_params(axis='y', labelsize=13)
        plt.xlabel(res['params'][0], fontsize=15)
        plt.ylabel('site', fontsize=15)

    def saveAnalysis(self, run, **kwds):
        """
        Save the data from analysis to a local file, so that one can easily find and copy the data.
        Data will be saved to '...\\Daily\\YYMM\\YYMMDD\\analysisData\\runXX.json'.
        e.g. saveAnalysis('run4', WL=556e-9, linewidth=182e+3)
        """
        analysisPath = os.path.join(self.folder, 'analysisData') + os.sep
        if os.path.exists(analysisPath) == False:
            os.makedirs(analysisPath)
        filepath = analysisPath + run

        if os.path.exists(filepath):
            old_dat = pickle.load(open(filepath, 'rb')).copy()
            new_dat = old_dat.update(kwds)
            pickle.dump(old_dat, open(filepath, 'wb'))
        else:
            pickle.dump(kwds, open(filepath, 'wb'))

    def loadAnalysis(self, run, date='today'):
        """
        Load the analysis data of a given run on a given day.
        'run' can either be 'runXX' or an integer (equivalent to 'runX').
        'date' should be either 'today' or 'YYMMDD'.
        'resType' should be either 'experiment' or 'simulation'
        e.g. loadAnalysis('run3') or loadAnalysis(6, date='220427').
        """
        if date == 'today':
            analysisPath = os.path.join(self.folder, 'analysisData') + os.sep
        else:
            parent_folder = os.path.abspath(
                os.path.join(self.folder, '..', '..'))  # Should be 'I:\\thompsonlab\\AMO\\Daily'
            analysisPath = os.path.join(parent_folder, date[0:4], date, 'analysisData')
        if isinstance(run, int):
            run = 'run' + str(run)
        filename = run
        dat = pickle.load(open(os.path.join(analysisPath, filename), 'rb'))
        return dat
    
    def fitResult(self, run, label, fn, x, y, yerr=None, fit_options=None, print_result=True):
        """
        Help fit the analyzed result of a certain run. 
        """

        fit, cov = curve_fit(fn, x, y, sigma=yerr, **fit_options)
        err = np.diag(cov) ** 0.5
        fitForPrint = []
        parameterNames = list(inspect.signature(fn).parameters.keys())
        if print_result:
            for i in range(len(parameterNames) - 1):
                fitForPrint.append([i, parameterNames[i+1], np.round(fit[i], 3), np.round(err[i],3)])
            print('Fit function: '+ fn.__name__)
            print('-*'*10+'Function details'+'*-'*10)
            help(fn)
            print('-*'*10+'*-'*10)
            print(tabulate(fitForPrint, headers=["index", "Parameter name", "Fit", "1-sigma uncertainty"], tablefmt='pipe'))
        res = {run + label + '_fits': {'fits': fit, 'errs': err}}
        self.analyzedData.update(res)
        return fit, err

        
    
    def plotFittedCurve(self, fit_func, x_data, fit_result, num_steps=1000):
        xx = np.linspace(np.amin(x_data), np.amax(x_data), num_steps)
        return plt.plot(xx, fit_func(xx, *fit_result))
        
    