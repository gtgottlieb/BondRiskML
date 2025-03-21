#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Imports
import numpy as np
import pandas as pd
import scipy.io as sio
import multiprocessing as mp
import NNFuncBib as NFB
import os
import time
import psutil
import statsmodels.api as sm
from datetime import datetime
from scipy.stats import t as tstat

import shutil

#%% Functions

def multProcessOwn(NNfunc, ncpus, nMC, X, Y, **kwargs):
    outputCons = None
    while outputCons==None:
        try:
            pool = mp.Pool(processes=ncpus)
            output = [pool.apply_async(NNfunc, args=(X,Y,no), kwds=kwargs) for no in range(nMC)]
            outputCons = [p.get(timeout=3000) for p in output]
            pool.close()
            pool.join()
            time.sleep(1)
            # Get process results from the output queue
        except Exception as e:
            print(str(e))
            print("Timed out, shutting pool down")
            pool.close()
            pool.terminate()
            time.sleep(1)
            continue

    return outputCons

def multProcessOwnExog(NNfunc,ncpus,nMC,X,Xexog,Y,**kwargs):
    outputCons = None
    while outputCons==None:
        try:
            pool = mp.Pool(processes=ncpus)
            output = [pool.apply_async(NNfunc, args=(X,Xexog,Y,no,),kwds=kwargs) for no in range(nMC)]
            outputCons = [p.get(timeout=3000) for p in output]
            pool.close()
            pool.join()
            time.sleep(1)
            # Get process results from the output queue
        except Exception as e:
            print(e)
            print("Timed out, shutting pool down")
            pool.close()
            pool.terminate()
            time.sleep(1)
            continue

    return outputCons

def multProcessOwnEnsem(NNfunc,ncpus,nMC,X,A,Y,**kwargs):
    outputCons = None
    while outputCons==None:
        try:
            pool = mp.Pool(processes=ncpus)
            output = [pool.apply_async(NNfunc, args=(X,A,Y,no),kwds=kwargs) for no in range(nMC)]
            outputCons = [p.get(timeout=3000) for p in output]
            pool.close()
            pool.join()
            time.sleep(1)
            # Get process results from the output queue
        except Exception as e:
            print(e)
            print("Timed out, shutting pool down")
            pool.close()
            pool.terminate()
            time.sleep(1)
            continue

    return outputCons



def CP_factor(exrets,fwdrate):

    #Regress & Predict
    model = sm.OLS(np.mean(exrets[:-1,:],axis=1),sm.add_constant(fwdrate[:-1,:]))
    model = model.fit()
    CP_factor=(model.predict(sm.add_constant(fwdrate))).reshape(-1,1)
    return CP_factor


def R2OOS(y_true, y_forecast):
    import numpy as np

    # Compute conidtional mean forecast
    y_condmean = np.divide(y_true.cumsum(),(np.arange(y_true.size)+1))

    # lag by one period
    y_condmean = np.insert(y_condmean,0,np.nan)
    y_condmean = y_condmean[:-1]
    y_condmean[np.isnan(y_forecast)] = np.nan

    # Sum of Squared Resids
    SSres = np.nansum(np.square(y_true-y_forecast))
    SStot = np.nansum(np.square(y_true-y_condmean))

    return 1-SSres/SStot


def RSZ_Signif(y_true, y_forecast):

    # Compute conidtional mean forecast
    y_condmean = np.divide(y_true.cumsum(),(np.arange(y_true.size)+1))

    # lag by one period
    y_condmean = np.insert(y_condmean,0,np.nan)
    y_condmean = y_condmean[:-1]
    y_condmean[np.isnan(y_forecast)] = np.nan

    # Compute f-measure
    f = np.square(y_true-y_condmean)-np.square(y_true-y_forecast) + np.square(y_condmean-y_forecast)

    # Regress f on a constant
    x = np.ones(np.shape(f))
    model = sm.OLS(f,x, missing='drop', hasconst=True)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':12})


    return 1-tstat.cdf(results.tvalues[0],results.nobs-1)


#%%

if __name__ == "__main__":
#%% Global Params
    #=========================================================================
    #                           User Settings
    #=========================================================================

    # Set True for testing False otherwise
    TestFlag = False
    LocalCPU = False
    if LocalCPU:
        nMC=8
        nAvg=4
    else:
        nMC = 100
        nAvg = 10

    OOS_Start = '1989-01-31'
    data_path = '../../../../Data/MACRO/BondPFs/LN_data_1964_2016_monthly.mat'
    data_set = 'PFs'
    run_tag = 'SmallRHS'


    HyperFreq = 5*12
    NN_Params = {'Dropout': [0.3, 0.5], 'l1': [0.01, 0.001],
                 'l2': [0.01, 0.001]}


    #=========================================================================

    # timestamp for file
    now = str(datetime.now())
    # Reduce Computational Cost for Testing
    if TestFlag:
        nMC = 4
        nAvg = 2
    usemultproc = 1
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()
    print("CPU count is: "+str(ncpus))


    #MultProc Dump Loc
    dumploc_base = './trainingDumps_'

    i = 0
    succeeded_flag = False
    while not succeeded_flag:
        dumploc = dumploc_base+str(i)
        try:
            # Create target DirecShort
            os.mkdir(dumploc)
            print("Directory " , dumploc,  " Created ")
            succeeded_flag = True
        except FileExistsError:
            print("Directory " , dumploc ,  " already exists")
            i = i + 1

    CP_data = sio.loadmat(data_path)
    exrets = CP_data['ExRets']
    fwdrate = CP_data['fwdrate']
    FREDdata = CP_data['FRED_data']
    dates = pd.DataFrame(CP_data['dates_excel'],columns=['date'])

    dates = pd.TimedeltaIndex(dates['date'], unit='d') + datetime(1900,1,1)
    dates = dates + pd.offsets.MonthEnd(-1)
    Xexog = fwdrate
    X = FREDdata
    Y = exrets

    A = CP_data['McCrackenNG_Group']

    # Determine IS vs OS indices
    T = X.shape[0]
    tstart = np.argmax(dates==OOS_Start)
    OoS_indeces = range(tstart, int(T))
    # Number of outputs
    M = Y.shape[1]

    if TestFlag:
        OoS_indeces = OoS_indeces[:2]

# %% FRED factors + Term Structure - Select models to run

    models = [NFB.NN1LayerExog, NFB.NN2LayerExog, NFB.NN3LayerExog,
              NFB.NN1LayerExog_ExogNet1Layer, NFB.NN2LayerExog_ExogNet1Layer,
              NFB.NN3LayerExog_ExogNet1Layer,
              NFB.NN1LayerExog_ExogNet, NFB.NN2LayerExog_ExogNet,
              NFB.NN3LayerExog_ExogNet,
              NFB.NN1LayerEnsemExog,NFB.NN1LayerEnsemExog_ExogNet,
              NFB.NN2LayerEnsemExog,NFB.NN2LayerEnsemExog_ExogNet,
              NFB.NN3LayerEnsemExog,NFB.NN3LayerEnsemExog_ExogNet]


    modelnames = ['NN1LayerExog', 'NN2LayerExog', 'NN3LayerExog',
                  'NN1LayerExog_ExogNet1Layer', 'NN2LayerExog_ExogNet1Layer',
                  'NN3LayerExog_ExogNet1Layer',
                  'NN1LayerExog_ExogNet', 'NN2LayerExog_ExogNet',
                  'NN3LayerExog_ExogNet',
                  'NN1LayerEnsemExog','NN1LayerEnsemExog_ExogNet',
                  'NN2LayerEnsemExog','NN2LayerEnsemExog_ExogNet']



    for modelnum,modelfunc in enumerate(models):

        Y_forecast = np.full([T, nMC, M], np.nan)
        Y_forecast_agg = np.full([T, M], np.nan)
        val_loss = np.full([T, nMC], np.nan)
        print(modelnames[modelnum])

        # Case Distinction: single core vs multi-core as per flag above
        if usemultproc == 1 and (('NN' in modelnames[modelnum]) or ('AutoEnc' in modelnames[modelnum])) and ('Ensem' not in modelnames[modelnum]) and ('CP' not in modelnames[modelnum]):
            print("Use multproc")
            j = 1
            for i in OoS_indeces:
                start = time.time()
                if (j == 1) or (j % HyperFreq == 0):
                    refit = True
                else:
                    refit = False
                j += 1
                output = multProcessOwnExog(modelfunc,ncpus,nMC,X[:i+1,:],Xexog[:i+1,:],Y[:i+1,:],
                                            dumploc=dumploc, params=NN_Params,
                                        refit=refit)
                val_loss[i, :] = np.array([output[k][1] for k in range(nMC)])
                Y_forecast[i, :, :] = np.concatenate([output[k][0] for k in range(nMC)],axis=0)
                tempsort = np.argsort(val_loss[i, :])
                ypredmean = np.mean(Y_forecast[i, tempsort[:nAvg], :], axis=0)
                Y_forecast_agg[i, :] = ypredmean
                print(i)
                print(time.time() - start)
                processmem = psutil.Process(os.getpid())
                print(processmem.memory_info().rss)
                print(np.array(
                [R2OOS(Y[:, k], Y_forecast_agg[:, k]) for k in range(np.size(Y, axis=1))]))

        elif 'Ensem' in modelnames[modelnum] and usemultproc == 1 and ('FullCV' not in modelnames[modelnum]):
            print("Use multproc")
            j = 1
            for i in OoS_indeces:
                start = time.time()
                if (j == 1) or (j % HyperFreq == 0):
                    refit = True
                else:
                    refit = False
                j += 1
                output = multProcessOwnExog(modelfunc,ncpus,nMC,X[:i+1,:],
                                            Xexog[:i+1,:],Y[:i+1,:],
                                            A=A, params=NN_Params, refit=refit,
                                            dumploc=dumploc)
                val_loss[i, :] = np.array([output[k][1] for k in range(nMC)])
                Y_forecast[i, :, :] = np.concatenate([output[k][0] for k in range(nMC)],axis=0)
                tempsort = np.argsort(val_loss[i, :])
                ypredmean = np.mean(Y_forecast[i, tempsort[:nAvg], :], axis=0)
                Y_forecast_agg[i, :] = ypredmean
                print(i)
                print(time.time() - start)
                print(np.array(
                [R2OOS(Y[:, k], Y_forecast_agg[:, k]) for k in range(np.size(Y, axis=1))]))

        else:
            raise Exception("modelfunc not matchable")

        VarSave["ValLoss_"+modelnames[modelnum]] = val_loss
        VarSave["Y_forecast_agg_"+modelnames[modelnum]] = Y_forecast_agg
        VarSave["Y_forecast_"+modelnames[modelnum]] = Y_forecast

        # Mean Squared Error
        VarSave["MSE_"+modelnames[modelnum]] = np.nanmean(np.square(Y-Y_forecast_agg), axis=0)

        # R2-00S
        VarSave["R2OOS_"+modelnames[modelnum]] = np.array(
                [R2OOS(Y[:, k], Y_forecast_agg[:, k]) for k in range(np.size(Y, axis=1))])

        print('R2OOS: ',VarSave["R2OOS_"+modelnames[modelnum]])

        # Rapach, Strauss, Zhou (2010 Significance)
        VarSave["R2OOS_pval_"+modelnames[modelnum]] = np.array(
        [RSZ_Signif(Y[:, k], Y_forecast_agg[:, k]) for k in range(np.size(Y, axis=1))])

        # If Test append to output file name
        if TestFlag:
            TestString='_Test'
        else:
            TestString=''

        savesuccess_flag = False
        while savesuccess_flag==False:
            # Save new result to SOA file
            try:
                VarSaveSOA = sio.loadmat('LN_ModelComparison_'+data_set+'_SOA'+TestString+'_'+run_tag+'.mat')
                VarSaveSOA.update(VarSave)
                sio.savemat('LN_ModelComparison_'+data_set+'_SOA'+TestString+'_'+run_tag+'.mat', VarSaveSOA)
                savesuccess_flag = True
                print('Updated SOA file')
            except FileNotFoundError:
                sio.savemat('LN_ModelComparison_'+data_set+'_SOA'+TestString+'_'+run_tag+'.mat', VarSave)
                savesuccess_flag = True
                print('Created new SOA file')
            except:
                time.sleep(1)
                print('Could not write. Sleeping 1 sec.')

    # Delete Dumploc
    try:
        shutil.rmtree(dumploc)
        print('Removed dir: '+dumploc+' succesfully')
    except:
        print('Directory: '+dumploc+' could not be removed')
