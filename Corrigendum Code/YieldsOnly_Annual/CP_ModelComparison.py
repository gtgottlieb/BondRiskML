#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Imports
import numpy as np
import pandas as pd
import scipy.io as sio
import multiprocessing as mp
import NNFuncBib_CP as NFB
import os
import time
import psutil
import statsmodels.api as sm
from datetime import datetime
import shutil
from scipy.stats import t as tstat


# %% Functions

def multProcessOwn(NNfunc,ncpus,nMC,X,Y,**kwargs):
    outputCons = None
    while outputCons==None:
        try:
            pool = mp.Pool(processes=ncpus)
            output = [pool.apply_async(NNfunc, args=(X,Y,no),kwds=kwargs) for no in range(nMC)]
            outputCons = [p.get(timeout=500) for p in output]
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


def R2OOS(y_true, y_forecast):
    import numpy as np

    # Compute conidtional mean forecast
    y_condmean = np.divide(y_true.cumsum(),(np.arange(y_true.size)+1))

    # lag by twelve periods
    y_condmean = np.insert(y_condmean,0,np.full((12,), np.nan))
    y_condmean = y_condmean[:-12]
    y_condmean[np.isnan(y_forecast)] = np.nan

    # Sum of Squared Resids
    SSres = np.nansum(np.square(y_true-y_forecast))
    SStot = np.nansum(np.square(y_true-y_condmean))

    return 1-SSres/SStot


def RSZ_Signif(y_true, y_forecast):

    # Compute conidtional mean forecast
    y_condmean = np.divide(y_true.cumsum(),(np.arange(y_true.size)+1))

    # lag by one period
    y_condmean = np.insert(y_condmean,0,np.full((12,), np.nan))
    y_condmean = y_condmean[:-12]
    y_condmean[np.isnan(y_forecast)] = np.nan

    # Compute f-measure
    f = np.square(y_true-y_condmean)-np.square(y_true-y_forecast) + np.square(y_condmean-y_forecast)

    # Regress f on a constant
    x = np.ones(np.shape(f))
    model = sm.OLS(f,x, missing='drop', hasconst=True)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':12})


    return 1-tstat.cdf(results.tvalues[0],results.nobs-1)


# %%
if __name__ == "__main__":

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
    data_path = '../../../Data/YIELDS_FWDS/CPDataStore.h5'
    data_set = 'LW'
    rhs_vars_set = 'Large'
    run_tag = 'RHS_'+rhs_vars_set

    HyperFreq = 5*12
    NN_Params = {'Dropout': [0.2, 0.4], 'l1l2': [0.5, 1]}


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

    # %% Load Data
    exrets = pd.read_hdf(data_path, key='exret_'+data_set)
    exrets.set_index(['Date'], inplace=True)
    fwdrate = pd.read_hdf(data_path, key='yields_'+data_set+'_'+rhs_vars_set)
    fwdrate.set_index(['Date'], inplace=True)

    X = fwdrate.values
    Y = exrets.values


    # Determine IS vs OS indices
    T = X.shape[0]
    tstart = np.argmax(fwdrate.index==OOS_Start)
    OoS_indeces = range(tstart, int(T))
    # Number of outputs
    M = Y.shape[1]

    if TestFlag:
        OoS_indeces = OoS_indeces[:2]


    #%% Choose Models to Compute
    models = [NFB.NN1Layer_3, NFB.NN1Layer_5, NFB.NN1Layer_7,
              NFB.NN2Layer_3, NFB.NN2Layer_5, NFB.NN2Layer_7,
              NFB.NN3Layer_3, NFB.NN3Layer_5, NFB.NN3Layer_543, NFB.NN3Layer_7,
              NFB.NN4Layer_3, NFB.NN4Layer_5, NFB.NN4Layer_5432]

    modelnames = ['NN1Layer_3', 'NN1Layer_5', 'NN1Layer_7',
                  'NN2Layer_3', 'NN2Layer_5', 'NN2Layer_7',
                  'NN3Layer_3', 'NN3Layer_5', 'NN3Layer_543', 'NN3Layer_7',
                  'NN4Layer_3', 'NN4Layer_5', 'NN4_Layer_5432']

    # Initialize Object for Saving
    VarSave = {}
    VarSave['Note'] = ''
    VarSave["Y_True"] = Y
    VarSave['RF'] = X[:, 0]
    VarSave["Dates"] = list(fwdrate.index.strftime("%Y-%m-%d"))

    for modelnum, modelfunc in enumerate(models):

        Y_forecast = np.full([T, nMC, M], np.nan)
        Y_forecast_agg = np.full([T, M], np.nan)
        val_loss = np.full([T, nMC], np.nan)
        print(modelnames[modelnum])

        # Case Distinction: single core vs multi-core as per flag above
        if usemultproc == 1 and (('NN' in modelnames[modelnum])):
            print("Use multproc")
            j = 1  # index to keep track of grid-search steps
            for i in OoS_indeces:
                start = time.time()
                if (j == 1) or (j % HyperFreq == 0):
                    refit = True
                else:
                    refit = False
                j += 1



                # Run the model in parallel for nMC times
                output = multProcessOwn(modelfunc, ncpus, nMC,
                                        X[:i+1, :], Y[:i+1, :],
                                        params=NN_Params,
                                        refit=refit, dumploc=dumploc)
                # Collect and organize output
                val_loss[i, :] = np.array([output[k][1] for k in range(nMC)])
                Y_forecast[i, :, :] = np.concatenate([output[k][0] for k in range(nMC)],axis=0)
                tempsort = np.argsort(val_loss[i, :])
                ypredmean = np.mean(Y_forecast[i, tempsort[:nAvg], :], axis=0)
                Y_forecast_agg[i, :] = ypredmean

                print(i)
                print(time.time() - start)
                processmem = psutil.Process(os.getpid())
                print(processmem.memory_info().rss)


                print('R2OOS: ',np.array([R2OOS(Y[:, k], Y_forecast_agg[:, k]) for k in range(np.size(Y, axis=1))]))


        else:
            raise Exception("modelfunc not matchable")

        VarSave["ValLoss_"+modelnames[modelnum]] = val_loss
        VarSave["Y_forecast_agg_"+modelnames[modelnum]] = Y_forecast_agg
        VarSave["Y_forecast"+modelnames[modelnum]] = Y_forecast

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
                VarSaveSOA = sio.loadmat('CP_ModelComparison_'+data_set+'_SOA'+TestString+'_'+run_tag+'.mat')
                VarSaveSOA.update(VarSave)
                sio.savemat('CP_ModelComparison_'+data_set+'_SOA'+TestString+'_'+run_tag+'.mat', VarSaveSOA)
                savesuccess_flag = True
                print('Updated SOA file')
            except FileNotFoundError:
                sio.savemat('CP_ModelComparison_'+data_set+'_SOA'+TestString+'_'+run_tag+'.mat', VarSave)
                savesuccess_flag = True
                print('Created new SOA file')
            except:
                print('Could not write. Sleeping 1 sec.')
                time.sleep(1)


    # Delete Dumploc
    try:
        shutil.rmtree(dumploc)
        print('Removed dir: '+dumploc+' succesfully')
    except:
        print('Directory: '+dumploc+' could not be removed')
