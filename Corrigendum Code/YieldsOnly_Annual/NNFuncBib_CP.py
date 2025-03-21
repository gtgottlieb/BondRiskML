#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import os.path
from sklearn.model_selection import ParameterGrid

def NNGeneric(X, Y, no, archi, dropout_u=None, l1l2penal=None, refit=None, dumploc=None):

    if dumploc == None:
        raise ValueError('Missing Dumploc argument')

    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization
    from sklearn.utils import shuffle
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.optimizers import SGD
    from keras.models import load_model
    from keras import regularizers

    #Split Data for Test and Training
    X_train = X[:-12,:]
    Y_train = Y[:-12,:]
    X_test = X[-1,:].reshape(1,-1)

    #Scale the predictors for training
    Xscaler_train = MinMaxScaler(feature_range=(-1,1))
    X_scaled_train = Xscaler_train.fit_transform(X_train)


    # define base model
    tf.set_random_seed(no)
    np.random.seed(no)

    if refit:
        model = Sequential()
        for layer_i, nodes in enumerate(archi):
            if layer_i == 0:
                model.add(Dropout(dropout_u,input_shape=(X_scaled_train.shape[1],)))
                model.add(Dense(nodes, input_shape=(X_scaled_train.shape[1],),
                                kernel_initializer='he_normal',
                                bias_initializer='he_normal', activation='relu',
                                kernel_regularizer=regularizers.l1_l2(l1l2penal)))
            else:
                model.add(Dense(nodes,kernel_initializer='he_normal',
                                bias_initializer='he_normal',
                                kernel_regularizer=regularizers.l1_l2(l1l2penal),
                                activation='relu'))
            model.add(Dropout(dropout_u))
            #model.add(BatchNormalization())

        model.add(Dense(Y_train.shape[1], activation='linear',
                        bias_initializer='he_normal',kernel_initializer='he_normal'))

        # Compile model
        sgd_fine = SGD(lr=0.02, momentum=0.9, decay=0.001, nesterov=True)
        earlystopping = EarlyStopping(monitor='val_loss',min_delta=1e-6,patience=20,verbose=0, mode='auto')
        mcp = ModelCheckpoint(dumploc+'/BestModelWeights_'+str(no)+'.hdf5',monitor='val_loss',save_best_only=True)
        model.compile(loss='mean_squared_error', optimizer=sgd_fine)
        history=model.fit(X_scaled_train, Y_train, epochs=500,
                          callbacks=[earlystopping,mcp], validation_split=0.15,
                          batch_size=32, shuffle=True, verbose=0)
        #Start refit
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')
        model.save(dumploc+'/BestModel_'+str(no)+'.hdf5')

    else:
        model = load_model(dumploc+'/BestModel_'+str(no)+'.hdf5')
        sgd_fine = SGD(lr=0.02, momentum=0.9, decay=0.001, nesterov=True)
        earlystopping = EarlyStopping(monitor='val_loss',min_delta=1e-6,patience=20,verbose=0, mode='auto')
        mcp = ModelCheckpoint(dumploc+'/BestModelWeights_'+str(no)+'.hdf5',monitor='val_loss',save_best_only=True)
        model.compile(loss='mean_squared_error', optimizer=sgd_fine)
        history=model.fit(X_scaled_train, Y_train, epochs=500,
                  callbacks=[earlystopping,mcp], validation_split=0.15,
                  batch_size=32, shuffle=True, verbose=0)
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')


    #Perform Forecast and Test Model
    X_scaled_test = Xscaler_train.transform(X_test)
    Ypred = model.predict(X_scaled_test)


    return Ypred, np.min(history.history['val_loss'])



def NNGridSearchWrapper(X, Y, no, archi, params=None, refit=None, dumploc=None):
    # Construct grid of parameters from dictionary, containing param ranges
    paramgrid = list(ParameterGrid(params))

    # Loop over all param grid combinations and save val_loss
    val_loss = list()
    for i, param_i in enumerate(paramgrid):
        _, val_loss_temp = NNGeneric(X, Y, no, archi,
                                     dropout_u=param_i['Dropout'],
                                     l1l2penal=param_i['l1l2'],
                                     refit=True, dumploc=dumploc)
        val_loss.append(val_loss_temp)

    # Determine best model according to grid-search val_loss
    bestm = np.argmin(val_loss)

    # Fit best model again
    Ypred, val_loss = NNGeneric(X, Y, no, archi, dropout_u=paramgrid[bestm]['Dropout'],
                                l1l2penal=paramgrid[bestm]['l1l2'],
                                refit=True, dumploc=dumploc)

    return Ypred, val_loss



def NN1Layer_3(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss



def NN1Layer_5(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [5]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss

def NN1Layer_7(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [7]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss

def NN1Layer_1(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [1]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss



def NN2Layer_3(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [3, 3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss


def NN2Layer_5(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [5, 5]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss

def NN2Layer_7(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [7, 7]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss

def NN3Layer_543(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [5, 4, 3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss


def NN3Layer_3(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [3, 3, 3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss

def NN3Layer_5(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [5, 5, 5]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss

def NN3Layer_7(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [7, 7, 7]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss

def NN4Layer_5432(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [5, 4, 3, 2]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss

def NN4Layer_3(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [3, 3, 3, 3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss

def NN4Layer_5(X, Y, no, params=None, refit=None, dumploc=None):

    archi = [5, 5, 5, 5]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(X, Y, no, archi, params=params,
                                               refit=True, dumploc=dumploc)
    # Use existing model
    else:
        Ypred, val_loss = NNGeneric(X, Y, no, archi,
                                    refit=False, dumploc=dumploc)

    return Ypred, val_loss


