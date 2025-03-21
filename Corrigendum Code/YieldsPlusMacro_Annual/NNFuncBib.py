###!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.model_selection import ParameterGrid
from random import randrange
import pickle



def NNGridSearchWrapper(NNfunc, X, Y, no, params=None, refit=None, dumploc=None, **kwargs):
    # Construct grid of parameters from dictionary, containing param ranges
    paramgrid = list(ParameterGrid(params))

    # Loop over all param grid combinations and save val_loss
    val_loss = list()
    for i, param_i in enumerate(paramgrid):
        _, val_loss_temp = NNfunc(X, Y, no,
                                     dropout_u=param_i['Dropout'],
                                     l1l2penal=param_i['l1l2'],
                                     refit=True, dumploc=dumploc,
                                     **kwargs)
        val_loss.append(val_loss_temp)

    # Determine best model according to grid-search val_loss
    bestm = np.argmin(val_loss)

    # Fit best model again
    Ypred, val_loss = NNfunc(X, Y, no, dropout_u=paramgrid[bestm]['Dropout'],
                                l1l2penal=paramgrid[bestm]['l1l2'],
                                refit=True, dumploc=dumploc,
                                **kwargs)

    return Ypred, val_loss



def NNExogGeneric(X, Y, no, dropout_u=None, l1l2penal=None, refit=None, dumploc=None, **kwargs):

    if dumploc == None:
        raise ValueError('Missing Dumploc argument')

    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, BatchNormalization
    from sklearn.utils import shuffle
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.layers.merge import concatenate
    from keras.optimizers import SGD
    from keras.models import load_model
    from keras import regularizers


    Xexog = kwargs['Xexog']
    archi = kwargs['archi']

    #Split Data for Test and Training
    X_train = X[:-12,:]
    Xexog_train = Xexog[:-12,:]
    Y_train = Y[:-12,:]


    X_test = X[-1,:].reshape(1,-1)
    Xexog_test = Xexog[-1,:].reshape(1,-1)


    #Scale the predictors for training
    Xscaler_train =  MinMaxScaler(feature_range=(-1,1))
    X_scaled_train = Xscaler_train.fit_transform(X_train)
    Xexog_scaler_train =  MinMaxScaler(feature_range=(-1,1))
    Xexog_scaled_train = Xexog_scaler_train.fit_transform(Xexog_train)
    X_scaled_train = np.expand_dims(X_scaled_train, axis=1)
    Xexog_scaled_train = np.expand_dims(Xexog_scaled_train, axis=1)
    #Scale Responses for training
    Y_train = np.expand_dims(Y_train, axis=1)


    # define base model
    tf.set_random_seed(no)
    np.random.seed(no)

    if refit:
        #base model
        n = len(archi)
        layers = dict()
        for i in range(n+1):
            if i == 0:
                layers['ins_main'] = Input(shape=(1,X_scaled_train.shape[2]))
            elif i == 1:
                layers['dropout'+str(i)] = Dropout(dropout_u)(layers['ins_main'])
                layers['hidden'+str(i)] = Dense(archi[i-1],kernel_regularizer=regularizers.l1_l2(l1l2penal),bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu')(layers['dropout'+str(i)])
            elif i > 1 & i <= n:
                layers['dropout'+str(i)] = Dropout(dropout_u)(layers['hidden'+str(i-1)])
                layers['hidden'+str(i)] = Dense(archi[i-1],kernel_regularizer=regularizers.l1_l2(l1l2penal),bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu')(layers['dropout'+str(i)])

        #exog model
        layers['ins_exog'] = Input(shape=(1,Xexog_scaled_train.shape[2]))
        #merge models
        layers['merge'] = concatenate([layers['hidden'+str(n)], layers['ins_exog']])
        layers['dropout_final'] = Dropout(dropout_u)(layers['merge'])
        layers['BN'] = BatchNormalization()(layers['dropout_final'])
        layers['output'] = Dense(Y_train.shape[2],bias_initializer='he_normal', kernel_initializer='he_normal')(layers['BN'])
        model = Model(inputs=[layers['ins_main'], layers['ins_exog']], outputs=layers['output'])

        # Compile model
        sgd_fine = SGD(lr=0.01, momentum=0.9, decay=0.01, nesterov=True)
        earlystopping = EarlyStopping(monitor='val_loss',min_delta=1e-6,patience=20,verbose=0, mode='auto')
        mcp = ModelCheckpoint(dumploc+'/BestModelWeights_'+str(no)+'.hdf5',monitor='val_loss',save_best_only=True)
        model.compile(loss='mean_squared_error', optimizer=sgd_fine)
        history=model.fit([X_scaled_train, Xexog_scaled_train] , Y_train, epochs=500,
                          callbacks=[earlystopping,mcp], validation_split=0.15,
                          batch_size=32, shuffle=True, verbose=0)
        #Start refit
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')
        model.save(dumploc+'/BestModel_'+str(no)+'.hdf5')
        history = np.min(history.history['val_loss'])
        pickle.dump( history, open( dumploc+'/BestModelHist_'+str(no)+'.pkl', "wb" ))

    else:
        # For ease of computational burden do not retrain models every period
        model = load_model(dumploc+'/BestModel_'+str(no)+'.hdf5')
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')
        history = pickle.load( open( dumploc+'/BestModelHist_'+str(no)+'.pkl', "rb" ) )


    #Perform Forecast and Test Model
    X_scaled_test = Xscaler_train.transform(X_test)
    Xexog_scaled_test = Xexog_scaler_train.transform(Xexog_test)
    X_scaled_test = np.expand_dims(X_scaled_test, axis=1)
    Xexog_scaled_test = np.expand_dims(Xexog_scaled_test, axis=1)

    Ypred = model.predict([X_scaled_test,Xexog_scaled_test])
    Ypred = np.squeeze(Ypred,axis=1)



    return Ypred, history


def NN1LayerExog(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [32]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExogGeneric, X, Y, no, params=params,
                                               refit=True, dumploc=dumploc,
                                               archi=archi, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExogGeneric(X, Y, no,
                                        refit=False, dumploc=dumploc,
                                        archi=archi, Xexog=Xexog)

    return Ypred, val_loss

def NN2LayerExog(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [32, 16]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExogGeneric, X, Y, no, params=params,
                                               refit=True, dumploc=dumploc,
                                               archi=archi, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExogGeneric(X, Y, no,
                                        refit=False, dumploc=dumploc,
                                        archi=archi, Xexog=Xexog)

    return Ypred, val_loss

def NN3LayerExog(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [32, 16, 8]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExogGeneric, X, Y, no, params=params,
                                               refit=True, dumploc=dumploc,
                                               archi=archi, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExogGeneric(X, Y, no,
                                        refit=False, dumploc=dumploc,
                                        archi=archi, Xexog=Xexog)

    return Ypred, val_loss

def NN4LayerExog(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [64, 32, 16, 8]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExogGeneric, X, Y, no, params=params,
                                               refit=True, dumploc=dumploc,
                                               archi=archi, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExogGeneric(X, Y, no,
                                        refit=False, dumploc=dumploc,
                                        archi=archi, Xexog=Xexog)

    return Ypred, val_loss


def NNExog_ExogNetGeneric(X, Y, no, dropout_u=None, l1l2penal=None,
                           refit=None, dumploc=None, **kwargs):

    if dumploc == None:
        raise ValueError('Missing Dumploc argument')

    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, BatchNormalization
    from sklearn.utils import shuffle
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.layers.merge import concatenate
    from keras.optimizers import SGD
    from keras.models import load_model
    from keras import regularizers

    Xexog = kwargs['Xexog']
    archi = kwargs['archi']
    archiexog = kwargs['archiexog']

    #Split Data for Test and Training
    X_train = X[:-12,:]
    Xexog_train = Xexog[:-12,:]
    Y_train = Y[:-12,:]
    #X_train, Xexog_train, Y_train = shuffle(X_train,Xexog_train,Y_train, random_state=no)


    X_test = X[-1,:].reshape(1,-1)
    Xexog_test = Xexog[-1,:].reshape(1,-1)


    #Scale the predictors for training
    Xscaler_train =  MinMaxScaler(feature_range=(-1,1))
    X_scaled_train = Xscaler_train.fit_transform(X_train)
    Xexog_scaler_train =  MinMaxScaler(feature_range=(-1,1))
    Xexog_scaled_train = Xexog_scaler_train.fit_transform(Xexog_train)
    X_scaled_train = np.expand_dims(X_scaled_train, axis=1)
    Xexog_scaled_train = np.expand_dims(Xexog_scaled_train, axis=1)
    #Scale Responses for training
    Y_train = np.expand_dims(Y_train, axis=1)

    # define base model
    tf.set_random_seed(no)
    np.random.seed(no)

    if refit:
        #base model
        layers = dict()
        n = len(archi)
        n_exog = len(archiexog)
        for i in range(n+1):
            if i == 0:
                layers['ins_main'] = Input(shape=(1,X_scaled_train.shape[2]))
            elif i == 1:
                layers['dropout'+str(i)] = Dropout(dropout_u)(layers['ins_main'])
                layers['hidden'+str(i)] = Dense(archi[i-1],kernel_regularizer=regularizers.l1_l2(l1l2penal),bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu')(layers['dropout'+str(i)])
            elif i > 1 & i<=n:
                layers['dropout'+str(i)] = Dropout(dropout_u)(layers['hidden'+str(i-1)])
                layers['hidden'+str(i)] = Dense(archi[i-1],kernel_regularizer=regularizers.l1_l2(l1l2penal),bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu')(layers['dropout'+str(i)])


        #exog model
        for i in range(n_exog+1):
            if i == 0:
                layers['ins_exog'] = Input(shape=(1,Xexog_scaled_train.shape[2]))
            elif i == 1:
                layers['dropoutexog'+str(i)] = Dropout(dropout_u)(layers['ins_exog'])
                layers['hiddenexog'+str(i)] = Dense(archiexog[i-1],kernel_regularizer=regularizers.l1_l2(l1l2penal),bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu')(layers['dropoutexog'+str(i)])
            elif i > 1 & i<=n_exog:
                layers['dropoutexog'+str(i)] = Dropout(dropout_u)(layers['hiddenexog'+str(i-1)])
                layers['hiddenexog'+str(i)] = Dense(archiexog[i-1],kernel_regularizer=regularizers.l1_l2(l1l2penal),bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu')(layers['dropoutexog'+str(i)])

        #merge models
        layers['merge'] = concatenate([layers['hidden'+str(n)], layers['hiddenexog'+str(n_exog)]])

        layers['dropout_final'] = Dropout(dropout_u)(layers['merge'])
        layers['BN'] = BatchNormalization()(layers['dropout_final'])
        layers['output'] = Dense(Y_train.shape[2],bias_initializer='he_normal',kernel_initializer='he_normal')(layers['BN'])
        model = Model(inputs=[layers['ins_main'], layers['ins_exog']], outputs=layers['output'])


        # Compile model
        sgd_fine = SGD(lr=0.03, momentum=0.9, decay=0.01, nesterov=True)
        earlystopping = EarlyStopping(monitor='val_loss',min_delta=1e-6,patience=20,verbose=0, mode='auto')
        mcp = ModelCheckpoint(dumploc+'/BestModelWeights_'+str(no)+'.hdf5',monitor='val_loss',save_best_only=True)
        model.compile(loss='mean_squared_error', optimizer=sgd_fine)
        history=model.fit([X_scaled_train, Xexog_scaled_train], Y_train, epochs=500,
                          callbacks=[earlystopping,mcp], validation_split=0.15,
                          batch_size=32, shuffle=True, verbose=0)
        #Start refit
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')
        model.save(dumploc+'/BestModel_'+str(no)+'.hdf5')
        history = np.min(history.history['val_loss'])
        pickle.dump( history, open( dumploc+'/BestModelHist_'+str(no)+'.pkl', "wb" ))


    else:
        # To ease the computational burden the model weights as per the last
        # hyper-parameter search are used.
        model = load_model(dumploc+'/BestModel_'+str(no)+'.hdf5')
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')
        history = pickle.load( open( dumploc+'/BestModelHist_'+str(no)+'.pkl', "rb" ) )

    #Perform forecast and test model
    X_scaled_test = Xscaler_train.transform(X_test)
    Xexog_scaled_test = Xexog_scaler_train.transform(Xexog_test)
    X_scaled_test = np.expand_dims(X_scaled_test, axis=1)
    Xexog_scaled_test = np.expand_dims(Xexog_scaled_test, axis=1)


    Ypred = model.predict([X_scaled_test,Xexog_scaled_test])
    Ypred = np.squeeze(Ypred,axis=1)



    return Ypred, history



def NN1LayerExog_ExogNet(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [32]
    archiexog = [3, 3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExog_ExogNetGeneric, X, Y, no,  params=params,
                                               refit=True, dumploc=dumploc,
                                               Xexog=Xexog, archi=archi, archiexog=archiexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExog_ExogNetGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    Xexog=Xexog, archi=archi, archiexog=archiexog)

    return Ypred, val_loss


def NN1LayerExog_ExogNet1Layer(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [32]
    archiexog = [3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExog_ExogNetGeneric, X, Y, no,  params=params,
                                               refit=True, dumploc=dumploc,
                                               Xexog=Xexog, archi=archi, archiexog=archiexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExog_ExogNetGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    Xexog=Xexog, archi=archi, archiexog=archiexog)

    return Ypred, val_loss



def NN2LayerExog_ExogNet(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [32, 16]
    archiexog = [3, 3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExog_ExogNetGeneric, X, Y, no,  params=params,
                                               refit=True, dumploc=dumploc,
                                               Xexog=Xexog, archi=archi, archiexog=archiexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExog_ExogNetGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    Xexog=Xexog, archi=archi, archiexog=archiexog)

    return Ypred, val_loss


def NN2LayerExog_ExogNet1Layer(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [32, 16]
    archiexog = [3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExog_ExogNetGeneric, X, Y, no,  params=params,
                                               refit=True, dumploc=dumploc,
                                               Xexog=Xexog, archi=archi, archiexog=archiexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExog_ExogNetGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    Xexog=Xexog, archi=archi, archiexog=archiexog)

    return Ypred, val_loss


def NN3LayerExog_ExogNet1Layer(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [32, 16, 8]
    archiexog = [3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExog_ExogNetGeneric, X, Y, no,  params=params,
                                               refit=True, dumploc=dumploc,
                                               Xexog=Xexog, archi=archi, archiexog=archiexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExog_ExogNetGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    Xexog=Xexog, archi=archi, archiexog=archiexog)

    return Ypred, val_loss

def NN3LayerExog_ExogNet(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [32, 16, 8]
    archiexog = [3, 3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExog_ExogNetGeneric, X, Y, no,  params=params,
                                               refit=True, dumploc=dumploc,
                                               Xexog=Xexog, archi=archi, archiexog=archiexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExog_ExogNetGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    Xexog=Xexog, archi=archi, archiexog=archiexog)

    return Ypred, val_loss

def NN4LayerExog_ExogNet(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [64, 32, 16, 8]
    archiexog = [3, 3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExog_ExogNetGeneric, X, Y, no,  params=params,
                                               refit=True, dumploc=dumploc,
                                               Xexog=Xexog, archi=archi, archiexog=archiexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExog_ExogNetGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    Xexog=Xexog, archi=archi, archiexog=archiexog)

    return Ypred, val_loss


def NN4LayerExog_ExogNet1Layer(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    archi = [64, 32, 16, 8]
    archiexog = [3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExog_ExogNetGeneric, X, Y, no,  params=params,
                                               refit=True, dumploc=dumploc,
                                               Xexog=Xexog, archi=archi, archiexog=archiexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExog_ExogNetGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    Xexog=Xexog, archi=archi, archiexog=archiexog)

    return Ypred, val_loss






def NNEnsemExogGeneric(X, Y, no,  dropout_u=None, l1l2penal=None, refit=None, dumploc=None, **kwargs):

    if dumploc == None:
        raise ValueError('Missing Dumploc argument')

    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, BatchNormalization
    from sklearn.utils import shuffle
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.layers.merge import concatenate
    from keras.optimizers import SGD
    from keras.models import load_model
    from keras import regularizers

    A = kwargs['A']
    archi = kwargs['archi']
    Xexog = kwargs['Xexog']

    #Split Data for Test and Training
    X_train = X[:-12,:]
    Xexog_train = Xexog[:-12,:]
    Y_train = Y[:-12,:]


    X_test = X[-1,:].reshape(1,-1)
    Y_test = Y[-1,:].reshape(1,-1)
    Xexog_test = Xexog[-1,:].reshape(1,-1)


    #Scale the predictors for training
    Xscaler_train = MinMaxScaler(feature_range=(-1,1))
    X_scaled_train = Xscaler_train.fit_transform(X_train)
    X_scaled_train = np.expand_dims(X_scaled_train, axis=1)
    X_scaled_test = Xscaler_train.transform(X_test)
    Xexog_scaler_train = MinMaxScaler(feature_range=(-1,1))
    Xexog_scaled_train = Xexog_scaler_train.fit_transform(Xexog_train)
    Xexog_scaled_test = Xexog_scaler_train.transform(Xexog_test)
    Xexog_scaled_train = np.expand_dims(Xexog_scaled_train, axis=1)
    Xexog_scaled_test = np.expand_dims(Xexog_scaled_test, axis=1)
    Y_train = np.expand_dims(Y_train, axis=1)

    #Split X_train by group
    X_scaled_train_grouped = []
    X_scaled_test_grouped = []
    n_groups = len(np.unique(A))
    for i, group in enumerate(np.unique(A)):
        temp = X_scaled_train[:,A==group]
        X_scaled_train_grouped.append(np.expand_dims(temp, axis=1))
        temp = X_scaled_test[A==group].reshape(1,-1)
        X_scaled_test_grouped.append(np.expand_dims(temp, axis=1))


    # define base model
    tf.set_random_seed(no)
    np.random.seed(no)

    if refit:
        n = len(archi)
        layers = dict()

        for i in range(n+1):
            if i == 0:
                layers['ins_main'] = [Input(shape=(1,X_scaled_train_grouped[j].shape[2])) for j in range(n_groups)]
            elif i == 1:
                layers['dropout'+str(i)] = [Dropout(dropout_u)(input_tensor) for input_tensor in layers['ins_main']]
                layers['hiddens'+str(i)] = [Dense(archi[i-1], activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(l1l2penal))(input_tensor) for input_tensor in layers['dropout'+str(i)]]
            elif i > 1 & i <= n:
                layers['dropout'+str(i)] = [Dropout(dropout_u)(input_tensor) for input_tensor in layers['hiddens'+str(i-1)]]
                layers['hiddens'+str(i)] = [Dense(archi[i-1], activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(l1l2penal))(input_tensor) for input_tensor in layers['dropout'+str(i)]]

        layers['ins_exog'] = Input(shape=(1,Xexog_scaled_train.shape[2]))
        layers['merge'] = concatenate(layers['hiddens'+str(n)])
        layers['merge1'] = concatenate([layers['merge'], layers['ins_exog']])
        layers['dropout_final'] = Dropout(dropout_u)(layers['merge1'])
        layers['BN'] = BatchNormalization()(layers['dropout_final'])
        layers['output'] = Dense(Y_train.shape[2],  kernel_initializer='he_normal')(layers['BN'])

        model = Model(inputs=layers['ins_main']+[layers['ins_exog']],outputs=layers['output'])


        # Compile model
        sgd_fine = SGD(lr=0.015, momentum=0.9, decay=0.01, nesterov=True)
        earlystopping = EarlyStopping(monitor='val_loss',min_delta=1e-6,patience=20,verbose=0, mode='auto')
        mcp = ModelCheckpoint(dumploc+'/BestModelWeights_'+str(no)+'.hdf5',monitor='val_loss',save_best_only=True)
        model.compile(loss='mean_squared_error', optimizer=sgd_fine)
        history=model.fit(X_scaled_train_grouped+[Xexog_scaled_train], Y_train, epochs=500,
                          callbacks=[earlystopping,mcp], validation_split=0.15,
                          batch_size=32, shuffle=True, verbose=0)
        #Start refit
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')
        model.save(dumploc+'/BestModel_'+str(no)+'.hdf5')
        history = np.min(history.history['val_loss'])
        pickle.dump( history, open( dumploc+'/BestModelHist_'+str(no)+'.pkl', "wb" ))

    else:
        # To ease the computational burden the model weights as per the last
        # hyper-parameter search are used.
        model = load_model(dumploc+'/BestModel_'+str(no)+'.hdf5')
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')
        history = pickle.load( open( dumploc+'/BestModelHist_'+str(no)+'.pkl', "rb" ) )

    #Perform Forecast and Test Model
    Ypred = model.predict(X_scaled_test_grouped+[Xexog_scaled_test])
    Ypred = np.squeeze(Ypred,axis=1)


    return Ypred, history

def NN1LayerEnsemExog(X,Xexog,Y,no,params=None, refit=None, dumploc=None, A=None):
    archi = [1]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNEnsemExogGeneric, X, Y, no, params=params,
                                               refit=True, dumploc=dumploc,
                                               archi=archi, A=A, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNEnsemExogGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    archi=archi, A=A, Xexog=Xexog)

    return Ypred, val_loss


def NN2LayerEnsemExog(X,Xexog,Y,no,params=None, refit=None, dumploc=None, A=None):
    archi = [2, 1]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNEnsemExogGeneric, X, Y, no, params=params,
                                               refit=True, dumploc=dumploc,
                                               archi=archi, A=A, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNEnsemExogGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    archi=archi, A=A, Xexog=Xexog)

    return Ypred, val_loss


def NN3LayerEnsemExog(X,Xexog,Y,no,params=None, refit=None, dumploc=None, A=None):
    archi = [3, 2, 1]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNEnsemExogGeneric, X, Y, no, params=params,
                                               refit=True, dumploc=dumploc,
                                               archi=archi, A=A, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNEnsemExogGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    archi=archi, A=A, Xexog=Xexog)

    return Ypred, val_loss


def NNEnsemExog_ExogNetGeneric(X, Y, no,  dropout_u=None, l1l2penal=None, refit=None, dumploc=None, **kwargs):

    if dumploc == None:
        raise ValueError('Missing Dumploc argument')

    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, BatchNormalization
    from sklearn.utils import shuffle
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.layers.merge import concatenate
    from keras.optimizers import SGD
    from keras.models import load_model
    from keras import regularizers

    archi = kwargs['archi']
    archiexog = kwargs['archiexog']
    A = kwargs['A']
    Xexog = kwargs['Xexog']

    #Split Data for Test and Training
    X_train = X[:-12,:]
    Xexog_train = Xexog[:-12,:]
    Y_train = Y[:-12,:]


    X_test = X[-1,:].reshape(1,-1)
    Y_test = Y[-1,:].reshape(1,-1)
    Xexog_test = Xexog[-1,:].reshape(1,-1)


    #Scale the predictors for training
    Xscaler_train = MinMaxScaler(feature_range=(-1,1))
    X_scaled_train = Xscaler_train.fit_transform(X_train)
    X_scaled_train = np.expand_dims(X_scaled_train, axis=1)
    X_scaled_test = Xscaler_train.transform(X_test)
    Xexog_scaler_train = MinMaxScaler(feature_range=(-1,1))
    Xexog_scaled_train = Xexog_scaler_train.fit_transform(Xexog_train)
    Xexog_scaled_test = Xexog_scaler_train.transform(Xexog_test)
    Xexog_scaled_train = np.expand_dims(Xexog_scaled_train, axis=1)
    Xexog_scaled_test = np.expand_dims(Xexog_scaled_test, axis=1)
    Y_train = np.expand_dims(Y_train, axis=1)

    #Split X_train by group
    X_scaled_train_grouped = []
    X_scaled_test_grouped = []
    n_groups = len(np.unique(A))
    for i, group in enumerate(np.unique(A)):
        temp = X_scaled_train[:,A==group]
        X_scaled_train_grouped.append(np.expand_dims(temp, axis=1))
        temp = X_scaled_test[A==group].reshape(1,-1)
        X_scaled_test_grouped.append(np.expand_dims(temp, axis=1))


    # define base model
    tf.set_random_seed(no)
    np.random.seed(no)

    if refit:
        n = len(archi)
        nexog = len(archiexog)
        layers = dict()

        #main model
        for i in range(n+1):
            if i == 0:
                layers['ins_main'] = [Input(shape=(1,X_scaled_train_grouped[i].shape[2])) for i in range(n_groups)]
            elif i == 1:
                layers['dropout'+str(i)] = [Dropout(dropout_u)(input_tensor) for input_tensor in layers['ins_main']]
                layers['hiddens'+str(i)] = [Dense(archi[i-1], activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(l1l2penal))(input_tensor) for input_tensor in layers['dropout'+str(i)]]
            elif i > 1 & i <=n:
                layers['dropout'+str(i)] = [Dropout(dropout_u)(input_tensor) for input_tensor in layers['hiddens'+str(i-1)]]
                layers['hiddens'+str(i)] = [Dense(archi[i-1], activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(l1l2penal))(input_tensor) for input_tensor in layers['dropout'+str(i)]]

        #exog model
        for i in range(nexog+1):
            if i == 0:
                layers['ins_exog'] = Input(shape=(1,Xexog_scaled_train.shape[2]))
            elif i == 1:
                layers['dropoutexog'+str(i)] = Dropout(dropout_u)(layers['ins_exog'])
                layers['hiddensexog'+str(i)] = Dense(archiexog[i-1], activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(l1l2penal))(layers['dropoutexog'+str(i)])
            elif i > 1 & i <=nexog:
                layers['dropoutexog'+str(i)] = Dropout(dropout_u)(layers['hiddensexog'+str(i-1)])
                layers['hiddensexog'+str(i)] = Dense(archiexog[i-1], activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(l1l2penal))(layers['dropoutexog'+str(i)])

        # merge models
        layers['merge'] = concatenate(layers['hiddens'+str(n)])
        layers['merge1'] = concatenate([layers['merge'], layers['hiddensexog'+str(nexog)]])
        layers['dropout_final'] = Dropout(dropout_u)(layers['merge1'])
        layers['BN'] = BatchNormalization()(layers['dropout_final'])
        layers['output'] = Dense(Y_train.shape[2],  kernel_initializer='he_normal')(layers['BN'])


        model = Model(inputs=layers['ins_main']+[layers['ins_exog']],outputs=layers['output'])

        # Compile model
        sgd_fine = SGD(lr=0.02, momentum=0.9, decay=0.01, nesterov=True)
        earlystopping = EarlyStopping(monitor='val_loss',min_delta=1e-6,patience=20,verbose=0, mode='auto')
        mcp = ModelCheckpoint(dumploc+'/BestModelWeights_'+str(no)+'.hdf5',monitor='val_loss',save_best_only=True)
        model.compile(loss='mean_squared_error', optimizer=sgd_fine)
        history=model.fit(X_scaled_train_grouped+[Xexog_scaled_train], Y_train, epochs=500,
                          callbacks=[earlystopping,mcp], validation_split=0.15,
                          batch_size=32, shuffle=True, verbose=0)

        #Start refit
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')
        model.save(dumploc+'/BestModel_'+str(no)+'.hdf5')
        history = np.min(history.history['val_loss'])
        pickle.dump( history, open( dumploc+'/BestModelHist_'+str(no)+'.pkl', "wb" ))

    else:
        # To ease the computational burden the model weights as per the last
        # hyper-parameter search are used.
        model = load_model(dumploc+'/BestModel_'+str(no)+'.hdf5')
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')
        history = pickle.load( open( dumploc+'/BestModelHist_'+str(no)+'.pkl', "rb" ) )

    #Perform Forecast and Test Model
    Ypred = model.predict(X_scaled_test_grouped+[Xexog_scaled_test])
    Ypred = np.squeeze(Ypred,axis=1)


    return Ypred, history

def NN1LayerEnsemExog_ExogNet(X,Xexog,Y, no, params=None, refit=None, dumploc=None, A=None):
    archi = [1]
    archiexog = [3,3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNEnsemExog_ExogNetGeneric, X, Y, no, params=params,
                                               refit=True, dumploc=dumploc,
                                               archi=archi, archiexog=archiexog, A=A, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNEnsemExog_ExogNetGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    archi=archi, archiexog=archiexog, A=A, Xexog=Xexog)

    return Ypred, val_loss



def NN2LayerEnsemExog_ExogNet(X,Xexog,Y,no,params=None, refit=None, dumploc=None, A=None):
    archi = [2,1]
    archiexog = [3,3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNEnsemExog_ExogNetGeneric, X, Y, no, params=params,
                                               refit=True, dumploc=dumploc,
                                               archi=archi, archiexog=archiexog, A=A, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNEnsemExog_ExogNetGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    archi=archi, archiexog=archiexog, A=A, Xexog=Xexog)

    return Ypred, val_loss


def NN3LayerEnsemExog_ExogNet(X,Xexog,Y,no,params=None, refit=None, dumploc=None, A=None):
    archi = [3,2,1]
    archiexog = [3,3]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNEnsemExog_ExogNetGeneric, X, Y, no, params=params,
                                               refit=True, dumploc=dumploc,
                                               archi=archi, archiexog=archiexog, A=A, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNEnsemExog_ExogNetGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    archi=archi, archiexog=archiexog, A=A, Xexog=Xexog)

    return Ypred, val_loss



