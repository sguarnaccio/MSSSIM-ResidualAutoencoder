# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 19:36:13 2018

@author: sebastian
"""
from __future__ import print_function

from keras.optimizers import Adam
from keras.layers import Input
from keras.utils import plot_model, multi_gpu_model
from sklearn.model_selection import train_test_split
import numpy as np
from keras.datasets import cifar10
from matplotlib import pyplot as plt

from models import generate_model
from networkUtils import ResidualCalculation, lossMSSSIML1_V3, binarizationLayer, tf_binarization

from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.framework import ops                                                          #Actually needed to load the model with custom objects that relies on them
import tensorflow as tf                                                                              #Actually needed to load the model with custom objects that relies on them
from keras.models import load_model
from keras.losses import mean_squared_error

import argparse
import os

loss_functions = ['lossMSSSIML1_V3', 'mean_squared_error', 'mean_absolute_error']


def main(args):
    
    
    # Create the model directory if does not exist
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        
    # Create the model's weights directory if does not exist
    if not os.path.exists(args.weights_path):
        os.makedirs(args.weights_path)
        
    # Create the model's training plots directory if does not exist
    if not os.path.exists(args.trainPlots_path):
        os.makedirs(args.trainPlots_path)
    
    multiGpu = args.multiGpu                        #flag to indicate if I want to distribute the training between several GPUs
    batch_size = args.batch_size                    #64 for residual model, 256 for the simple autoencoder model
    epochs = args.num_epochs
    inChannel = 3
    x, y = args.patch_size, args.patch_size
    input_img = Input(shape = (x, y, inChannel))
    bitsPerPixel = args.bitsPerPixel
    steps = args.num_blocks                         #number of blocks for the residual autoencoder model
    residual = args.residual                        #flag that indicates if I want to train a single autoencoder model or a residual autoencoder model
    loadWeights = args.loadWeights                  #flag to load the weights of the best iteration
    loadModel = args.loadModel                      #if the model was saved instead of loading the weights I an just load the entire model
    loadSuccess = False                             #flag that indicates if the model was loaded succesfully
    
    plotName = args.modelName
    modelName = args.model_path + args.modelName + '.h5'
    gpusAvail = args.gpusAvail
    stop_patience = args.stop_patience
    learningRate = args.learning_rate
    lrDecay = args.lr_decay
    colorMap = args.colorMap
    
    if args.lossFunction in loss_functions:
        if args.lossFunction == 'lossMSSSIML1_V3':
            if residual == True:
                lossF = [mean_squared_error for i in range(steps)]
                lossF.append(lossMSSSIML1_V3)
            else:
                lossF = lossMSSSIML1_V3
        else:
            lossF = args.lossFunction
    else:
        lossF = lossMSSSIML1_V3
        
        
    
        
    
    
    """ MODEL GENERATION """
    try:
        if loadModel == True:
            if residual == True:
                filepath = args.weights_path + "weights-improvement-ResAutoencoder-best.hdf5"
                if args.lossFunction == 'lossMSSSIML1_V3':
                    lossWeights = [0 for i in range(steps)]
                    lossWeights.append(1) 
                else:
                    lossWeights = [0 for i in range(steps+1)]
                convmodel = load_model(modelName, custom_objects={'ResidualCalculation':ResidualCalculation, 'lossMSSSIML1_V3':lossMSSSIML1_V3, 'binarizationLayer':binarizationLayer, 'tf_binarization':tf_binarization})
            else:
                filepath = args.weights_path + "weights-improvement-Autoencoder-best.hdf5"
                convmodel = load_model(modelName, custom_objects={'lossMSSSIML1_V3':lossMSSSIML1_V3, 'binarizationLayer':binarizationLayer, 'tf_binarization':tf_binarization})
            print('Model Loaded')
            loadSuccess = True
        else:
            if residual == True:
                filepath = args.weights_path + "weights-improvement-ResAutoencoder-best.hdf5"
                convmodel = generate_model(input_img, colorMap, residual = 1, residualBlocks = steps, bitsPerPixel = bitsPerPixel)
                if args.lossFunction == 'lossMSSSIML1_V3':
                    lossWeights = [0 for i in range(steps)]
                    lossWeights.append(1)
                else:
                    #lossWeights = [0 for i in range(steps+1)]
                    lossWeights = [0 for i in range(steps)]
                    lossWeights.append(1)
                plot_model(convmodel, to_file = args.trainPlots_path + 'ResAutoencoder.png',show_shapes=True)
            else:
                filepath = args.weights_path + "weights-improvement-Autoencoder-best.hdf5"
                encoderModel, decoderModel, convmodel = generate_model(input_img, colorMap, residual = 0, bitsPerPixel = bitsPerPixel)
                plot_model(encoderModel, to_file = args.trainPlots_path + 'encoderModel.png',show_shapes=True)
                plot_model(decoderModel, to_file = args.trainPlots_path + 'decoderModel.png',show_shapes=True)
                plot_model(convmodel, to_file = args.trainPlots_path + 'Autoencoder.png',show_shapes=True)
    
    except Exception:
        print('Fail to load Model')
        loadSuccess = False
        
        if residual == True:
            filepath = args.weights_path + "weights-improvement-ResAutoencoder-best.hdf5"
            convmodel = generate_model(input_img, colorMap, residual = 1, residualBlocks = steps, bitsPerPixel = bitsPerPixel)
            if args.lossFunction == 'lossMSSSIML1_V3':
                    lossWeights = [0 for i in range(steps)]
                    lossWeights.append(1)
            else:
                lossWeights = [0 for i in range(steps+1)]
            plot_model(convmodel, to_file = args.trainPlots_path + 'ResAutoencoder.png',show_shapes=True)
        else:
            filepath = args.weights_path + "weights-improvement-Autoencoder-best.hdf5"
            encoderModel, decoderModel, convmodel = generate_model(input_img, colorMap, residual = 0, bitsPerPixel = bitsPerPixel)
            plot_model(encoderModel, to_file = args.trainPlots_path + 'encoderModel.png',show_shapes=True)
            plot_model(decoderModel, to_file = args.trainPlots_path + 'decoderModel.png',show_shapes=True)
            plot_model(convmodel, to_file = args.trainPlots_path + 'Autoencoder.png',show_shapes=True)
                
                
    if multiGpu == True:
        try:
            convmodel = multi_gpu_model(convmodel, gpus=gpusAvail)
        except Exception:
           pass 
        
    if residual == True:
        convmodel.compile(loss=lossF, optimizer = Adam(lr = learningRate, decay = lrDecay), loss_weights=lossWeights)
    else:
        convmodel.compile(loss=lossF, optimizer = Adam(lr = learningRate, decay = lrDecay))
        
    convmodel.summary()
    
    
    """ LOAD DATA """		
    (X, y), (X_test, y_test) = cifar10.load_data()
    
    
    """ PRE-PROCESSING """
    'normalizo'
    X = X.astype('float32')
    # rango de -0.9 a 0.9
    #X = ((X / 255)- 0.5) * 1.8
    X = ((X / 255))
    
    X_train, X_val, y_train, y_val = train_test_split(X, X,
                                                      test_size=0.20,
                                                      random_state=42)
    
    
    'Shuffle training data'
    perm = np.arange(len(X_train))
    np.random.shuffle(perm)
    X_train = X_train[perm]
    y_train = y_train[perm]
    
    'dimensions checking'
    print('X train:', X_train.shape)
    print('X vaalidation:', X_val.shape)
    
    
    
    """ TRAINING """
       
    if residual == True:
        #residuals ideally should be 0, the recoinstruction only should be as close as the input image
        outputY = [np.zeros_like(y_train)] * steps
        outputY.append(y_train)
        outputValidationY = [np.zeros_like(y_val)] * steps
        outputValidationY.append(y_val)
    
    
    try:
        if loadWeights == True and loadSuccess == False:
                # load weights
                #por un bug presente en Keras 2.2.0 tengo que entrenar la red al menos durante un epoch para que determine todas las dimensiones del modelo, sino la carga de los pesos falla
                'RECORDAR PONER loadWeights = FALSE la primera vez sino voy a tener un epoch extra al pedo'
                if residual == True:
                    autoencoder_train = convmodel.fit(X_train, outputY, batch_size=batch_size,epochs=1,verbose=1,validation_data=(X_val, outputValidationY))
                else:
                    convmodel.fit(X_train, y_train, batch_size=500,epochs=1,verbose=1,validation_data=(X_val, y_val))
                    print('Loading Weights')
                    convmodel.load_weights(filepath)
                
    except Exception: 
      pass
    
    # checkpoint
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto')
    if args.erlStop == True:
        # Early Stopping
        earlystp = EarlyStopping(monitor='val_loss', min_delta=0, patience=stop_patience, verbose=2, mode='auto')
        callbacks_list = [checkpoint, earlystp]
    else:
        callbacks_list = [checkpoint]
    
    if residual == True:
        autoencoder_train = convmodel.fit(X_train, outputY, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val, outputValidationY), callbacks=callbacks_list)
    else:
        autoencoder_train = convmodel.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val, y_val), callbacks=callbacks_list)
    
    
    """ TRAINING VS VALIDATION LOSS PLOT"""
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(len(autoencoder_train.history['loss']))
    
    plt.figure()
    try:
        plt.plot(epochs, loss, 'bo', label='Training loss')
    except Exception: 
      pass
    
    try:
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        
    except Exception: 
      pass
    
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(args.trainPlots_path + plotName + '_lossplot.png')
    #plt.show()
    
    #load best weights
    try:
        convmodel.load_weights(filepath)
    except Exception: 
      pass
    convmodel.save(modelName)

#=============================================================================
# - Train PARAMETERS
#=============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ==================================================================================================================
    # MODEL PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--modelName', type=str, default='resConvAutoenc',
                        help='name of the model to be trained')
    parser.add_argument('--residual', type=bool, default=False,
                        help='Set True if the model is residual, otherwise False')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='mini-batch size. If GPU memory overflows, try a smaller value')
    parser.add_argument('--bitsPerPixel', type=int, default=2,
                        help='number of bits representing the encoded patch')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='size for the patch of the input image')   #x and y
    parser.add_argument('--num_blocks', type=int, default=16,
                        help='number of blocks for residual architectures') #steps
    parser.add_argument('--loadWeights', type=bool, default=False,
                        help='Set True to load the saved weights but model was not saved')
    parser.add_argument('--loadModel', type=bool, default=False,
                        help='Set True to load the saved model to continue training')
    

    # ==================================================================================================================
    # OPTIMIZATION
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of iterations where the system sees all the data')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--erlStop', type=bool, default=False,
                        help='Set True to enable early stopping')
    parser.add_argument('--stop_patience', type=int, default=10,
                        help='Number of epochs without loss improvement')
    parser.add_argument('--multiGpu', type=bool, default=False,
                        help='Set True to distribute work over several GPUs')
    parser.add_argument('--gpusAvail', type=int, default=2,
                        help='Sets the number of GPUs available for training')
    parser.add_argument('--lossFunction', type=str, default='lossMSSSIML1_V3',
                        help='Loss function to use between: lossMSSSIML1_V3, mean_squared_error and mean_absolute_error')
    parser.add_argument('--colorMap', type=str, default='RGB',
                        help='Color map to use between: RGB, YUV and HSV')

    # ==================================================================================================================
    # SAVING & PRINTING
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--model_path', type=str, default='./saved_models/',
                        help='path were the models should be saved')
    parser.add_argument('--weights_path', type=str, default='./saved_weights/',
                        help='path were the model\'s weights should be saved')
    parser.add_argument('--trainPlots_path', type=str, default='./trainPlots/',
                        help='path were the model\'s train plots should be saved')



    #__________________________________________________________________________________________________________________
    args = parser.parse_args()
    print(args)
    main(args)
    
    




