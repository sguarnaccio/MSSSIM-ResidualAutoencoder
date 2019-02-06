# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 12:35:37 2018

@author: sebastian
"""

from __future__ import print_function
from keras.models import Model

from keras.layers import Conv2D, Conv2DTranspose, Input, Lambda, add, PReLU, LeakyReLU
from keras import backend as K

from networkUtils import ResidualCalculation, binarizationLayer, output_of_lambda, addPredictions, yuv_conversion, yuvToRgb_conversion, hsv_conversion, hsvToRgb_conversion
import tensorflow as tf
import networkUtils




decOutputs = []


color_maps = ['RGB', 'YUV', 'HSV']
rgbMap = True
mapConv = yuv_conversion
mapConvInv = yuvToRgb_conversion

#para este caso, como no calculo resduos, no encadeno varias etapas, solo uso 1
def generate_model(input_img, colorMap, residual = 1, residualBlocks = 16, bitsPerPixel = 2):
    if residual == 1:
        return residualAutoencoder(input_img, colorMap, steps = residualBlocks, bitsPerPixel = bitsPerPixel)
    else:
        return autoencoder(input_img, colorMap, bitsPerPixel = bitsPerPixel)
    


def encoder(input_img, bits = 32):
	#encoder
	#input = 32 x 32 x 3 (wide and thin)
    
    conv1 = Conv2D(64, (3, 3), strides=2)(input_img) #15 x 15 x 64
    conv1 = PReLU(shared_axes=[1,2])(conv1)
    conv2 = Conv2D(256, (3, 3), strides=1)(conv1) #13 x 13 x 256
    conv2 = PReLU(shared_axes=[1,2])(conv2)
    conv3 = Conv2D(512, (3, 3), strides=2)(conv2) #6 x 6 x 512 (small and thick)
    conv3 = PReLU(shared_axes=[1,2])(conv3)
    	
    #implement kernel size of 1x1 (networks in networks)
    preBinary = Conv2D(bits, (1, 1), activation='tanh', strides=1, name='compressedPatch')(conv3) #6 x 6 x bitsPerPixel
    
    return preBinary
    
	
def decoder(input_code):

    conv4 = Conv2DTranspose(128, (3, 3), strides=1)(input_code) #8 x 8 x 128
    conv4 = PReLU(shared_axes=[1,2])(conv4)
    conv5 = Conv2DTranspose(512, (2, 2), strides=2)(conv4) # 16 x 16 x 512
    conv5 = PReLU(shared_axes=[1,2])(conv5)
    conv6 = Conv2DTranspose(512, (2, 2), strides=2)(conv5) # 32 x 32 x 512
    conv6 = PReLU(shared_axes=[1,2])(conv6)
	
	#RGB conversion
    rgb_conv1 = Conv2D(3, (1, 1), activation='tanh', padding='same')(conv6) # 32 x 32 x 3
	
    return rgb_conv1

def autoencoder(inputpatch, colorMap, bitsPerPixel = 32):
    
    #encoder
    #input = 32 x 32 x 3
    if colorMap in color_maps:
        if colorMap == 'RGB':
            rgbMap = True
        else:
            rgbMap = False
            if colorMap == 'YUV':
                mapConv = yuv_conversion
                mapConvInv = yuvToRgb_conversion
            elif colorMap == 'HSV':
                mapConv = hsv_conversion
                mapConvInv = hsvToRgb_conversion
    else:
        rgbMap = True
    
    if rgbMap == False:
        inputpatch = Lambda(mapConv, output_shape=output_of_lambda)(inputpatch)
    
    
    binary_code = encoder(inputpatch, bits = bitsPerPixel)
    encModel = Model(inputpatch, binary_code)
    
    binCodeDimension = encModel.output_shape
    binCodeDimension = binCodeDimension[1:]
    binCode = Input(shape = binCodeDimension)
    
    #binarizacion
    bincode = Lambda(binarizationLayer, output_shape=output_of_lambda)(binCode)
    binmodel = Model(binCode, bincode)
    binmodel_output = binmodel(encModel.output)
    tempModel = Model(inputs=encModel.input, outputs=binmodel_output)
    
    
    #decoder
    #define code dimensions
    codeDimension = encModel.get_layer('compressedPatch').output_shape
    codeDimension = codeDimension[1:]
    
    #define decoder input
    code = Input(shape = codeDimension)
    
    
    prediction = decoder(code)
    if rgbMap == False:
        prediction = Lambda(mapConvInv, output_shape=output_of_lambda)(prediction)
        
    decModel= Model(code, prediction)
    
    #defino el modelo completo para el entrenamiento basado en los submodelos decode y encode
    finalmodel_output = decModel(tempModel.output)
    full_model  = Model(inputs=tempModel.input, outputs=finalmodel_output)
    return encModel, decModel, full_model





def residualAutoencoder(inputpatch, colorMap, steps = 16, bitsPerPixel = 2):
    
    if colorMap in color_maps:
        if colorMap == 'RGB':
            rgbMap = True
        else:
            rgbMap = False
            if colorMap == 'YUV':
                mapConv = yuv_conversion
                mapConvInv = yuvToRgb_conversion
            elif colorMap == 'HSV':
                mapConv = hsv_conversion
                mapConvInv = hsvToRgb_conversion
    else:
        rgbMap = True
    
    inputPatch = inputpatch
    
    if rgbMap == False:
        inputPatch = Lambda(mapConv, output_shape=output_of_lambda)(inputPatch)
    
    
    for i in range(steps):
        #encoder
        #input = 32 x 32 x 3 (wide and thin)
        
        conv1 = Conv2D(64, (3, 3), strides=2)(inputPatch) #15 x 15 x 64
        conv1 = PReLU(shared_axes=[1,2])(conv1)
        conv2 = Conv2D(256, (3, 3), strides=1)(conv1) #13 x 13 x 256
        conv2 = PReLU(shared_axes=[1,2])(conv2)
        conv3 = Conv2D(512, (3, 3), strides=2)(conv2) #6 x 6 x 512 (small and thick)
        conv3 = PReLU(shared_axes=[1,2])(conv3)
        
        #implemento kernel size de 1x1 (networks in networks)
        #uso tanh para generar una salida que este entre -1 y -1 la cual luego sera llevada estrictamente al set {-1;1} con la capa de binarizacion
        preBinary = Conv2D(bitsPerPixel, (1, 1), activation='tanh', strides=1)(conv3) #6 x 6 x bitsPerPixel
        binCode = Lambda(binarizationLayer, output_shape=output_of_lambda)(preBinary)
        
        
        #decoder
        conv4 = Conv2DTranspose(128, (3, 3), strides=1)(binCode) #8 x 8 x 128
        conv4 = PReLU(shared_axes=[1,2])(conv4)
        conv5 = Conv2DTranspose(512, (2, 2), strides=2)(conv4) # 16 x 16 x 512
        conv5 = PReLU(shared_axes=[1,2])(conv5)
        conv6 = Conv2DTranspose(512, (2, 2), strides=2)(conv5) # 32 x 32 x 512
        conv6 = PReLU(shared_axes=[1,2])(conv6)
        	
        #RGB conversion
        #uso relu porque la salida normalizada de la imagen deberia estar entre 0 y 1 idealmente
        rgb_conv1 = Conv2D(3, (1, 1), activation='relu', padding='same')(conv6) # 32 x 32 x 3
        
        networkUtils.steps = steps
        #Calculo las perdidas del residuo aca porque no quiero usar la misma lossfunction que la usada para la salida general (Imagen reconstruida). 
        res = ResidualCalculation()([inputPatch,rgb_conv1])
        
        decOutputs.append(rgb_conv1)
        

        inputPatch = res
    
    
    z = add(decOutputs)
    if rgbMap == False:
        z = Lambda(mapConvInv, output_shape=output_of_lambda)(z)
    
    
    decOutputs.append(z)
    
    
    return Model(inputpatch, decOutputs)
