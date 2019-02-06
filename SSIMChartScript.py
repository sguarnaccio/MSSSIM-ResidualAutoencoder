# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 20:49:17 2019
Descripcion: El script presente se encarga de cargar los dicccionarios de cada
             uno de los segmentos definidos para la tasa de compresion y hallar
             el valor promedio del indice SSIM para cada uno de estos
             
             Este indice puede obtenerse para la imagen completa o se puede
             realizar previamente la segmentacion de la imagen para ver que
             tan bien se conservan los detalles locales
@author: sebastian
"""

import numpy as np

from keras.datasets import cifar10
from keras.models import load_model
import pandas as pd
from networkUtils import ResidualCalculation, lossMSSSIML1_V3, binarizationLayer, tf_binarization, getImg, yuv_conversion, yuvToRgb_conversion, hsv_conversion, hsvToRgb_conversion
from matplotlib import pyplot as plt
import tensorflow as tf
import json
from keras import backend as K
import gc
import argparse
import os


""" LOAD DATA """		
(X, y), (X_test, y_test) = cifar10.load_data()


state_path = './jsonFiles/'


loopIndex = 0
load = True

file_name = 'jpegdic3'
file_nameNoChroma = 'jpegdic4NoChrom'


loadDataFrame = False

jpegdic1 = dict()
jpegdic2 = dict()
jpegdic3 = dict()
jpegdic4 = dict()
jpegdic5 = dict()
jpegdic6 = dict()
jpegdic7 = dict()
jpegdic8 = dict()
jpegdic9 = dict()

jpegdicNoChroma1 = dict()
jpegdicNoChroma2 = dict()
jpegdicNoChroma3 = dict()
jpegdicNoChroma4 = dict()
jpegdicNoChroma5 = dict()
jpegdicNoChroma6 = dict()
jpegdicNoChroma7 = dict()
jpegdicNoChroma8 = dict()
jpegdicNoChroma9 = dict()




getDict = dict()
case1 = {1:jpegdic1, 2:jpegdic2, 3:jpegdic3, 4:jpegdic4, 5:jpegdic5, 6:jpegdic6, 7:jpegdic7, 8:jpegdic8, 9:jpegdic9 }
case2 = {1:jpegdicNoChroma1, 2:jpegdicNoChroma2, 3:jpegdicNoChroma3, 4:jpegdicNoChroma4, 5:jpegdicNoChroma5, 6:jpegdicNoChroma6, 7:jpegdicNoChroma7, 8:jpegdicNoChroma8, 9:jpegdicNoChroma9 }


segmentIterations = len(case1) #numero de rangos en la tasa de compresion

"""
Descripcion: Funcion que devuelve el valor del indice SSIM dada una imagen, la
             cual fue sujeta a una transformacion, y la referencia correspondiente 
             sobre la cual se realizara la comparacion
"""
def getSSIM(refImg, img): #testImg y predImg deben ser las imagenes desnormalizadas, uint8 con valores de 0 a 255
    
    refImg = tf.image.convert_image_dtype(refImg, tf.uint8)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    
    'Calculate MS-SSIM for the JPEG image and the predicted image'
    ssim_calc = tf.image.ssim(refImg, img, 255)
    with tf.Session() as sess:
        ssim = sess.run(ssim_calc)
        
    K.clear_session()
    
    return ssim


"""
Descripcion: Funcion que la compresion JPEG de una imagen dada

"""
def toJPEG(testImg, cmpQuality, chrDwnSamp = True):
    
    'encode to JPEG'
    input_i = tf.placeholder(tf.uint8)
    input_j = tf.placeholder(tf.string)
    
    encoded_img = tf.image.encode_jpeg(input_i, format="rgb", quality=cmpQuality, chroma_downsampling=chrDwnSamp)
    decoded_img = tf.image.decode_jpeg(input_j, channels=3)
    
    with tf.Session() as sess:
        encode = sess.run(encoded_img, feed_dict={input_i: testImg})
        decode = sess.run(decoded_img, feed_dict={input_j: encode})
        
    K.clear_session()
    
    return decode


"""
Descripcion: Funcion que separa la imagen en segmentos dados por networkDimension
             Las dimensiones de estos segmentos no deben ser mayores a 32 ya 
             estas son las dimensiones de salida de la red autoencoder
"""
def splitImage(img, networkDimension = 32):
    
    testImgx, _, _ = img.shape
        
    numOfPatchesInRow = testImgx // networkDimension
    
    patches = np.zeros((numOfPatchesInRow * numOfPatchesInRow,networkDimension,networkDimension, 3), dtype=np.uint8)
    for i in range(numOfPatchesInRow):
        for j in range(numOfPatchesInRow):
            #we want to extract a patch of 32x32 dimension 
            patches[i*numOfPatchesInRow + j] = img[networkDimension*i:networkDimension*(i+1),networkDimension*j:networkDimension*(j+1)] # Change
    return patches


def main(args):
    
    # Create the directory for predicted Images if does not exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    
    models = []
    
    if args.modelName1 != 'None':
        modelFile1 = args.model_path + args.modelName1 + '.h5'
        models.append(modelFile1)
    else:
        #al menos tiene q haber un modelo
        return -1
    
    if args.modelName2 != 'None':
        modelFile2 = args.model_path + args.modelName2 + '.h5'
        models.append(modelFile2)
        
    if args.modelName1 != 'None':
        modelFile3 = args.model_path + args.modelName2 + '.h5'
        models.append(modelFile3)
    
    
    colorMap = args.colorMap
    
    
    
    
    
    loadDict = dict()
    if load == True:
        
        for k in range(1, segmentIterations + 1): 			
            getDict = case1[k]
    		
            
            with open(state_path + file_name + str(k) + '.json', 'r') as fp:
                loadDict = json.load(fp)
    
            for key in loadDict.keys():
                intKey = int(key)
                getDict[intKey] = loadDict[key]
                
            
            getDict = case2[k]
            with open(state_path + file_nameNoChroma + str(k) + '.json', 'r') as fp:
                loadDict = json.load(fp)
    
            for key in loadDict.keys():
                intKey = int(key)
                getDict[intKey] = loadDict[key]
    
    
            
        print("Dict Loaded!")
            
            
            
            
    """MODELTEST"""
    
    
    
    averageSSIM = args.averageSSIM
    avgRange = 10
    
    encode_patch = args.encPatch * args.encPatch
    encodePatchBpp = args.encPatchBpp
    
    imgDim = 32
    imgChannels = 3
    'cada pixel esta compuesto por un uint8'
    imgBpp = 8 
    imageBits = imgDim *imgDim * imgChannels * imgBpp
    
    
        
    X_test2 = X_test.astype('float32')
    X_test2 = X_test2/255  
        
    
    
    
    
    
    
    
    if args.modelName1 != 'None':
        #El primer modelo deberia ser el que usa el MSSSIM
        model=load_model(modelFile1, custom_objects={'ResidualCalculation':ResidualCalculation, 'lossMSSSIML1_V3':lossMSSSIML1_V3, 'binarizationLayer':binarizationLayer, 'tf_binarization':tf_binarization, 'yuv_conversion':yuv_conversion, 'yuvToRgb_conversion':yuvToRgb_conversion, 'hsv_conversion':hsv_conversion, 'hsvToRgb_conversion':hsvToRgb_conversion})
        print("Done Loading")
        print("Predictions")
        pred = model.predict(X_test2)
    
    
        #Segundo modelo deberia ser el que usa L1
        model=load_model(modelFile2, custom_objects={'ResidualCalculation':ResidualCalculation, 'lossMSSSIML1_V3':lossMSSSIML1_V3, 'binarizationLayer':binarizationLayer, 'tf_binarization':tf_binarization, 'yuv_conversion':yuv_conversion, 'yuvToRgb_conversion':yuvToRgb_conversion, 'hsv_conversion':hsv_conversion, 'hsvToRgb_conversion':hsvToRgb_conversion})
        print("Done Loading")
        print("Predictions")
        predL2 = model.predict(X_test2)
    
    if args.modelName3 != 'None':
        #Tercer modelo deberia ser el que usa L2
        model=load_model(modelFile3, custom_objects={'ResidualCalculation':ResidualCalculation, 'lossMSSSIML1_V3':lossMSSSIML1_V3, 'binarizationLayer':binarizationLayer, 'tf_binarization':tf_binarization, 'yuv_conversion':yuv_conversion, 'yuvToRgb_conversion':yuvToRgb_conversion, 'hsv_conversion':hsv_conversion, 'hsvToRgb_conversion':hsvToRgb_conversion})
        print("Done Loading")
        print("Predictions")
        predL1 = model.predict(X_test2)
    
    del model
    K.clear_session()
    gc.collect()
    
    
    plotIndex = 1
    ste = 3
    steRange = len(case1)
    #steRange = 9 #9 to include ste up to 12
    
    jpeg_SSIMNoChromDownS = np.zeros((steRange,))
    jpeg_SSIM = np.zeros((steRange,))
    
    autenc_SSIM = np.zeros((steRange,))
    autenc_L2 = np.zeros((steRange,))
    autenc_L1 = np.zeros((steRange,))
    bpp = np.zeros((steRange,))
    
    
    plt.figure(figsize=(10, 25))
    for i in range (1,steRange+1):
        
        ste = ste + 1
        
        print("ste:",  ste)
        
        recImg = getImg(pred, ste, colorMap)
        recImgL2 = getImg(predL2, ste, colorMap)
        recImgL1 = getImg(predL1, ste, colorMap)
        
        
        auxdict = case1[i]
        auxdictNoChromSub = case2[i]
    
        #Si average SSIM =True en vez de calcular el indice para la imagen completa,
        #esta la separo en segmentos sobre los cuales voy a calcular el SSIM para
        #poder evaluar que tan bien se conservan los detalles locales 
        if averageSSIM == True:
             
            
            ssim_jpeg = 0
            ssim_jpegNoChromDownS = 0
            ssim_pred = 0
            ssim_L2 = 0
            ssim_L1 = 0
            for j in range(avgRange):
                #Separo la imagen y la prediccion de la red en segmentos de 16x16
                #tf.image.ssim usa un kernell gaussiano de 11x11 por lo que los segmentos de la imagen no pueden ser mas chico q esto
                origImgPatches = splitImage(X_test[j], networkDimension = 16)
                predPatches = splitImage(recImg[j], networkDimension = 16)
                predL2Patches = splitImage(recImgL2[j], networkDimension = 16)
                predL1Patches = splitImage(recImgL1[j], networkDimension = 16)
                
                decode = toJPEG(X_test[j], auxdict[j])
                decode = decode.astype('uint8')
                decode = splitImage(decode, networkDimension = 16)
                
                
                #JPEG No Chroma Down sampling encoding
                decodeNoChromDwnSamp = toJPEG(X_test[j], auxdictNoChromSub[j], chrDwnSamp = False)
                decodeNoChromDwnSamp = decodeNoChromDwnSamp.astype('uint8')
                decodeNoChromDwnSamp = splitImage(decodeNoChromDwnSamp, networkDimension = 16)
                
                
                patches, _, _, _ = origImgPatches.shape
                
                
                ssimPredPatchAvg = 0
                ssimPredL2PatchAvg = 0
                ssimPredL1PatchAvg = 0
                ssimJpegPatchAvg = 0
                ssimJpegNoChrDwnSampPatchAvg = 0
                
                    
                ssimPredPatchAvg = getSSIM(origImgPatches , predPatches)
                ssimPredL2PatchAvg = getSSIM(origImgPatches , predL2Patches)
                ssimPredL1PatchAvg = getSSIM(origImgPatches , predL1Patches)
                
                ssimJpegPatchAvg = getSSIM(origImgPatches, decode)
                ssimJpegNoChrDwnSampPatchAvg = getSSIM(origImgPatches, decodeNoChromDwnSamp)
                    
                ssim_jpeg = ssim_jpeg + np.mean(ssimJpegPatchAvg)
                ssim_jpegNoChromDownS = ssim_jpegNoChromDownS + np.mean(ssimJpegNoChrDwnSampPatchAvg)
                ssim_pred = ssim_pred + np.mean(ssimPredPatchAvg)
                ssim_L2 = ssim_L2 + np.mean(ssimPredL2PatchAvg)
                ssim_L1 = ssim_L1 + np.mean(ssimPredL1PatchAvg)
    
                
                if j % 1000 == 0:
                    print("Image: ", j)
                
                
            ssim_jpeg = ssim_jpeg / avgRange
            ssim_jpegNoChromDownS = ssim_jpegNoChromDownS / avgRange
            ssim_pred = ssim_pred / avgRange
            ssim_L2 = ssim_L2 / avgRange
            ssim_L1 = ssim_L1 / avgRange
                
            print("#####################")
            print("ssim_jpeg pred:",  ssim_jpeg)
            print("ssim_jpegNoChrom pred:",  ssim_jpegNoChromDownS)
            print("ssim_pred:",  ssim_pred)
            print("ssim_L2:",  ssim_L2)
            print("ssim_L1:",  ssim_L1)
            print("#####################")
            
            jpeg_SSIMNoChromDownS[i-1] = ssim_jpegNoChromDownS
            jpeg_SSIM[i-1] = ssim_jpeg
            autenc_SSIM[i-1] = ssim_pred
            autenc_L2[i-1] = ssim_L2
            autenc_L1[i-1] = ssim_L1
            
        
        else:
    
            ssim_jpeg = 0
            ssim_jpegNoChromDownS = 0
            ssim_pred = 0
            ssim_L2 = 0
            ssim_L1 = 0
            for j in range(avgRange):
                #key = int(keys[j])
                decode = toJPEG(X_test[j], auxdict[j])
                decode = decode.astype('uint8')
                ssim_jpeg += getSSIM(X_test[j], decode)
                
                'JPEG No Chroma Subsampling encoding'
                decode = toJPEG(X_test[j], auxdictNoChromSub[j], chrDwnSamp = False)
                decode = decode.astype('uint8')
                ssim_jpegNoChromDownS += getSSIM(X_test[j], decode)
                
                
                if j % 1000 == 0:
                    print("Image: ", j)
    
            gc.collect()    
            
            
            ssim_pred = getSSIM(X_test, recImg)
            ssim_pred = np.mean(ssim_pred)
            ssim_L2 = getSSIM(X_test, recImgL2)
            ssim_L2 = np.mean(ssim_L2)
            ssim_L1 = getSSIM(X_test, recImgL1)
            ssim_L1 = np.mean(ssim_L1)
            
                
            
            ssim_jpeg = ssim_jpeg / avgRange
            ssim_jpegNoChromDownS = ssim_jpegNoChromDownS / avgRange
            
            
            print("#####################")
            print("ssim_jpeg pred:",  ssim_jpeg)
            print("ssim_jpegNoChrom pred:",  ssim_jpegNoChromDownS)
            print("ssim_pred:",  ssim_pred)
            print("ssim_L2:",  ssim_L2)
            print("ssim_L1:",  ssim_L1)
            print("#####################")
            
            jpeg_SSIMNoChromDownS[i-1] = ssim_jpegNoChromDownS
            jpeg_SSIM[i-1] = ssim_jpeg
            autenc_SSIM[i-1] = ssim_pred
            autenc_L2[i-1] = ssim_L2
            autenc_L1[i-1] = ssim_L1
    
        compRate_net = (encode_patch * encodePatchBpp * ste) / imageBits
        netCompBpp = imgBpp * compRate_net
        bpp[i-1] = netCompBpp
        
        if ste%2==0:
            plt.subplot(len(models) + 3, 5, plotIndex)
            plt.imshow(recImg[4107])
            
            plt.subplot(len(models) + 3, 5, plotIndex+5)
            plt.imshow(recImgL1[4107])
            
            plt.subplot(len(models) + 3, 5, plotIndex+10)
            plt.imshow(recImgL2[4107])
            
    
            if ste == 4:
                'JPEG'
                decode = toJPEG(X_test[4107], 16)
                plt.subplot(len(models) + 3, 5, plotIndex+15)
                plt.imshow(decode)
                
                'JPEG No Chroma Down samplig'
                decode = toJPEG(X_test[4107], 13, chrDwnSamp = False)
                plt.subplot(len(models) + 3, 5, plotIndex+20)
                plt.imshow(decode)
            if ste == 6:
                'JPEG'
                decode = toJPEG(X_test[4107], 34)
                plt.subplot(len(models) + 3, 5, plotIndex+15)
                plt.imshow(decode)
                
                'JPEG No Chroma Down samplig'
                decode = toJPEG(X_test[4107], 24, chrDwnSamp = False)
                plt.subplot(len(models) + 3, 5, plotIndex+20)
                plt.imshow(decode)
            elif ste == 8:
                decode = toJPEG(X_test[4107], 53)
                plt.subplot(len(models) + 3, 5, plotIndex+15)
                plt.imshow(decode)
                
                'JPEG No Chroma Down samplig'
                decode = toJPEG(X_test[4107], 39, chrDwnSamp = False)
                plt.subplot(len(models) + 3, 5, plotIndex+20)
                plt.imshow(decode)
            elif ste == 10:
                decode = toJPEG(X_test[4107], 70)
                plt.subplot(len(models) + 3, 5, plotIndex+15)
                plt.imshow(decode)
                
                'JPEG No Chroma Down samplig'
                decode = toJPEG(X_test[4107], 55, chrDwnSamp = False)
                plt.subplot(len(models) + 3, 5, plotIndex+20)
                plt.imshow(decode)
            elif ste == 12:
                decode = toJPEG(X_test[4107], 79)
                plt.subplot(len(models) + 3, 5, plotIndex+15)
                plt.imshow(decode)
                
                'JPEG No Chroma Down samplig'
                decode = toJPEG(X_test[4107], 68, chrDwnSamp = False)
                plt.subplot(len(models) + 3, 5, plotIndex+20)
                plt.imshow(decode)
    
            plotIndex+=1
    plt.savefig(args.results_path + 'ModelComp.png')
    
    plt.figure(figsize=(10, 4))
    plt.imshow(X_test[4107])
    plt.savefig(args.results_path + 'OriginalIMage.png')
    
    if loadDataFrame == True:
        df = pd.read_csv(state_path + 'CompDataFrame.csv', keep_default_na=False, na_values=[""])
        
    df=pd.DataFrame({'bpp': bpp, 'autenc_SSIM': autenc_SSIM, 'autenc_L1': autenc_L1, 'autenc_L2': autenc_L2, 'jpeg_SSIM':jpeg_SSIM, 'jpeg_SSIMNoChromDownS':jpeg_SSIMNoChromDownS})
    df.to_csv(path_or_buf=state_path + 'CompDataFrame.csv', index=False)
    plt.figure(figsize=(20, 12))
    plt.plot( 'bpp', 'jpeg_SSIM', data=df, marker='', color='blue', linewidth=2)
    plt.plot( 'bpp', 'jpeg_SSIMNoChromDownS', data=df, marker='', color='orange', linewidth=2)
    plt.plot( 'bpp', 'autenc_SSIM', data=df, marker='', color='olive', linewidth=2)
    plt.plot( 'bpp', 'autenc_L1', data=df, marker='', color='grey', linewidth=2, linestyle='dashed')
    plt.plot( 'bpp', 'autenc_L2', data=df, marker='', color='red', linewidth=2, linestyle='dashed')
    plt.tick_params(labelsize=16)
    plt.ylabel('SSIM', fontsize=16)
    plt.xlabel('bpp', fontsize=16)
    plt.legend(prop={'size': 16})
    plt.grid()
    
    plt.savefig(args.results_path + 'SSIMChart.png')




#=============================================================================
# - SSIMChartScript PARAMETERS
#=============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ==================================================================================================================
    # MODEL PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--modelName1', type=str, default='resConvAutoenc',
                        help='name of the first model to be loaded')
    
    parser.add_argument('--modelName2', type=str, default='None',
                        help='name of the Second model to be loaded')
    
    parser.add_argument('--modelName3', type=str, default='None',
                        help='name of the third model to be loaded')
    
    parser.add_argument('--averageSSIM', type=bool, default=False,
                        help='Set True to split image in segments to calculate SSIM')
    parser.add_argument('--encPatch', type=int, default=6,
                        help='Patch dimension for the autoencoder Bottleneck')
    
    parser.add_argument('--encPatchBpp', type=int, default=8,
                        help='channel dimension for the bottleneck')
    
    parser.add_argument('--colorMap', type=str, default='RGB',
                        help='Color map to use between: RGB, YUV and HSV')


    # ==================================================================================================================
    # LOADING  & SAVING Parameters
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--model_path', type=str, default='./saved_models/',
                        help='path were the models should be located')
    parser.add_argument('--img_path', type=str, default='./test_Imgs/',
                        help='path were the images should be located')

    parser.add_argument('--results_path', type=str, default='./evaluationResults/',
                        help='path were the models results should be located')
    
    
    


    #__________________________________________________________________________________________________________________
    args = parser.parse_args()
    print(args)
    main(args)




