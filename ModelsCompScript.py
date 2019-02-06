# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 15:18:20 2018

@author: sebastian
"""

from networkUtils import ResidualCalculation, lossMSSSIML1_V3, binarizationLayer, tf_binarization, getImg, yuv_conversion, yuvToRgb_conversion, hsv_conversion, hsvToRgb_conversion
from imageUtils import splitImage, jpegEncRawSize, toJPEG, getSSIM
from keras.models import load_model
from matplotlib import pyplot as plt

from keras.datasets import cifar10
import numpy as np
import pandas as pd
import gc

import argparse
import os

def main(args):
    
    # Create the directory for predicted Images if does not exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    colorMap = args.colorMap

    averageSSIM = args.averageSSIM
    
    
    
    ste = args.rec_steps
    
    
    avgJpegBitSize = 0
    
    
    rows = ['Img. %d' % x for x in range(1, 11)]
    
    
       
    isRes = [args.residual, args.residual, args.residual]
    
    
    evaluation_path = args.results_path
    
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
    
    print(models)
    
    """ LOAD DATA """		
    (X, y), (X_test, y_test) = cifar10.load_data()
    
    
    'Array contiene el indice de las imagenes y la calidad correspondiente para codificar en JPEG con una tasa de compresion equivalente a la de la red'
    aux4 = [(5447, 52), (8466, 52), (9529, 52), (9905, 52), (1452, 50), (3074, 48), (4263, 52), (1969, 50), (4107, 52), (7063, 42)]
    
    
    plt.figure(figsize=(20, 12))
    print("Test Images")
    for i in range(10):
        imgLabel = 'Img. ' + str(i + 1)
        plt.subplot(len(models) + 2, 10, i+1)
        plt.imshow(X_test[aux4[i][0]])
        plt.title(imgLabel)
    
    X_test2 = X_test.astype('float32')
   
    X_test2 = X_test2/255
    
    clust_data = np.zeros((10,len(models)+1))
    
    
    for j in range(len(models)):
        print("Loading Model")
        model=load_model(models[j], custom_objects={'ResidualCalculation':ResidualCalculation, 'lossMSSSIML1_V3':lossMSSSIML1_V3, 'binarizationLayer':binarizationLayer, 'tf_binarization':tf_binarization, 'yuv_conversion':yuv_conversion, 'yuvToRgb_conversion':yuvToRgb_conversion, 'hsv_conversion':hsv_conversion, 'hsvToRgb_conversion':hsvToRgb_conversion})
        
        print("Done Loading")
        print("Predictions")
        pred = model.predict(X_test2)
        print("Predictions Done")
        
        
        if isRes[j]:
            
            recImg = getImg(pred, ste, colorMap)
        else:
            recImg = pred * 255
            
            recImg = recImg.astype('uint8')
        
        gc.collect()
        for i in range(10):
            
            if averageSSIM == True:
                'Split image and prediction in 16x16 patches to see how well local details are preserved'
                'tf.image.ssim usa un kernell gaussiano de 11x11 por lo que los segmentos de la imagen no pueden ser mas chico q esto'
                origImgPatches = splitImage(X_test[aux4[i][0]], networkDimension = 16)
                predPatches = splitImage(recImg[aux4[i][0]], networkDimension = 16)
                patches, _, _, _ = origImgPatches.shape
                
                
                ssimPred = 0
                for k in range(patches):
                    ssim2_pred = getSSIM(origImgPatches[k], predPatches[k])
                    ssimPred = ssimPred + ssim2_pred
                    
                #ssimJPEG = ssimJPEG / patches
                ssimPred = ssimPred / patches
                
                clust_data[i][j+1] = ssimPred
                
            
            else:
                ssim2_pred = getSSIM(X_test[aux4[i][0]] , recImg[aux4[i][0]])
                
                clust_data[i][j+1] = ssim2_pred
                
            
            
            avgJpegBitSize = jpegEncRawSize(X_test[aux4[i][0]], aux4[i][1]) + avgJpegBitSize
            
            plt.subplot(len(models) + 2, 10, (j+1)*10 + i + 1)
            plt.imshow(recImg[aux4[i][0]])
            
            
    
    
    
    _, dim2, dim3, dim4 = X_test.shape
    decode= np.zeros((10,dim2, dim3, dim4))
    
    for i in range(10):
    	decode[i] = toJPEG(X_test[aux4[i][0]], aux4[i][1])
    	
    	
    decode = decode.astype('uint8')
    aux, _, _, _ = decode.shape
    for i in range(aux):
        if averageSSIM == True:
            ssimJPEG = 0
            origImgPatches = splitImage(X_test[aux4[i][0]], networkDimension = 16)
            jpegPatches = splitImage(decode[i], networkDimension = 16)
            patches, _, _, _ = jpegPatches.shape
            for k in range(patches):
                ssim_jpeg = getSSIM(origImgPatches[k], jpegPatches[k])
                ssimJPEG = ssimJPEG + ssim_jpeg
        
                
            ssimJPEG = ssimJPEG / patches
            clust_data[i][0] = ssimJPEG
            
        else:
            ssim_jpeg = getSSIM(X_test[aux4[i][0]], decode[i])
            clust_data[i][0] = ssim_jpeg
    
    
    for i in range(aux):
        plt.subplot(len(models) + 2, 10, 40 + i + 1)
        plt.imshow(decode[i])
        
        
    
    
    plt.savefig(args.results_path + 'ImgEvaluation.png')
    
    
    
    collabel=("SSIM JPEG", "L1 Autoencoder", "L2 Autoencoder", "SSIM Autoencoder")
    plt.figure(figsize=(20, 12))
    plt.axis('off')
    plt.table(cellText=clust_data,rowLabels=rows,colLabels=collabel,loc='center')
    
    df=pd.DataFrame({'rows': rows, 'SSIM JPEG': clust_data[:,0], 'L1 Autoencoder': clust_data[:,1], 'L2 Autoencoder': clust_data[:,2], 'SSIM Autoencoder':clust_data[:,3]})
    df.to_csv(path_or_buf=evaluation_path+'SSIMTable.csv', index=False)



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
    
    parser.add_argument('--residual', type=bool, default=False,
                        help='Set True if the model is residual, otherwise False')
    
    parser.add_argument('--rec_steps', type=int, default=8,
                        help='number of residuals to consider at image reconstruction')
    
    
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