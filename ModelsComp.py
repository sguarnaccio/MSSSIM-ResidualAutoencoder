# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 15:18:20 2018

@author: sebastian
"""

from networkUtils import ResidualCalculation, lossMSSSIML1_V3, zero_loss, binarizationLayer, tf_binarization, getImg, yuv_conversion, yuvToRgb_conversion, hsv_conversion, hsvToRgb_conversion
from imageUtils import splitImage, buildImage, getMetric, jpegEncRawSize, toJPEG, getSSIM, numpyToJPEGTensor
from keras.models import load_model
from matplotlib import pyplot as plt
#from keras.preprocessing import image
#from tensorflow.python.framework import ops
#import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import pandas as pd
import gc



averageSSIM = False

encode_patch = 6*6
encodePatchBpp = 8

ste = 8

imgDim = 32
imgChannels = 3
'cada pixel esta compuesto por un uint8'
imgBpp = 8 
imageBits = imgDim *imgDim * imgChannels * imgBpp

jpegQuality = 90
avgJpegBitSize = 0


rows = ['Img. %d' % x for x in range(1, 11)]

#modelos = ['resConvMSSSIML1']
modelos = ['resConvMSSSIML1_rgb4', 'resConvMSSSIML1_rgb5', 'resConvMSSSIML1_rgb2']

#'resConvMSSSIML1', 'resConvL1', 'resConvL2'
#isRes = [True, True, True]
#isRes = [False, True, True, True, True, True, True]
isRes = [True, True, True]
model_path = './saved_models/'
results_path = './PredictionResults/'
evaluation_path = './evaluationResults/'

models = []

for i in range(len(modelos)):
    models.append(model_path + modelos[i] + '.h5')

print(models)

""" LOAD DATA """		
(X, y), (X_test, y_test) = cifar10.load_data()

#TODO:Volver a estimar este array para jpeg sin subsampling del canal chroma
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
# rango de -0.9 a 0.9
#X_test2 = ((X_test2 / 255) - 0.5) * 1.8
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
        if j == 5:
            ste = 8
        elif j == 6:
            ste = 16
        recImg = getImg(pred, ste, 'RGB')
    else:
        recImg = pred * 255
        #recImg = ((pred / 1.8) + 0.5) * 255
        recImg = recImg.astype('uint8')
    
    gc.collect()
    for i in range(10):
        
        if averageSSIM == True:
            'Split image and prediction in 16x16 patches to see how well local details are preserved'
            'tf.image.ssim usa un kernell gaussiano de 11x11 por lo que los segmentos de la imagen no pueden ser mas chico q esto'
            origImgPatches = splitImage(X_test[aux4[i][0]], networkDimension = 16)
            predPatches = splitImage(recImg[aux4[i][0]], networkDimension = 16)
            patches, _, _, _ = origImgPatches.shape
            
            #ssimJPEG = 0
            ssimPred = 0
            for k in range(patches):
                ssim2_pred = getSSIM(origImgPatches[k], predPatches[k])
                #ssim_jpeg, ssim2_pred, _ = getMetric(origImgPatches[k] , predPatches[k], jpegQuality)
                #ssimJPEG = ssimJPEG + ssim_jpeg
                ssimPred = ssimPred + ssim2_pred
                
            #ssimJPEG = ssimJPEG / patches
            ssimPred = ssimPred / patches
            
            clust_data[i][j+1] = ssimPred
            #clust_data[i] = ssimJPEG, ssimPred
            'obtengo decodificacion JPEG de la imagen'
            'definiendo el parametro de calidad para la compresion jpeg obtenemos un encoding de 2320 bits, lo cual es bastante cerca respecto a los 2304 bits obtenidos por nuestra red (6 x 6 x 16 bitsperpixel x 4 etapas | dado este modelo puntual)'
            #decode = toJPEG(X_test[i], jpegQuality)
            #jpegList.append(decode)
        
        else:
            ssim2_pred = getSSIM(X_test[aux4[i][0]] , recImg[aux4[i][0]])
            #ssim_jpeg, ssim2_pred, decode = getMetric(X_test[i] , recImg[i], jpegQuality)
            clust_data[i][j+1] = ssim2_pred
            #clust_data[i] = ssim_jpeg, ssim2_pred
            #jpegList.append(decode)
        
        
        avgJpegBitSize = jpegEncRawSize(X_test[aux4[i][0]], aux4[i][1]) + avgJpegBitSize
        
        plt.subplot(len(models) + 2, 10, (j+1)*10 + i + 1)
        plt.imshow(recImg[aux4[i][0]])
        
        """
        plt.subplot(len(models) + 2, 10, (j+1)*20 + i + 1)
        plt.imshow(decode)
        """



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
    
    

plt.show()
#plt.savefig(results_path + 'ComparacionModelos.png')

compRate_net = (encode_patch * encodePatchBpp * ste) / imageBits
netCompBpp = imgBpp * compRate_net
    
avgCompRateJpeg = (avgJpegBitSize/10) / imageBits
jpegCompBpp = imgBpp * avgCompRateJpeg

collabel=("SSIM JPEG", "L1 Autoencoder", "L2 Autoencoder", "SSIM Autoencoder")
plt.figure(figsize=(20, 12))
plt.axis('off')
plt.table(cellText=clust_data,rowLabels=rows,colLabels=collabel,loc='center')

df=pd.DataFrame({'rows': rows, 'SSIM JPEG': clust_data[:,0], 'L1 Autoencoder': clust_data[:,1], 'L2 Autoencoder': clust_data[:,2], 'SSIM Autoencoder':clust_data[:,3]})
df.to_csv(path_or_buf=evaluation_path+'SSIMTable.csv', index=False)



'''

modelsL2 = model_path + 'resConvMSSSIML1_rgb5' + '.h5'
model=load_model(modelsL2, custom_objects={'ResidualCalculation':ResidualCalculation, 'lossMSSSIML1_V3':lossMSSSIML1_V3, 'binarizationLayer':binarizationLayer, 'tf_binarization':tf_binarization, 'yuv_conversion':yuv_conversion, 'yuvToRgb_conversion':yuvToRgb_conversion, 'hsv_conversion':hsv_conversion, 'hsvToRgb_conversion':hsvToRgb_conversion})
print("Done Loading")
print("Predictions")
predL2 = model.predict(X_test2)

modelsL1 = model_path + 'resConvMSSSIML1_rgb4' + '.h5'
model=load_model(modelsL1, custom_objects={'ResidualCalculation':ResidualCalculation, 'lossMSSSIML1_V3':lossMSSSIML1_V3, 'binarizationLayer':binarizationLayer, 'tf_binarization':tf_binarization, 'yuv_conversion':yuv_conversion, 'yuvToRgb_conversion':yuvToRgb_conversion, 'hsv_conversion':hsv_conversion, 'hsvToRgb_conversion':hsvToRgb_conversion})
print("Done Loading")
print("Predictions")
predL1 = model.predict(X_test2)

gc.collect()




plt.figure(figsize=(10, 4))
plt.imshow(X_test[aux4[8][0]])

ste = 3
plotIndex = 1
plt.figure(figsize=(10, 25))
for i in range (1,10):
    #ste = 2**i
    ste = ste + 1
    
    recImg = getImg(pred, ste, 'RGB')
    recImgL2 = getImg(predL2, ste, 'RGB')
    recImgL1 = getImg(predL1, ste, 'RGB')
    
    

    
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
            
        elif ste == 6:            
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
plt.show()

'''