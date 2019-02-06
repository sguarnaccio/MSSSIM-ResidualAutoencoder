# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 22:03:22 2018

@author: sebastian
"""

from networkUtils import ResidualCalculation, lossMSSSIML1_V3, getImg, zero_loss, binarizationLayer, tf_binarization, yuv_conversion, yuvToRgb_conversion, hsv_conversion, hsvToRgb_conversion
from imageUtils import splitImage, buildImage, getMetric
from keras.models import load_model
from matplotlib import pyplot as plt
from keras.preprocessing import image
from tensorflow.python.framework import ops
import tensorflow as tf

import argparse
import os


def main(args):
        
    # Create the directory for predicted Images if does not exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    recSteps = args.rec_steps                                       #tiene que ser menor o igual que 16. Cantidad de etapas que tiene mi red
    resAutoencoder = args.residual                                  #flag that indicates if it is a residual model or not
    
    modelFile = args.model_path + args.modelName + '.h5'
    
    img_path = args.img_path + args.image
    imgDim = args.imgDim
    colorMap = args.colorMap
    
    img = image.load_img(img_path, target_size=(imgDim, imgDim))
    
    print(type(img))
    testImg = image.img_to_array(img)
    print(type(testImg))
    print(testImg.shape)

    if resAutoencoder == True:
        model=load_model(modelFile, custom_objects={'ResidualCalculation':ResidualCalculation, 'lossMSSSIML1_V3':lossMSSSIML1_V3, 'binarizationLayer':binarizationLayer, 'tf_binarization':tf_binarization, 'yuv_conversion':yuv_conversion, 'yuvToRgb_conversion':yuvToRgb_conversion, 'hsv_conversion':hsv_conversion, 'hsvToRgb_conversion':hsvToRgb_conversion})
        
        X_test2 = testImg.astype('float32')
        
        #Normalizo
        X_test2 = ((X_test2 / 255))
        
        patches = splitImage(X_test2)
        
        temp1 = model.predict(patches)
        temp2 = getImg(temp1, recSteps, colorMap)
        
        xDim, _, _ = testImg.shape
        
        
        print("Reconstruccion de la Imagen")
        recImg = buildImage(temp2, xDim)
        plt.figure()
        plt.imshow(recImg)
        plt.title("Reconstruccion: " + str(recSteps) + " etapas")
        plt.savefig(args.results_path + args.modelName + '_PredictedResidualImg.png')
        
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 2, 1)
        # for RGB images, The input to the imshow() expects it to be in the 0-1 range
        plt.imshow(testImg/255)
        plt.title("Imagen Original")
        plt.subplot(1, 2, 2)
        plt.imshow(recImg)
        plt.title("Prediccion")
        plt.savefig(args.results_path + args.modelName + '_ResidualComparision.png')
      
        
        #voy a imprimir la imagen original y las reconstrucciones con 1, 4, 8 y 16 steps
        plt.figure(figsize=(20, 4))
        
        for i in range(5):
            
            if i == 0:
                testImg.astype('uint8')
                recImg = testImg/255.
                plt.subplot(1, 5, i+1)
                plt.imshow(recImg)
                plt.title("Imagen Original")
            elif i == 1:
                recSteps = 1
                recImg = getImg(temp1, recSteps, colorMap)
                recImg = buildImage(recImg, imgDim)
                plt.subplot(1, 5, i+1)
                plt.imshow(recImg)
                plt.title("Reconstruccion: 1 etapa")
                recSteps = 2
            else:
                recSteps = recSteps * 2
                recImg = getImg(temp1, recSteps, colorMap)
                recImg = buildImage(recImg, imgDim)
                plt.subplot(1, 5, i+1)
                plt.imshow(recImg)
                plt.title("Reconstruccion: " + str(recSteps) + " etapas")
        plt.savefig(args.results_path + args.modelName + '_reconstructionPlot.png')
       
    
    else:
        model = load_model(modelFile, custom_objects={'lossMSSSIML1_V3':lossMSSSIML1_V3, 'binarizationLayer':binarizationLayer, 'tf_binarization':tf_binarization, 'yuv_conversion':yuv_conversion, 'yuvToRgb_conversion':yuvToRgb_conversion, 'hsv_conversion':hsv_conversion, 'hsvToRgb_conversion':hsvToRgb_conversion})
                
        X_test2 = testImg.astype('float32')
        
        #normalizo
        X_test2 = ((X_test2 / 255))
        
        patches = splitImage(X_test2)
        
        temp2 = model.predict(patches)
        
        #denormalizo
        temp2 = temp2 * 255
        temp2 = temp2.astype('uint8')
        print("Reconstruccion de la Imagen")
        recImg = buildImage(temp2, imgDim)
        plt.figure()
        plt.imshow(recImg)
        plt.savefig(args.results_path + args.modelName + '_PredictedImg.png')
        
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 2, 1)
        # for RGB images, The input to the imshow() expects it to be in the 0-1 range
        plt.imshow(testImg/255)
        plt.title("Imagen Original")
        plt.subplot(1, 2, 2)
        plt.imshow(recImg)
        plt.title("Prediccion")
        plt.savefig(args.results_path + args.modelName + '_Comparision.png')
        
        
    
    
#=============================================================================
# - Test PARAMETERS
#=============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ==================================================================================================================
    # MODEL PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--modelName', type=str, default='resConvAutoenc',
                        help='name of the model to be loaded')
    parser.add_argument('--residual', type=bool, default=False,
                        help='Set True if the model is residual, otherwise False')
    parser.add_argument('--rec_steps', type=int, default=16,
                        help='number of residuals to consider at image reconstruction')
    parser.add_argument('--colorMap', type=str, default='RGB',
                        help='Color map to use between: RGB, YUV and HSV')


    # ==================================================================================================================
    # LOADING  & Img Parameters
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--model_path', type=str, default='./saved_models/',
                        help='path were the models should be located')
    parser.add_argument('--img_path', type=str, default='./test_Imgs/',
                        help='path were the images should be located')
    parser.add_argument('--image', type=str, default='Pug.png',
                        help='name of the image to be loaded')
    parser.add_argument('--imgDim', type=int, default=512)
    parser.add_argument('--results_path', type=str, default='./PredictionResults/',
                        help='path were the models results should be located')
    
    
    


    #__________________________________________________________________________________________________________________
    args = parser.parse_args()
    print(args)
    main(args)