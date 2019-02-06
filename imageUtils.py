# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:47:51 2018

@author: sebastian
"""
import numpy as np
import tensorflow as tf
#from keras import backend as K

#split the image in patches of 32x32
def splitImage(img, networkDimension = 32):
    #networkDimension = 32
        
    #img = img.astype('uint8')
    testImgx, _, _ = img.shape
        
    numOfPatchesInRow = testImgx // networkDimension
    
    patches = np.zeros((numOfPatchesInRow * numOfPatchesInRow,networkDimension,networkDimension, 3), dtype=np.float32)
    for i in range(numOfPatchesInRow):
        for j in range(numOfPatchesInRow):
            #we want to extract a patch of 32x32 dimension 
            patches[i*numOfPatchesInRow + j] = img[networkDimension*i:networkDimension*(i+1),networkDimension*j:networkDimension*(j+1)] # Change
    return patches

def buildImage(patchesArray, imgDim, networkDimension = 32):
    
    reconstructedImg = np.zeros((imgDim,imgDim, 3), dtype=np.uint8)
    #networkDimension = 32
    numOfPatchesInRow = imgDim // networkDimension
    
    for i in range(numOfPatchesInRow):
        for j in range(numOfPatchesInRow):
            #image reconstruction
            reconstructedImg[networkDimension*i:networkDimension*(i+1),networkDimension*j:networkDimension*(j+1)] = patchesArray[numOfPatchesInRow*i + j]
    return reconstructedImg


""" Quality metrics """
def getMetric(testImg, predImg, cmpQuality): #testImg y predImg deben ser las imagenes desnormalizadas, uint8 con valores de 0 a 255
    
    #predImg = splitImage(predImg, networkDimension = 8) 
    
    'encode to JPEG'
    input_i = tf.placeholder(tf.uint8)
    input_j = tf.placeholder(tf.string)
    
    encoded_img = tf.image.encode_jpeg(input_i, format="rgb", quality=cmpQuality)
    decoded_img = tf.image.decode_jpeg(input_j, channels=3)
    
    with tf.Session() as sess:
        encode = sess.run(encoded_img, feed_dict={input_i: testImg})
        decode = sess.run(decoded_img, feed_dict={input_j: encode})
    #decode = splitImage(decode, networkDimension = 8)
        
    'Calculate MS-SSIM for the JPEG image and the predicted image'
    input_a = tf.placeholder(tf.uint8)
    input_b = tf.placeholder(tf.uint8)
    
    ssim_calc = tf.image.ssim(input_a, input_b, 255)
    with tf.Session() as sess:
        ssim = sess.run(ssim_calc, feed_dict={input_a: testImg, input_b: decode})
        
    with tf.Session() as sess:
        ssim2 = sess.run(ssim_calc, feed_dict={input_a: testImg, input_b: predImg})
        
    #K.clear_session()
    
    return ssim, ssim2, decode


def getMetricForBatch(testImg, predImg, cmpQuality): #testImg y predImg deben ser las imagenes desnormalizadas, uint8 con valores de 0 a 255
    
    #predImg = splitImage(predImg, networkDimension = 8) 
    
    'encode to JPEG'
    jpegTensor = toJPEG(testImg, cmpQuality)
    #decode = splitImage(decode, networkDimension = 8)
        
    testImg = tf.image.convert_image_dtype(testImg, tf.uint8)
    predImg = tf.image.convert_image_dtype(predImg, tf.uint8)
    
    'Calculate MS-SSIM for the JPEG image and the predicted image'
    ssim_calc = tf.image.ssim(testImg, predImg, 255)
    with tf.Session() as sess:
        ssim_predictions = sess.run(ssim_calc)
    
    ssim_calc = tf.image.ssim(jpegTensor, predImg, 255)
    with tf.Session() as sess:
        ssim_jpeg = sess.run(ssim_calc)
        
    #K.clear_session()
        
    return ssim_predictions, ssim_jpeg, jpegTensor



def getSSIM(refImg, img): #testImg y predImg deben ser las imagenes desnormalizadas, uint8 con valores de 0 a 255
    
    refImg = tf.image.convert_image_dtype(refImg, tf.uint8)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    
    'Calculate MS-SSIM for the JPEG image and the predicted image'
    ssim_calc = tf.image.ssim(refImg, img, 255)
    with tf.Session() as sess:
        ssim = sess.run(ssim_calc)
        
    #K.clear_session()
    
    return ssim



"Funcion definida para estimar el tamano bruto en bits de la codificacion jpeg sin tener en cuenta el encabezado"
def jpegEncRawSize(testImg, cmpQuality, chrDwnSamp = True):
    
    'encode to JPEG'
    input_i = tf.placeholder(tf.uint8)
    
    encoded_img = tf.image.encode_jpeg(input_i, format="rgb", quality=cmpQuality, chroma_downsampling=chrDwnSamp)
    
    
    with tf.Session() as sess:
        encode = sess.run(encoded_img, feed_dict={input_i: testImg})
        
        
    'find SOS(\xffda) in JPEG encoding. After this segment we find the raw encoding of teh image'
    index = encode.find(b'\xff\xda')
    
    'the SOS segment is composed by 10 bytes, then we have to remove everything before the index plus ten extra bytes'
    encode = encode[index+10:]
    
    'encode is a byte array, so me multiplied by 8 to get the number of bits that encodes the image'
    
    #K.clear_session()
    
    return len(encode)*8


def toJPEG(testImg, cmpQuality, chrDwnSamp = True):
    
    'encode to JPEG'
    input_i = tf.placeholder(tf.uint8)
    input_j = tf.placeholder(tf.string)
    
    encoded_img = tf.image.encode_jpeg(input_i, format="rgb", quality=cmpQuality, chroma_downsampling=chrDwnSamp)
    decoded_img = tf.image.decode_jpeg(input_j, channels=3)
    
    with tf.Session() as sess:
        encode = sess.run(encoded_img, feed_dict={input_i: testImg})
        decode = sess.run(decoded_img, feed_dict={input_j: encode})
        
    #K.clear_session()
    
    return decode

def numpyToJPEGTensor(testImg, cmpQuality, chrDwnSamp = True):
    
    jpegArray= np.zeros(testImg.shape)
    index, _, _, _ = testImg.shape
    
    'encode to JPEG'
    input_i = tf.placeholder(tf.uint8)
    input_j = tf.placeholder(tf.string)
    
    
    for i in range(index):
        encoded_img = tf.image.encode_jpeg(input_i, format="rgb", quality=cmpQuality, chroma_downsampling=chrDwnSamp)
        decoded_img = tf.image.decode_jpeg(input_j, channels=3)
        
        with tf.Session() as sess:
            encode = sess.run(encoded_img, feed_dict={input_i: testImg[i]})
            decode = sess.run(decoded_img, feed_dict={input_j: encode})
        
        jpegArray[i] = decode
            
    
    #K.clear_session()    
        
    return jpegArray


def jpegEncRawSizeTensor(testImg, cmpQuality, chrDwnSamp = True):
    
    aux, _, _, _ = testImg.shape
    jpegSizeArray= np.zeros((aux,))
    
    
    'encode to JPEG'
    input_i = tf.placeholder(tf.uint8)
    
    for i in range(aux):
        encoded_img = tf.image.encode_jpeg(input_i, format="rgb", quality=cmpQuality, chroma_downsampling=chrDwnSamp)
        
        
        with tf.Session() as sess:
            encode = sess.run(encoded_img, feed_dict={input_i: testImg[i]})
            
            
        'find SOS(\xffda) in JPEG encoding. After this segment we find the raw encoding of teh image'
        index = encode.find(b'\xff\xda')
        
        'the SOS segment is composed by 10 bytes, then we have to remove everything before the index plus ten extra bytes'
        encode = encode[index+10:]
        
        'encode is a byte array, so me multiplied by 8 to get the number of bits that encodes the image'
        jpegSizeArray[i] = len(encode)*8
        
    #K.clear_session()
    
    return jpegSizeArray