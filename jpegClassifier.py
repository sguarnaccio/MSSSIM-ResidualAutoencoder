# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:59:36 2019
Descripcion: Este script se encarga de hayar la calidad de compresion JPEG 
             correspondiente para cada imagen en cada uno de los intervalos
             de la tasa de compresion definidos a medida que se agrega una
             etapa del autoencoder
@author: sebastian
"""



import tensorflow as tf
from keras.datasets import cifar10


import json

from keras import backend as K

import gc

"""
Descripcion: Funcion definida para estimar el tamaño bruto en bits de la 
             codificacion jpeg sin tener en cuenta el encabezado
"""
def jpegEncRawSize(testImg, cmpQuality, chrDwnSamp = True):
    
    #encode to JPEG
    input_i = tf.placeholder(tf.uint8)
    
    encoded_img = tf.image.encode_jpeg(input_i, format="rgb", quality=cmpQuality, chroma_downsampling=chrDwnSamp)
    
    
    with tf.Session() as sess:
        encode = sess.run(encoded_img, feed_dict={input_i: testImg})
        
    #Buscamos el comienzo del segmento SOS(\xffda) en el encabezado de la codificacion
    #Luego de este segmento se encuentran los bits de la codificacion bruta de la imagen
    index = encode.find(b'\xff\xda')
    
    #El segmento SOS esta compuesto por 10 bytes, por lo tanto debemos eliminar
    #todo lo que se encuentra antes del indice mas 10 bytes extras correspondientes
    #al segmento SOS
    encode = encode[index+10:]
    
    #la codificacion es un array de bytes por lo que multiplicamos el largo de 
    #este array por 8 para obtener la cantidad de bits de la imagen codificada
    
    K.clear_session()
    
    return len(encode)*8


""" LOAD DATA """		
(X, y), (X_test, y_test) = cifar10.load_data()




datSize = 10000


jpegSegments = True

jpegdic1 = dict()
jpegdic2 = dict()
jpegdic3 = dict()
jpegdic4 = dict()
jpegdic5 = dict()
jpegdic6 = dict()
jpegdic7 = dict()
jpegdic8 = dict()
jpegdic9 = dict()
segmentIterations = 9 #number of jpeg range dictionaries

auxDict = dict()
getDict = dict()
case1 = {1:jpegdic1, 2:jpegdic2, 3:jpegdic3, 4:jpegdic4, 5:jpegdic5, 6:jpegdic6, 7:jpegdic7, 8:jpegdic8, 9:jpegdic9 }

#rangos de la tasa de compresion para una red donde cada etapa tiene una 
#cantidad de bits = 6x6x8xcantEtapas, donde cantEtapas_min = 4 y cantEtapas_max = 12
case2 = {1:1152, 2:1440, 3:1728, 4:2016, 5:2304, 6:2592, 7:2880, 8:3168, 9:3456}

state_path = './jsonFiles/'
file_name = 'jpegdic4NoChrom'

quality = 10
qualityOffset = 1
loopIndex = 0


load = False
chromaDownSamp = False

#guardo el estado para poder retomar de este punto en caso que se interrumpa 
#la ejecucion
state=dict()
state['q'] = quality
state['i'] = loopIndex

loadDict = dict()

#cargo el estado de ejecucion y los diccionarios 
if load == True:
    
    for k in range(1, segmentIterations + 1): 			
        getDict = case1[k]
		
        
        with open(state_path + file_name + str(k) + '.json', 'r') as fp:
            loadDict = json.load(fp)

        for key in loadDict.keys():
            intKey = int(key)
            getDict[intKey] = loadDict[key]

    try:
        with open(state_path + file_name + 'state.json', 'r') as fp:
            state = json.load(fp)
            quality = state['q']
            loopIndex = state['i']
    except Exception:
        print('No such file or directory: ' + state_path + file_name + 'state.json')
	

#ejecuto a partir del estado cargado o desde el inicio
if jpegSegments == True:
    for i in range(loopIndex, datSize): 
        quality = 10
	
        for j in range(1, segmentIterations + 1): 
			
            auxDict = case1[j]
            segmentLimit = case2[j]
			
		
            aux = jpegEncRawSize(X_test[i], quality, chrDwnSamp = chromaDownSamp)
            while aux < segmentLimit:
                quality = quality + qualityOffset
                aux = jpegEncRawSize(X_test[i], quality, chrDwnSamp = chromaDownSamp)
			
            auxDict[i] = quality

        
        if i % 250 ==0 and i > 0:
            print("Image ", i, " done")
			
            for k in range(1, segmentIterations + 1): 
			
                saveDict = case1[k]
				
                aux5 = dict()
                for key in saveDict.keys():
                    strKey = str(key)
                    aux5[strKey] = saveDict[key]
					
                with open(state_path + file_name + str(k) + '.json', 'w') as fp:
                    json.dump(aux5, fp)
			
            'guardo el estado del loop'
            state['q'] = quality
            state['i'] = i
            
            with open(state_path + file_name + 'state.json', 'w') as fp:
                json.dump(state, fp)
        
            print("Estate Saved")
        
        gc.collect()
        
        
        