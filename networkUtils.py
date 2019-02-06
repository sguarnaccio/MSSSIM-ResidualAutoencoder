# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:40:31 2018

@author: sebastian
"""
from __future__ import print_function

import numpy as np
from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from tensorflow.python.framework import ops

sigma = [0.5, 1, 2, 4, 8]
num_scale=5 
alpha = 0.84
beta = 0.0 #probar varial el beta para varial el peso q tiene el MSSSIM en el calculo de las perdidas
C1 = 0.01**2 #revisar valor
C2 = 0.03**2 #revisar valor
filterWidth = 11
inChannel = 3

steps = 16

deterministic = 1



color_maps = ['RGB', 'YUV', 'HSV']
rgbMap = True


def hsv_conversion(x):
    import tensorflow as tf
    return tf.image.rgb_to_hsv(x)

def hsvToRgb_conversion(x):
    import tensorflow as tf
    return tf.image.hsv_to_rgb(x)

def yuv_conversion(x):
    import tensorflow as tf
    return tf.image.rgb_to_yuv(x)

def yuvToRgb_conversion(x):
    import tensorflow as tf
    return tf.image.yuv_to_rgb(x)



"""
Definicion: layer de binarizacion
"""
def binarizationLayer(inputCode):
    return tf_binarization(inputCode)


"""
Definicion: Implementacion del layer de binarizacion como optimization function
            que recibe la matriz del cuello de botella con valores entre -1 y 1 
            (flotantes) y devuelve una matriz de las mismas dimensiones con 
            valores binarios (-1 o 1)
"""
def tf_binarization(x, name=None):
    def binarizationop(inputCode):
        
        #codeShape = (batch_size, 6, 6, bitsPerPixel)
        codeShape = inputCode.shape
        #codeShape = K.int_shape(inputCode)
        #la forma de la matriz de entrada no debe contener la dimension correspondiente a la cantidad de muestras
        #codeShape = codeShape[1:len(codeShape)]
        if deterministic == 1:
            code =  np.sign(inputCode)
        elif deterministic == 2:
            code = inputCode * 0
        else:
            #de [-1 , 1] a [0, 1] campo continuo
            code = np.clip((inputCode + 1.) / 2. , 0, 1)
            #{0 , 1} dos valores discretos
            code = np.random.binomial(1, code,codeShape)
            #{-1 , 1}
            condlist = [code<1, code==1]
            choicelist = [code-1, code]
            code = np.select(condlist, choicelist)
            code = np.float32(code)
            
        
        return code
    
    
    
    def binarizationgrad(op, grad):
        return grad#el gradiente pasa derecho
        
    def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
            # Need to generate a unique name to avoid duplicates:
            rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    
            tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
            g = tf.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name}):
                return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    
    with ops.name_scope(name, "binarizationfunc", [x]) as name:
        z = py_func(binarizationop,
                    [x],
                    [tf.float32],
                    name=name,
                    grad=binarizationgrad)  # <-- here's the call to the gradient
        return tf.reshape(z[0], tf.shape(x))
        #return tf.convert_to_tensor(z)


"""
Definicion: Funcion que genera el filtro Gaussiano para 5 escalas diferentes a
            partir del cual se calculara el MS-SSIM
"""
def GaussianFilter(num_scale, filterWidth, inChannel, batch_size=1):
    w = np.empty((num_scale, batch_size, filterWidth*2, filterWidth*2, inChannel))
    for i in range(num_scale):
        gaussian = np.exp(-1.*(np.arange(-(filterWidth//2), filterWidth//2 + 1)**2)/(2*sigma[i]**2))
        gaussian = np.outer(gaussian, gaussian.reshape((filterWidth, 1)))	# extend to 2D
        gaussian = gaussian/np.sum(gaussian)	# normailization
        gaussian = np.tile(gaussian, (2, 2))
        gaussian = np.reshape(gaussian, (1, filterWidth*2, filterWidth*2, 1)) 	# reshape to 4D
        gaussian = np.tile(gaussian, (batch_size, 1, 1, inChannel))
        w[i,:,:,:,:] = gaussian
        
    		
    w = tf.convert_to_tensor(w, dtype=tf.float32)
    return w


gaussianFilter = GaussianFilter(num_scale, filterWidth, inChannel, 1)



"""
Definicion: Funcion de perdida empleando la metrica MS-SSIM a partir del filtro 
            Gaussiano creado
"""
def lossMSSSIML1_V3(y_true, y_pred):
#Incluso cuando la red este aprendiendo los pesos maximizando el SSIM unicamente para el pixel central del segmento de imagen, , los kernels aprendidos luego son alicados en todos los pixeles de la imagen
#pero como mi imagen es de 32x32, para que los pixeles de los extremos no queden excesivamente atenuados, voy a implementar una matriz de 32x32 compuesta por varios 16 filtros gaussianos iguales de 8x8
		

    y_true2 = K.stack([y_true,y_true,y_true,y_true,y_true], axis=0)
    y_pred2 = K.stack([y_pred,y_pred,y_pred,y_pred,y_pred], axis=0)
    	
    #Hago un promedio de la implementacion en varias areas de la imagen. De esta forma no dependo de que el filtro y la imagen deban tener dimenciones impares
    mux = (K.sum(gaussianFilter * y_pred2[:,:,0:22, 0:22,:], axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_pred2[:,:,10:32, 0:22,:], axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_pred2[:,:,0:22, 10:32,:], axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_pred2[:,:,10:32, 10:32,:], axis=[2, 3], keepdims=True))/16#sumo sobre la imagen 2D, axis 2 y 3 corresponden a Width y Hight
    muy = (K.sum(gaussianFilter * y_true2[:,:,0:22, 0:22,:], axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_true2[:,:,10:32, 0:22,:], axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_true2[:,:,0:22, 10:32,:], axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_true2[:,:,10:32, 10:32,:], axis=[2, 3], keepdims=True))/16 #divido por 16 porque aplico 4 veces el kernel y a su vez el kernel posee 4 filtros
    y_pred2 = y_pred2 - mux
    y_true2 = y_true2 - muy
    
    sigmax2 = ((K.sum(gaussianFilter * y_pred2[:,:,0:22, 0:22,:] ** 2, axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_pred2[:,:,10:32, 0:22,:] ** 2, axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_pred2[:,:,0:22, 10:32,:] ** 2, axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_pred2[:,:,10:32, 10:32,:] ** 2, axis=[2, 3], keepdims=True)) / 16) 
    sigmay2 = ((K.sum(gaussianFilter * y_true2[:,:,0:22, 0:22,:] ** 2, axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_true2[:,:,10:32, 0:22,:] ** 2, axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_true2[:,:,0:22, 10:32,:] ** 2, axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_true2[:,:,10:32, 10:32,:] ** 2, axis=[2, 3], keepdims=True)) / 16) 
    sigmaxy = ((K.sum(gaussianFilter * y_pred2[:,:,0:22, 0:22,:] * y_true2[:,:,0:22, 0:22,:], axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_pred2[:,:,10:32, 0:22,:] * y_true2[:,:,10:32, 0:22,:], axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_pred2[:,:,0:22, 10:32,:] * y_true2[:,:,0:22, 10:32,:], axis=[2, 3], keepdims=True) + K.sum(gaussianFilter * y_pred2[:,:,10:32, 10:32,:] * y_true2[:,:,10:32, 10:32,:], axis=[2, 3], keepdims=True)) / 16) 


    l = (2 * mux * muy + C1)/(mux ** 2 + muy **2 + C1)
    cs = (2 * sigmaxy + C2)/(sigmax2 + sigmay2 + C2)
    Pcs = K.prod(cs, axis=0)
    
    loss_MSSSIM = 1 - K.mean(l[-1,:,:,:,:] * Pcs)
    diff = y_true - y_pred
    
    #Sumo sobre el ancho y alto, divido por la cantidad de filtros y promedio segun la cantidad de canales y muestraas
    loss_L1 = K.mean(K.sum((K.abs(diff)[:,0:22, 0:22,:] * gaussianFilter[-1, :, :, :, :] + K.abs(diff)[:,0:22, 10:32,:] * gaussianFilter[-1, :, :, :, :] + K.abs(diff)[:,10:32, 0:22,:] * gaussianFilter[-1, :, :, :, :] + K.abs(diff)[:,10:32, 10:32,:] * gaussianFilter[-1, :, :, :, :]) /16, axis=[1, 2], keepdims=True))  # L1 loss weighted by Gaussian
    
    totalLoss = (alpha * loss_MSSSIM + (1-alpha) * loss_L1) * (1 - beta)
    

    return totalLoss


"""
Definicion: Gaussian Filter para lossMSSSIML1_V4
"""
def GaussianFilter2(num_scale, filterWidth, inChannel):
    w = np.empty((filterWidth*2, filterWidth*2, inChannel, num_scale))
    for i in range(num_scale):
        gaussian = np.exp(-1.*(np.arange(-(filterWidth//2), filterWidth//2 + 1)**2)/(2*sigma[i]**2))
        gaussian = np.outer(gaussian, gaussian.reshape((filterWidth, 1)))	# extend to 2D
        gaussian = gaussian/np.sum(gaussian)	# normailization
        gaussian = np.tile(gaussian, (2, 2))
        gaussian = np.reshape(gaussian, (filterWidth*2, filterWidth*2, 1)) 	# reshape to 3D
        gaussian = np.tile(gaussian, (1, 1, inChannel))
        w[:,:,:,i] = gaussian
        
    		
    w = tf.convert_to_tensor(w, dtype=tf.float32)
    return w

gaussianFilter2 = GaussianFilter2(num_scale, filterWidth, inChannel)



"""
Esta funcion hace lo mismo que lossMSSSIML1_V3. Si bien es mas amena 
sintacticamente, comete el error que al usar la funcion de convolucion de keras
esta operacion se realiza sobre el volumen, dejando un unico valor ponderando
todos los canales, cuando en realidad cada canal debe ser tratado de forma 
independiente porque tiene su propia distribucion.
"""
def lossMSSSIML1_V4(y_true, y_pred):
#Incluso cuando la red este aprendiendo los pesos maximizando el SSIM unicamente para el pixel central del segmento de imagen, , los kernels aprendidos luego son alicados en todos los pixeles de la imagen
#pero como mi imagen es de 32x32, para que los pixeles de los extremos no queden excesivamente atenuados, voy a implementar una matriz de 32x32 compuesta por varios 16 filtros gaussianos iguales de 8x8


    #y = K.reshape(y, (1,y.get_shape().as_list()[0], y.get_shape().as_list()[1],y.get_shape().as_list()[2],y.get_shape().as_list()[3])) #no lo necesito, stack ya crea la nueva dimension
    y_true2 = K.stack([y_true,y_true,y_true,y_true,y_true], axis=4)
    y_pred2 = K.stack([y_pred,y_pred,y_pred,y_pred,y_pred], axis=4)
    
    #Hago un promedio de la implementacion en varias areas de la imagen. De esta forma no dependo de que el filtro y la imagen deban tener dimenciones impares
    #convoluciono la imagen y la prediccion por el filtro gaussiano para 5 escalas distintas
    mux = tf.nn.convolution(y_pred, gaussianFilter2, "VALID", strides=[10,10], data_format = "NHWC")
    mux = K.sum(mux, axis=[1,2], keepdims=True) / 16
    mux = K.expand_dims(mux, axis=3)
    y_pred2 = y_pred2 - mux
    
    muy = tf.nn.convolution(y_true, gaussianFilter2, "VALID", strides=[10,10], data_format = "NHWC")
    muy = K.sum(muy, axis=[1,2], keepdims=True) / 16
    muy = K.expand_dims(muy, axis=3)
    y_true2 = y_true2 - muy
    
    sigmax2 = tf.nn.convolution(y_pred**2, gaussianFilter2, "VALID", strides=[10,10], data_format = "NHWC")
    sigmax2 = K.sum(sigmax2, axis=[1,2], keepdims=True) / 16
    sigmax2 = K.expand_dims(sigmax2, axis=3)
    
    sigmay2 = tf.nn.convolution(y_true**2, gaussianFilter2, "VALID", strides=[10,10], data_format = "NHWC")
    sigmay2 = K.sum(sigmay2, axis=[1,2], keepdims=True) / 16
    sigmay2 = K.expand_dims(sigmay2, axis=3)
    
    sigmaxy = tf.nn.convolution(y_pred*y_true, gaussianFilter2, "VALID", strides=[10,10], data_format = "NHWC")
    sigmaxy = K.sum(sigmaxy, axis=[1,2], keepdims=True) / 16
    sigmaxy = K.expand_dims(sigmaxy, axis=3)
    
    l = (2 * mux * muy + C1)/(mux ** 2 + muy **2 + C1)
    cs = (2 * sigmaxy + C2)/(sigmax2 + sigmay2 + C2)
    Pcs = K.prod(cs, axis=4)
    
    loss_MSSSIM = 1 - K.mean(l[:,:,:,:,-1] * Pcs)
    abs_diff = K.abs(y_true - y_pred)
    
    #Sumo sobre el ancho y alto, divido por la cantidad de filtros y promedio segun la cantidad de canales y muestraas
    loss_L1 = tf.nn.convolution(abs_diff, gaussianFilter2, "VALID", strides=[10,10], data_format = "NHWC")
    loss_L1 = K.mean(K.sum(loss_L1, axis=[1,2], keepdims=True) / 16)
    
    totalLoss = (alpha * loss_MSSSIM + (1-alpha) * loss_L1) * (1 - beta)
    

    return totalLoss




def output_of_lambda(input_shape):
    return input_shape




"""
Definicion: Esta capa obtiene el residuo de la etapa correspondiente y calcula
            el error.
"""
class ResidualCalculation(Layer):
    def __init__(self, **kwargs):
        super(ResidualCalculation, self).__init__(**kwargs)

    def call(self ,x ,mask=None):
        input_patch = x[0]
        pred = x[1]
        
        loss = (K.mean(K.square(input_patch - pred)) / steps) * beta
        
        
        
        #add the loss to an array that is going to be added to the result of the loss function used for compilation
        if beta > 0.0:
            self.add_loss(loss,x)

        return (input_patch - pred) #or should y return res to sum at the end???

    def get_output_shape_for(self, input_shape):
        return (input_shape)
    
    def get_config(self):
        #esto es necesario para salval el modelo con custom layers, actualmente funciona sin esto pero si quiero agregar parametros extra q paso en el init, debo agregarlos en la config q se pasa
        config = super(ResidualCalculation, self).get_config()
        #config['output_size'] = # say self. _output_size  if you store the argument in __init__
        return config


""" Define a zero loss that actually sits and do nothing with y_true and y_pred. Then it just add the losses calculated through the
ResidualCalculation layer""" 
def zero_loss(y_true, y_pred):
    #return K.zeros_like(y_pred)#the losses added in the ResidualCalculation layer will be added to the output of K.zeros_like
    return 0.000001 * K.mean(K.square(y_pred - y_true))


def addPredictions(x):
  
    return x[0] + x[1]





"""
Definicion: Funcion recupera la imagen a partir de la suma de las predicciones 
            de los residuos realizada por la red.
            
            El unico objetivo de esta funcion es poder reconstruir la imagen 
            segun la cantidad de residuos que se quiera emplear y no estar
            forzado a usar solo la prediccion de la salida general la cual
            consiste en la suma de todas las etapas

"""
def getImg(predictionList, residualCount, colorMap):
    
    if colorMap in color_maps:
        if colorMap == 'RGB':
            rgbMap = True
        else:
            rgbMap = False
            if colorMap == 'YUV':
                mapConvInv = yuvToRgb_conversion
            elif colorMap == 'HSV':
                mapConvInv = hsvToRgb_conversion
    else:
        rgbMap = True
    
    if residualCount <= len(predictionList)-1:#the last component of pred is the image already reconstructed needed for caalculating the loss on training
        img = np.array(predictionList)
        img = np.sum(img[0:residualCount], axis=0)
        #img = ((img / 1.8) + 0.5) * 255
        if rgbMap == False:
            img = K.eval(mapConvInv(img))
         
        img = img * 255
        img = img.astype('uint8')
    else:
       img = np.array(predictionList)
       img = np.sum(img[0:len(img)-1], axis=0)
       #img = ((img / 1.8) + 0.5) * 255
       if rgbMap == False:
            img = K.eval(mapConvInv(img))
       
       img = img * 255
       img = img.astype('uint8')

    return img


