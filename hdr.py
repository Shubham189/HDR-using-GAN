import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers as tfl
img =cv2.imread("test.jpeg")
print(img.shape)
original_img=np.asarray(img)
#print(original_img)

#hYPER parameters 

epsilon=0.00005
class hdrGAN:
    def __init__(self,generator,discriminator,inputimg):
        self.generator=generator
        self.discriminator=discriminator
        self.img=inputimg
        self.input_dim=self.img.shape
    def generator(input_dim,output_dim,img):
           with tf.variable_scope("generator")  as scope: 

            model=tf.keras.Sequential()
       ############################################################ down sampling layers ############################
       
       
               # first block 
       
            model.add(tfl.Conv2D(filters=64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PRelu())

            #second block

            model.add(tfl.Conv2D(filters=128,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block2'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PRelu())

            #third  block

            model.add(tfl.Conv2D(filters=256,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block3'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PRelu())



            #fourth block

            model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block4'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PRelu())
             
            #fifth block

            model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block5'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PRelu())
             
              
class disc:
            self.img=inputimg
            self.input_dim=self.img.shape

            def dplus(self,input_dim,output_dim):
                      # with tf.compat.v1.VariableScope(reuse=False,name="discriminator") as scope: 
                      inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
                      tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

                      x = tf.keras.layers.concatenate([inp, tar])
                        
                      model=tf.keras.Sequential()
                      model.add(tfl.Conv2D(filters=6,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block0'))
                      
                      # first block 
                        
                      model.add(tfl.Conv2D(filters=64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
                      model.add(tfl.PReLU())

                      #second block

                      model.add(tfl.Conv2D(filters=128,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block2'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(tfl.PReLU())

                      #third  block

                      model.add(tfl.Conv2D(filters=256,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block3'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(tfl.PReLU())



                      #fourth block

                      model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block4'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(tfl.PReLU())
                      
                      #fifth block

                      model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block5',activation=tf.keras.activations.sigmoid))
                      model.add(tfl.PReLU())
                      
                      
                      return model

            def dminus(self,input_dim,output_dim):
           
                      model=tf.keras.Sequential()       
                      model.add(tfl.InputLayer(input_shape=input_dim,dtype=tf.int16))
                      
                      model.add(tfl.Conv2D(filters=6,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block0'))
                      
                      # first block 
                      
                      model.add(tfl.Conv2D(filters=64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
                      model.add(MPReLU())

                      #second block

                      model.add(tfl.Conv2D(filters=128,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block2'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(MPReLU())

                      #third  block

                      model.add(tfl.Conv2D(filters=256,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block3'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(MPReLU())



                      #fourth block

                      model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block4'))
                      model.add(tfl.BatchNormalization(epsilon=epsilon))
                      model.add(MPReLU())
                      
                      #fifth block

                      model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block5',activation=tf.keras.activations.sigmoid))
                      model.add(MPReLU())

                      return model            

                         
               

