import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers as tfl
from PIL import Image
import matplotlib.pyplot as plt
#img =cv2.imread("test.jpeg")
print(tfl)
#original_img=np.asarray(img)
#print(original_img)

#hYPER parameters 

epsilon=0.00005
class hdrGAN:
    def __init__(self,inputimg):
       
       
        self.img=inputimg
        self.input_dim=self.img.shape
    def generator(self,input_dim,output_dim):
          # with tf.compat.v1.VariableScope(reuse=False,name="generator") as scope: 
           
            model=tf.keras.Sequential()
       ############################################################ down sampling layers ############################
       
           # model.add(tfl.InputLayer(input_shape=input_dim,dtype=tf.int16))
            
           # model.add(tfl.Conv2D(filters=3,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
               # first block 
            
            model.add(tfl.Conv2D(filters=64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
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

            model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block5'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())
            
         

         ############################################################ UP sampling layers ############################

            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2D(filters=2*512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block6'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())
               

            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2D(filters=2*256,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block7'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())


            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2D(filters=2*128,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block8'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())

            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2DTranspose(filters=2*64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block9'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())
            
            model.add(tfl.UpSampling2D(size=(2,2),interpolation='nearest'))
            model.add(tfl.Conv2DTranspose(filters=2*3,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block10'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())  
            


           
            model.add(tfl.Conv2DTranspose(filters=3,kernel_size=(10,10),strides=(2,2),padding='same',name='deconvolution layer11',data_format="channels_last"))
           # model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PReLU())  
            # model.add(tfl.Flatten())
            
            return model  
 
img=np.array(Image.open('./images/ev'+str(0)+'.jpg'),dtype=np.int16)
#img=np.array(Image.open('./test.jpeg'),dtype=np.float32)
i,j,k=img.shape[0],img.shape[1],img.shape[2]

img=tf.image.resize(img,(256,256))
img=np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
GAN=hdrGAN(img)        
print(img.shape)

generator=GAN.generator(img.shape,img.shape)

generator=generator(img)

#generator=np.reshape(generator,(1,256,256,3))

img=generator[0]*256
#img=tf.dtypes.cast(img, tf.int16)
print(img)
imgplot = plt.imshow(img)
plt.show()

#tf.image.encode_png(img)
print(generator.shape)
                         
               

