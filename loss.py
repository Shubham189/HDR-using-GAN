import numpy as np
import math
# #########    https://www.tensorflow.org/tutorials/generative/pix2pix ######
############   https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py        ###########
#https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

class loss :

    def __init__(self,x,y,gplus,gminus,dplus,dminus):
        self.Dplus=dplus
        self.Dminus=dminus
        self.Gplus=gplus
        self.Gminus=gminus
        self.x=x
        self.y=np.ones_like()
        self.GaNLossD=0
  
    def l1loss(self,x,y,gloss):
         if(gloss=='gplus'):
           G=self.Gplus
         else if(gloss=='gminus'):
           G=self.Gminus

            
         l1loss= np.subtract(y,G(x))
         if(l1loss < 0):
           l1loss=-l1loss
      
         return l1loss


    def ganloss(self,x,gloss):
         
         if(gloss=='gplus'):
           G=self.Gplus
           D=self.Dplus
         else if(gloss=='gminus'):
           G=self.Gminus
           D=self.Dminus
        
          Gloss=np.square(D(G(self.x),self.x) - 1 ) 
          Dloss= 1/2(np.square(D(self.y,self.x)-1) ) + 1/2(np.square(D(G(self.x),self.x) -1))
          self.GaNLossD=Dloss
          


         return Gloss
    def Gloss(self,g):     
         delta=100
         Gloss=ganloss(self.x,g)+ delta*l1loss(self.x,self.y,g)
         #Gminus=ganloss(self.x,'gminus')[0]+ delta*l1loss(self.x,self.y,'gminus')
         
         return Gloss
    def Dloss(self,dloss,iev1,iev1p,iev1m,g):
         if(dloss == 'dplus'):
             D=self.Dplus
         else if (dloss=='dminus'):    
             D=self.Dminus
         Iev1p=iev1p
         Iev1=iev1
         Iev1m=iev1m
         if(g=='gplus'):
          Dreal=D(Iev1p,Iev1) 
          Dfake=D(G(Iev1),Iev1) 
          Dplus=tf.reduce_mean(math.log(Dreal))+tf.reduce_mean(1-math.log(Dfake))

          Dloss=Dplus
         else if(g=='gminus'):
           Dreal=D(Iev1m,Iev1) 
           Dfake=D(G(Iev1),Iev1) 
           Dminus=tf.reduce_mean(math.log(Dreal))+tf.reduce_mean(1-math.log(Dfake))
           Dloss=Dminus
         

         if(Dloss<0):
              Dloss=-Dloss
         return (1-Dloss)

     
            
           
