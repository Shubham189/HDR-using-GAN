import numpy as np
import math
# #########    https://www.tensorflow.org/tutorials/generative/pix2pix ######

#https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

class loss :

    def __init__(self,x,y,gplus,gminus,dplus,dminus):
        self.Dplus=dplus
        self.Dminus=dminus
        self.Gplus=gplus
        self.Gminus=gminus
        self.x=x
        self.y=np.ones_like()
  
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




         return Gloss,Dloss
    def Gloss(self,g):     
         delta=100
         Gloss=ganloss(self.x,g)[0]+ delta*l1loss(self.x,self.y,g)
         #Gminus=ganloss(self.x,'gminus')[0]+ delta*l1loss(self.x,self.y,'gminus')
         
         return Gloss
    def Dloss(self,dloss,iev1,iev1p,iev1m):
         if(dloss == 'dplus'):
             D=self.Dplus
         else if (dloss=='dminus'):    
             D=self.Dminus
         Iev1p=iev1p
         Iev1=iev1
         Iev1m=iev1m
         
         Dreal=D(Iev1p,Iev1) 
         Dfake=D(G(Iev1),Iev1) 
         Dplus=tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dreal,labels=tf.ones_like(Dreal)))
             +tf.math.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(logits=Dfake,labels=tf.zeros_like(Dfake))))  
 

         Dreal=D(Iev1m,Iev1) 
         Dfake=D(G(Iev1),Iev1) 
         Dminus=tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dreal,labels=tf.ones_like(Dreal)))
             +tf.math.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(logits=Dfake,labels=tf.zeros_like(Dfake))))  
 
            
           
