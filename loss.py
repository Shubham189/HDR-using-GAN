import numpy as np

# #########    https://www.tensorflow.org/tutorials/generative/pix2pix ######

#https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

class loss :

    def __init__(self,gplus,gminus,dplus,dminus):
        self.Dplus=dplus
        self.Dminus=dminus
        self.Gplus=gplus
        self.Gminus=gminus
  
    def l1loss(self,x,y,g):
         if(gloss=='gplus'):
           G=self.Gplus
         else if(gloss=='gminus'):
           G=self.Gminus

            
         l1loss= np.subtract(y,G(x))
         if(l1loss < 0):
           l1loss=-l1loss
      
         return l1loss


    def ganloss(self,x,g):
         if(gloss=='gplus'):
           G=self.Gplus
           D=self.Dplus
         else if(gloss=='gminus'):
           G=self.Gminus
           D=self.Dminus
        
          Gloss=np.square(D(G(x),x) - 1 ) 
          Dloss= 1/2(np.square(D(y,x)-1) ) + 1/2(np.square(D(G(x),x) -1))




         return Gloss,Dloss



