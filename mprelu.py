from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras import backend as k

from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.util.tf_export import keras_export



class MPReLU(tf.keras.layers.Layer, alpha_initializer='zeros',alpha_regularizer=None,alpha_constraint=None,shared_axes=None,):
  def __init__(self):
    super(MPReLU, self).__init__(**kwargs)
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.alpha_regularizer=regularizers.get(alpha_regularizer)
    self.alpha_constraint=alpha_constraint
    
    if(shared_axes is None):
        self.shared_axes=None
    else:    
     self.shared_axes=list(shared_axes)
    

    
    
    self.input_shape=(0,0)



  def build(self, input_shape):
    param_shape=list(input_shape[1:])

    if (self.shared_axes is not None):
         for(i in self.shared_axes):
             param_shape[i-1]=1
    self.alpha=self.add_weight(shape=param_shape,name='alpha',initializer=self.alpha_initializer,regularizer=self.alpha_regularizer,constraint=self.alpha_constraint)         
    

    if shared_axes:
        for i in range(1,len(input_shape)):
             if i not in shared_axes:
               axes[i]=input_shape[i]
    
    self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
    self.built = True           
   

  def get_config(self):
    config = {
        'alpha_initializer': initializers.serialize(self.alpha_initializer),
        'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
        'alpha_constraint': constraints.serialize(self.alpha_constraint),
        'shared_axes': self.shared_axes
    }
    base_config = super(MPReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape

  
  def call(self, inputs):
     self.input_shape=inputs.shape
     x=inputs
     build(self.input_shape)
     pos=self.alpha*k.relu(x)
     neg=k.relu(-x)

     return pos+neg


  

     

