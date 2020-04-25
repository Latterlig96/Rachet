import numpy as np 
from activation import Activation

class Dense:
    def __init__(self,
                units,
                use_bias,
                activation):
        self.units = units  
        self.activation = activation
        self.use_bias = use_bias 

    def build(self,input_shape):
        assert len(input_shape) >= 2 
        input_dim = input_shape[-1]

        self.weights = np.random.uniform(low=0,high=1,size=(input_dim,self.units))

        if self.use_bias:
            self.bias = np.random.uniform(low=0,high=1,size=(self.units,))
        
        return self
    
    def call(self,inputs):
        output = np.dot(inputs,self.weights)
        if self.use_bias:
            output = output + self.bias
        if self.activation:
            output = getattr(Activation,self.activation)(output)
        return output 
    
    def get_output_shape(self,input_shape):
        f_dim = input_shape[0]
        s_dim = self.weights.shape[-1]
        return (f_dim,s_dim)