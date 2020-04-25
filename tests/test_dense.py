import pytest 
import numpy as np
from NN.layers import Dense


activations = [ 
    'sigmoid',
    'relu',
    'exponential',
    'tanh',
    'elu',
    'leakyrelu',
    'softmax',
    'softsign'
]

class TestDense:

    def test_output_shape(self):
        my_input = np.random.normal(loc=0,scale=1,size=(10,20))
        Layer = Dense(10,True,'relu')
        Layer.build(my_input.shape)
        output = Layer.call(my_input)
        assert output.shape == (10,10)
    
    @pytest.mark.parametrize('activation',activations)
    def test_activations(self,activation):
        my_input = np.random.normal(loc=0,scale=1,size=(10,20))
        Layer = Dense(10,True,activation)
        Layer.build(my_input.shape)
        output = Layer.call(my_input)
        assert type(output) == np.ndarray
        assert output.shape == (10,10)