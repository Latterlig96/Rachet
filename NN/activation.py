import numpy as np 

class Activation: 

    @staticmethod
    def sigmoid(X):
        return 1 / (1+np.exp(-X))

    @staticmethod
    def relu(X):
        return np.maximum(0,X)

    @staticmethod
    def exponential(X):
        return np.exp(X)
    
    @staticmethod
    def tanh(X): 
        return np.tanh(X)
    
    @staticmethod
    def elu(X,alpha):
        return X if X > 0 else alpha*(np.exp(X)-1)

    @staticmethod
    def leakyrelu(X,alpha):
        return max(X*alpha,X)
    
    @staticmethod
    def softmax(X):
        nominator = np.exp(X)
        denominator = np.sum(np.exp(X)) 
        return nominator/denominator
    
    @staticmethod
    def softsign(X):
        return X / (abs(X)+1)