import numpy as np
import warnings


class MeanSquaredError:

    def __call__(self,true,predicted):
        n = 1/len(predicted)
        return n*np.sum([true-predict for true,predict in zip(true,predicted)])
    
class RootMeanSquaredError:

    def __call__(self,true,predicted):
        warnings.warn("Please remember that value under square must not be negative")
        n = 1/len(predicted)
        return np.sqrt(n*np.sum([true-predict for true,predict in zip(true,predicted)]))

class MeanAbsoluteError:

    def __call__(self,true,predicted):
        n = 1/len(predicted)
        return n*np.sum(np.abs([true-predict for true,predict in zip(true,predicted)]))
    
class LogLoss:

    def __call__(self,true,predicted):
        n = 1/len(predicted)
        return -n*np.sum([true*np.log(predict)+(1-true)*np.log(1-predict) 
                          for true,predict in zip(true,predicted)])