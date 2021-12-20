from abc import ABC, abstractmethod


## BasePerformanceModel
#
# Abstract class used to represent a performance model for predicting the 
# response time of a Graph.Component.Partition object deployed onto a given 
# Resources.Resource
class BasePerformanceModel(ABC):
    
    ## @var keyword
    # Keyword identifying the model
    
    ## @var allows_colocation
    # True if Graph.Component.Partition objects relying on this model 
    # can be co-located on a device
    
    ## BasePerformanceModel class constructor
    #   @param self The object pointer
    #   @param keyword Keyword identifying the model
    #   @param **kwargs Additional (unused) keyword arguments
    def __init__(self, keyword, **kwargs):
        self.keyword = keyword
        self.allows_colocation = False
    
    ## Method to get a dictionary with the features required by the predict 
    # method
    #   @param c_idx Index of the Graph.Component object
    #   @param p_idx Index of the Graph.Component.Partition object
    #   @param r_idx Index of the Resources.Resource object
    #   @param S A System.System object
    #   @param Y_hat Matrix denoting the amount of Resources assigned to each 
    #                Graph.Component.Partition object
    #   @param **kwargs Additional (unused) keyword arguments
    #   @return The dictionary of the required features
    @abstractmethod
    def get_features(self, c_idx, p_idx, r_idx, S, Y_hat, **kwargs):
        pass
        
    ## Method to evaluate the object performance through the class predictor
    #   @param self The object pointer
    #   @param **features Model features
    #   @return Predicted response time
    @abstractmethod
    def predict(self, **features):
        pass
    
    ## Operator to convert a BasePerformanceModel object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '"model":"{}"'.\
            format(self.keyword)
        return s

