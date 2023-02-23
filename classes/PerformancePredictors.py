from classes.PerformanceModels import BasePerformanceModel
from abc import abstractmethod
import importlib
from math import log10


## BasePredictor
#
# Abstract class used to represent a performance model based on an external 
# module predictor for predicting the response time of a 
# Graph.Component.Partition object deployed onto a given Resources.Resource
class BasePredictor(BasePerformanceModel):
    
    ## @var module_name
    # Name of the module that implements the method used to predict the 
    # response time
    
    ## @var predictor
    # Object that performs the prediction
    
    ## BasePredictor class constructor
    #   @param self The object pointer
    #   @param keyword Keyword identifying the model
    #   @param module_name Name of the module to be loaded
    #   @param **kwargs Additional (unused) keyword arguments
    def __init__(self, keyword, module_name, **kwargs):
        super().__init__(keyword)
        self.module_name = module_name
        self.predictor = None
    
    ## Method to get a dictionary with the features required by the predict 
    # method
    #   @param self The object pointer
    #   @param **attributes Attributes that are used to retrieve the features
    #   @return The dictionary of the required features
    @abstractmethod
    def get_features(self, **attributes):
        pass
        
    ## Method to evaluate the object performance through the class predictor
    #   @param self The object pointer
    #   @param **features Model features
    #   @return Predicted response time
    @abstractmethod
    def predict(self, **features):
        pass


## FaaSPredictor
#
# Abstract class used to represent a performance model for predicting the 
# response time of a Graph.Component.Partition object deployed onto a 
# Resources.FaaS instance
class FaaSPredictor(BasePredictor):
    
    ## FaaSPredictor class constructor
    #   @param self The object pointer
    #   @param keyword Keyword identifying the model
    #   @param module_name Name of the module to be loaded
    #   @param **kwargs Additional (unused) keyword arguments
    def __init__(self, keyword, module_name, **kwargs):
        super().__init__(keyword, module_name)
    
    ## Method to get a dictionary with the features required by the predict 
    # method
    #   @param self The object pointer
    #   @param * Positional arguments are not accepted
    #   @param c_idx Index of the Graph.Component object
    #   @param p_idx Index of the Graph.Component.Partition object
    #   @param r_idx Index of the Resources.Resource object
    #   @param S A System.System object
    #   @param **kwargs Additional (unused) keyword arguments
    #   @return The dictionary of the required features
    def get_features(self, *, c_idx, p_idx, r_idx, S, **kwargs):
        c = S.components[c_idx].name
        p = S.components[c_idx].partitions[p_idx].name
        r = S.resources[r_idx].name
        features = {"arrival_rate": S.components[c_idx].comp_Lambda,
                    "warm_service_time": S.faas_service_times[c][p][r][0],
                    "cold_service_time": S.faas_service_times[c][p][r][1],
                    "time_out": S.resources[r_idx].idle_time_before_kill}
        return features

    ## Method to evaluate the object performance through the class predictor
    #   @param self The object pointer
    #   @param * Positional arguments are not accepted
    #   @param arrival_rate Arrival rate of requests
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    #   @param time_out Time spent by an idle Resources.FaaS instance 
    #                   before being killed
    #   @param **kwargs Additional (unused) keyword arguments
    #   @return Predicted response time
    @abstractmethod
    def predict(self, *, arrival_rate, warm_service_time, cold_service_time,
                time_out, **kwargs):
        pass


## FaaSPredictorPacsltk
#
# Specialization of FaaSPredictor class to predict the response time of a 
# Graph.Component.Partition object deployed onto a Resources.FaaS instance 
# relying on the method implemented in pacsltk module
class FaaSPredictorPacsltk(FaaSPredictor):
    
    ## @var predictor
    # Object that performs the prediction
    
    ## FaaSPredictorPacsltk class constructor
    #   @param self The object pointer
    #   @param **kwargs Additional (unused) keyword arguments
    def __init__(self, **kwargs):
        super().__init__("PACSLTK", "pacsltk.perfmodel")
        predictor_module = importlib.import_module(self.module_name)
        self.predictor = predictor_module.get_sls_warm_count_dist

    ## Method to evaluate the object performance through the class predictor
    #   @param self The object pointer
    #   @param * Positional arguments are not accepted
    #   @param arrival_rate Arrival rate of requests
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    #   @param time_out Time spent by an idle Resources.FaaS instance 
    #                   before being killed
    #   @param **kwargs Additional (unused) keyword arguments
    #   @return Predicted response time
    def predict(self, *, arrival_rate, warm_service_time, cold_service_time,
                time_out, **kwargs):
        perf = self.predictor(arrival_rate, warm_service_time, 
                              cold_service_time, time_out)
        return perf[0]["avg_resp_time"]


## FaaSPredictorMLlib
#
# Specialization of FaaSPredictor class to predict the response time of a 
# Graph.Component.Partition object deployed onto a Resources.FaaS instance 
# relying on the method implemented in a-MLlibrary
class FaaSPredictorMLlib(FaaSPredictor):
    
    ## @var predictor
    # Object that performs the prediction
    
    ## @var regressor_file
    # Path to the Pickle binary file that stores the model to be used 
    # for prediction
    
    ## FaaSPredictorMLlib class constructor
    #   @param self The object pointer
    #   @param regressor_file Path to the Pickle binary file that stores the 
    #                         model to be used for prediction
    #   @param **kwargs Additional (unused) keyword arguments
    def __init__(self, regressor_file, **kwargs):
        super().__init__("MLLIBfaas", "aMLLibrary.model_building.predictor")
        self.regressor_file = regressor_file
        predictor_module = importlib.import_module(self.module_name)
        self.predictor = predictor_module.Predictor(regressor_file,
                                                    "/tmp", False)
    
    ## Method to evaluate the object performance through the class predictor
    #   @param self The object pointer
    #   @param * Positional arguments are not accepted
    #   @param arrival_rate Arrival rate of requests
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    #   @param time_out Time spent by an idle Resources.FaaS instance 
    #                   before being killed
    #   @param **kwargs Additional (unused) keyword arguments
    #   @return Predicted response time
    def predict(self, *, arrival_rate, warm_service_time, cold_service_time,
                time_out, **kwargs):
        pd = importlib.import_module("pandas")
        columns = "Lambda,warm_service_time,cold_service_time,expiration_time".split(",")
        data = pd.DataFrame(data=[[arrival_rate, 
                                   warm_service_time,
                                   cold_service_time,
                                   time_out]],
                            columns=columns)
        return self.predictor.predict_from_df(data, True)
    
    ## Operator to convert a FaaSPredictorMLlib object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '"model":"{}", "regressor_file":"{}"'.\
            format(self.keyword, self.regressor_file)
        return s


## CoreBasedPredictor
#
# Specialization of FaaSPredictor class to predict the response time of 
# Graph.Component.Partition objects executed on Resources.Resource instances 
# depending on the number of used cores
class CoreBasedPredictor(BasePredictor):
    
    ## @var predictor
    # Object that performs the prediction
    
    ## @var regressor_file
    # Path to the Pickle binary file that stores the model to be used 
    # for prediction
    
    ## CoreBasedPredictor class constructor
    #   @param self The object pointer
    #   @param regressor_file Path to the Pickle binary file that stores the 
    #                         model to be used for prediction
    #   @param **kwargs Additional (unused) keyword arguments
    def __init__(self, regressor_file, **kwargs):
        super().__init__("CoreBasedPredictor", 
                         "aMLLibrary.model_building.predictor")
        self.regressor_file = regressor_file
        predictor_module = importlib.import_module(self.module_name)
        self.predictor = predictor_module.Predictor(regressor_file,
                                                    "/tmp", False)
    
    ## Method to get a dictionary with the features required by the predict 
    # method
    #   @param self The object pointer
    #   @param * Positional arguments are not accepted
    #   @param c_idx Index of the Graph.Component object
    #   @param p_idx Index of the Graph.Component.Partition object
    #   @param r_idx Index of the Resources.Resource object
    #   @param S A System.System object
    #   @param Y_hat Matrix denoting the amount of Resources assigned to each 
    #                Graph.Component.Partition object
    #   @param **kwargs Additional (unused) keyword arguments
    #   @return The dictionary of the required features
    def get_features(self, *, c_idx, p_idx, r_idx, S, Y_hat, **kwargs):
        n_res = Y_hat[c_idx][p_idx, r_idx]
        cores_per_res = S.resources[r_idx].n_cores
        cores = n_res * cores_per_res
        features = {"cores": cores,
                    "log_cores": log10(cores)}
        return features

    ## Method to evaluate the object performance through the class predictor
    #   @param self The object pointer
    #   @param * Positional arguments are not accepted
    #   @param cores Number of cores assigned to the object
    #   @param log_cores Logarithm of the number of cores
    #   @param **kwargs Additional (unused) keyword arguments
    #   @return Predicted response time
    def predict(self, *, cores, log_cores, **kwargs):
        pd = importlib.import_module("pandas")
        columns = "cores,log(cores)".split(",")
        data = pd.DataFrame(data=[[cores, log_cores]], columns=columns)
        return self.predictor.predict_from_df(data, True)
    
    ## Operator to convert a FaaSPredictorMLlib object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '"model":"{}", "regressor_file":"{}"'.\
            format(self.keyword, self.regressor_file)
        return s

