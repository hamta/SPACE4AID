from abc import ABC, abstractmethod
import importlib


## FaaSPredictor
#
# Abstract class used to represent a performance model for predicting the 
# response time of a Resources.FaaS instance
class FaaSPredictor(ABC):
    
    ## @var module_name
    # Name of the module that implements the method used to predict the 
    # Resources.FaaS performance
    
    ## @var predictor
    # Object that performs the prediction
    
    ## @var keyword
    # Keyword identifying the model
    
    ## FaaSPredictor class constructor
    #   @param self The object pointer
    #   @param keyword Keyword identifying the model
    #   @param module_name Name of the module to be loaded
    #   @param **kwargs Additional (unused) keyword arguments
    def __init__(self, keyword, module_name, **kwargs):
        self.keyword = keyword
        self.module_name = module_name
        self.predictor = None
    
    ## Method to get a dictionary with the features required by the predict 
    # method
    #   @param c_idx Index of the Graph.Component object
    #   @param p_idx Index of the Graph.Component.Partition object
    #   @param r_idx Index of the Resources.Resource object
    #   @param S A System.System object
    #   @param **kwargs Additional (unused) keyword arguments
    #   @return The dictionary of the required features
    def get_features(self, c_idx, p_idx, r_idx, S, **kwargs):
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
    #   @param arrival_rate Arrival rate of requests
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    #   @param time_out Time spent by an idle Resources.FaaS instance 
    #                   before being killed
    #   @param **kwargs Additional (unused) keyword arguments
    #   @return Predicted response time
    @abstractmethod
    def predict(self, arrival_rate, warm_service_time, cold_service_time,
                time_out, **kwargs):
        pass
    
    ## Operator to convert a FaaSPredictor object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '"model":"{}"'.\
            format(self.keyword)
        return s



## FaaSPredictorPacsltk
#
# Specialization of FaaSPredictor class to predict a Resources.FaaS instance 
# response time relying on the method implemented in pacsltk module
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
    #   @param arrival_rate Arrival rate of requests
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    #   @param time_out Time spent by an idle Resources.FaaS instance 
    #                   before being killed
    #   @param **kwargs Additional (unused) keyword arguments
    #   @return Predicted response time
    def predict(self, arrival_rate, warm_service_time, cold_service_time,
                time_out, **kwargs):
        perf = self.predictor(arrival_rate, warm_service_time, 
                              cold_service_time, time_out)
        return perf[0]["avg_resp_time"]



## FaaSPredictorMLlib
#
# Specialization of FaaSPredictor class to predict a Resources.FaaS instance 
# response time according to a given model and relying on the method 
# implemented in a-MLlibrary
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
        super().__init__("MLLIBfaas", "a-MLlibrary.model_building.predictor")
        self.regressor_file = regressor_file
        predictor_module = importlib.import_module(self.module_name)
        self.predictor = predictor_module.Predictor(regressor_file,
                                                    "/tmp", False)
    
    ## Method to evaluate the object performance through the class predictor
    #   @param self The object pointer
    #   @param arrival_rate Arrival rate of requests
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    #   @param time_out Time spent by an idle Resources.FaaS instance 
    #                   before being killed
    #   @param **kwargs Additional (unused) keyword arguments
    #   @return Predicted response time
    def predict(self, arrival_rate, warm_service_time, cold_service_time,
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

