from abc import ABC, abstractmethod
import importlib


## FaaSPredictor
#
# Abstract class used to represent a performance model for predicting the 
# response time of a Resources.FaaS instance
class FaaSPredictor(ABC):
    
    ## @var predictor_module
    # Module that implements the method used to predict the 
    # Resources.FaaS performance
    
    ## @var predictor
    # Object that performs the prediction
    
    ## FaaSPredictor class constructor
    #   @param self The object pointer
    #   @param module_name Name of the module to be loaded
    def __init__(self, module_name):
        self.module_name = module_name
        self.predictor = None

    ## Method to evaluate the object performance through the class predictor
    #   @param self The object pointer
    #   @param arrival_rate Arrival rate of requests
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    #   @param time_out Time spent by an idle Resources.FaaS instance 
    #                   before being killed
    @abstractmethod
    def predict(self, arrival_rate, warm_service_time, cold_service_time,
                time_out):
        pass



## FaaSPredictorPacsltk
#
# Specialization of FaaSPredictor class to predict a Resources.FaaS instance 
# response time relying on the method implemented in pacsltk module
class FaaSPredictorPacsltk(FaaSPredictor):
    
    ## @var predictor
    # Object that performs the prediction
    
    ## FaaSPredictorPacsltk class constructor
    #   @param self The object pointer
    def __init__(self):
        super().__init__("pacsltk.perfmodel")
        predictor_module = importlib.import_module(self.module_name)
        self.predictor = predictor_module.get_sls_warm_count_dist

    ## Method to evaluate the object performance through the class predictor
    #   @param self The object pointer
    #   @param arrival_rate Arrival rate of requests
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    #   @param time_out Time spent by an idle Resources.FaaS instance 
    #                   before being killed
    #   @return Predicted response time
    def predict(self, arrival_rate, warm_service_time, cold_service_time,
                time_out):
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
    def __init__(self, regressor_file):
        super().__init__("a-MLlibrary.model_building.predictor")
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
    #   @return Predicted response time
    def predict(self, arrival_rate, warm_service_time, cold_service_time,
                time_out):
        pd = importlib.import_module("pandas")
        columns = "Lambda,warm_service_time,cold_service_time,expiration_time".split(",")
        data = pd.DataFrame(data=[[arrival_rate, 
                                   warm_service_time,
                                   cold_service_time,
                                   time_out]],
                            columns=columns)
        return self.predictor.predict_from_df(data, True)


