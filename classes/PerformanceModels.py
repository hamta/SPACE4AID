import importlib

class FaaSPredictorPacsltk:

    ## FaaSPredictorPacsltk class constructor
    #   @param self The object pointer
    def __init__(self):
        self.predictor = importlib.import_module("pacsltk.perfmodel")

    ## Method to evaluate the object performance through the class predictor
    #   @param self The object pointer
    #   @param arrival_rate Arrival rate of requests
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    #   @param time_out Time spent by an idle FaaS instance before being killed
    def predict(self, arrival_rate, warm_service_time, cold_service_time,
                time_out, **_ignored):
        return self.predictor.get_sls_warm_count_dist(arrival_rate,
                                                      warm_service_time, 
                                                      cold_service_time, 
                                                      time_out)

    ## Method to get the response time (one field returned by predict)
    #   @param self The object pointer
    #   @param arrival_rate Arrival rate of requests
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    #   @param time_out Time spent by an idle FaaS instance before being killed
    def get_response_time(self, arrival_rate, warm_service_time, 
                          cold_service_time, time_out, **_ignored):
        return self.predict(arrival_rate, warm_service_time, cold_service_time,
                            time_out)[0]['avg_resp_time']



