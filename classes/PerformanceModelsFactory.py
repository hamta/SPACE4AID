from classes.PerformanceModels import FaaSPredictorPacsltk, FaaSPredictorMLlib## PerformanceModelsFactory## Class to build a factory of performance models, characterized by a given # key and used to predict the response time of Graph.Component.Partition # objects executed without resource contention on different Resources.Resource class PerformanceModelsFactory:        ## @var models    # Dictionary of available predictors        ## PerformanceModelsFactory class constructor    def __init__(self):        self.models = {}    ## Method to register a new model in the dictionary of available     # predictors    #   @param self The object pointer    #   @param key The key used to identify the performance model    #   @param model The performance model    def register_model(self, key, model):        self.models[key] = model    ## Method to initialize a new model from the factory    #   @param self The object pointer    #   @param key The key used to identify the performance model    #   @param **kwargs List of all <parameter_name>=<parameter_value> pairs     #                   that are required to initialize the model    #   @return The performance model    def create(self, key, **kwargs):        model = self.models.get(key)        if not model:            raise ValueError(key)        return model(**kwargs)## Factory initializationPMfactory = PerformanceModelsFactory()PMfactory.register_model("PACSLTK", FaaSPredictorPacsltk)PMfactory.register_model("MLLIB", FaaSPredictorMLlib)