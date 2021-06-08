## NetworkTechnology
class NetworkDomain:

    ## @var access_delay

    ## @var bandwidth
    
    ## NetworkTechnology class constructor
    #   @param self The object pointer
    #   @param ND_name the name of network domain
    #   @param computationallayers a list of corresponding 
    #   computational layers located in the network domain 
    #   @param access_delay
    #   @param bandwidth
    #   @param performance_evaluator
    def __init__(self, ND_name,computationallayers, access_delay, bandwidth,performance_evaluator):
        self.ND_name=ND_name
        self.computationallayers = computationallayers
        self.access_delay = access_delay
        self.bandwidth = bandwidth
        self.performance_evaluator = performance_evaluator
        
    ## Operator to convert a NetworkTechnology object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '"AccessDelay":{}, "Bandwidth":{}'.\
            format(self.access_delay, self.bandwidth)
        return s
