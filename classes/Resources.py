from pacsltk import perfmodel
import json

## ComputationalLayer
#
# Class to represent a computational layer (namely a list of resources of a
# fixed type)
class ComputationalLayer:

    ## @var resources
    # List of Resource objects available in this layer
    
    ## ComputationalLayer class constructor
    #   @param self The object pointer
    #   @param performance_evaluator A specialized PerformanceEvaluator,
    #          whose type depends on the type of Resources stored in the 
    #          ComputationalLayer
    def __init__(self, name  ):
        self.name=name
        self.resources= []
        
       
    
    ## Method to add a given Resource to the ComputationalLayer
    #   @param self The object pointer
    #   @param resource Resources to be added
    def add_resource(self, resource_idx):
        self.resources.append(resource_idx)
     
    
    ## Operator to convert a ComputationalLayer object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '{'
        for resource in self.resources:
            s += (str(resource) + ',')
        s = s[:-1] + '}'
        s = json.dumps(json.loads(s))
        return s


## Resource
#
# Class to represent a generic resource
class Resource:

    ## @var name
    # Name of the resource (used to uniquely identify it)

    ## @var cost
    # Cost of the resource

    ## @var memory
    # Amount of available memory on the resource
    
    ## Resource class constructor
    #   @param self The object pointer
    #   @param CLname the corresponding computational layer of the resource 
    #   @param name
    #   @param cost
    #   @param memory
    def __init__(self, CLname,name, cost, memory,performance_evaluator):
        self.name = name
        self.cost = cost
        self.memory = memory
        self.CLname=CLname
        self.performance_evaluator = performance_evaluator
    ## Operator used to check if two Resource objects are equal
    #
    # It compares the corresponding names
    #   @param self The object pointer
    #   @param other The rhs Resource
    def __eq__(self, other):
        return self.name == other.name
    
    ## Operator to convert a Resource object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '"{}": {{"cost":{}, "memory":{}}}'.\
            format(self.name, self.cost, self.memory)
        return s
    

## VirtualMachine
#
# Class to represent a Virtual Machine (inherits from Resource)
class VirtualMachine(Resource):

    ## @var number
    # Number of machines

    ## VirtualMachine class constructor
    #   @param self The object pointer
    #   @param CLname the corresponding computational layer of the resource 
    #   @param name
    #   @param cost
    #   @param memory
    #   @param number
    def __init__(self, CLname, name, cost, memory, performance_evaluator, number):
        super().__init__(CLname, name, cost, memory, performance_evaluator)
        self.number = number
    
    ## Operator to convert an VirtualMachine object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '"{}": {{"cost":{}, "memory":{}, "number":{}}}'.\
            format(self.name, self.cost, self.memory, self.number)
        return s
    

## EdgeNode
#
# Class to represent an EdgeNode (inherits from Resource)
class EdgeNode(Resource):

    ## EdgeNode class constructor
    #   @param self The object pointer
    #   @param CLname the corresponding computational layer of the resource 
    #   @param name
    #   @param cost
    #   @param memory
    #   @param number
    def __init__(self, CLname, name, cost, memory, performance_evaluator, number):
        super().__init__(CLname, name, cost, memory, performance_evaluator)
        self.number = number

## FaaS
#
# Class to represent a FaaS object (inherits from Resource)
class FaaS(Resource):

    ## @var transition_cost
    # Transition cost of the functions (equal for all objects of this class)

    ## @var warm_service_time
    # Warm service time of the function

    ## @var cold_service_time
    # Cold service time of the function

    ## @var idle_time_before_kill
    # Time spent by an idle function before being killed
    
    ## FaaS class constructor
    #   @param self The object pointer
    #   @param CLname the corresponding computational layer of the resource 
    #   @param name
    #   @param cost
    #   @param memory
    #   @param transition_cost
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    #   @param idle_time_before_kill How long does the platform keep the 
    #          servers after being idle
    def __init__(self, CLname, name, cost, memory, performance_evaluator, transition_cost, 
                  idle_time_before_kill):
        super().__init__(CLname, name, cost, memory, performance_evaluator)
        self.transition_cost = transition_cost
        # self.warm_service_time = warm_service_time
        # self.cold_service_time = cold_service_time
        self.idle_time_before_kill = idle_time_before_kill
    
    ## Method to compute the average response time given the arrival rate of
    # requests
    #   @param self The object pointer
    #   @param arrival_rate
    def get_avg_res_time(self, arrival_rate, warm_service_time, cold_service_time):
        perf = perfmodel.get_sls_warm_count_dist(arrival_rate,
                                                 warm_service_time, 
                                                 cold_service_time, 
                                                 self.idle_time_before_kill)
        return perf[0]['avg_resp_time']
    
  
    
    ## Operator to convert a FaaS object into a string
    #   @param self The object pointer
    def __str__(self):
        s1 = '"{}": {{"cost":{}, "memory":{}, "transition_cost":{}, '.\
             format(self.name, self.cost, self.memory, self.transition_cost)
        s2 = '"warm_service_time":{}, "cold_service_time":{}, '.\
             format(self.warm_service_time, self.cold_service_time)
        s3 = '"idle_time_before_kill":{}}},'.\
             format(self.idle_time_before_kill)
        s4 = '"transition_cost":{}'.format(self.transition_cost)
        return s1 + s2 + s3 + s4
