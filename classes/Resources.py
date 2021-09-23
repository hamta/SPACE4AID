from pacsltk import perfmodel
import json


## ComputationalLayer
#
# Class to represent a computational layer (namely a list of resources of a
# fixed type)
class ComputationalLayer:

    ## @var name
    # Name of the current layer (used to uniquely identify it)
    
    ## @var resources
    # List of indices of the available Resources.Resource objects
    
    ## ComputationalLayer class constructor
    #   @param self The object pointer
    #   @param name Name of the current layer
    def __init__(self, name):
        self.name = name
        self.resources = []
    
    ## Method to add a given Resources.Resource to the layer
    #   @param self The object pointer
    #   @param resource_idx Index of the Resources.Resource to be added
    def add_resource(self, resource_idx):
        self.resources.append(resource_idx)
    
    ## Operator to convert a ComputationalLayer object into a string
    #   @param self The object pointer
    #   @param all_resources List of all Resources.Resource objects available 
    #                        in the system
    def __str__(self, all_resources):
        s = '{"' + self.name + '": {'
        for resource_idx in self.resources:
            s += (str(all_resources[resource_idx]) + ',')
        s = s[:-1] + '}}'
        s = json.dumps(json.loads(s))
        s = s[1:-1]
        return s


## Resource
#
# Class to represent a generic resource
class Resource:

    ## @var CLname
    # Name of the Resources.ComputationalLayer the resource belongs to

    ## @var name
    # Name of the resource (used to uniquely identify it)

    ## @var cost
    # Cost of the resource

    ## @var memory
    # Amount of available memory on the resource
    
    ## @var performance_evaluator
    # Object of class Performance.PerformanceEvaluator
    
    ## Resource class constructor
    #   @param self The object pointer
    #   @param CLname the corresponding computational layer of the resource 
    #   @param name Name of the resource
    #   @param cost Cost of the resource
    #   @param memory Amount of available memory on the resource
    #   @param performance_evaluator Object of class 
    #                                Performance.PerformanceEvaluator
    def __init__(self, CLname, name, cost, memory, performance_evaluator):
        self.CLname = CLname
        self.name = name
        self.cost = cost
        self.memory = memory
        self.performance_evaluator = performance_evaluator
    
    ## Operator used to check if two Resource objects are equal, comparing 
    # the corresponding names
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
    #   @param CLname the corresponding computational layer
    #   @param name Name of the VirtualMachine
    #   @param cost Cost of the VirtualMachine
    #   @param memory Amount of available memory on the VirtualMachine
    #   @param performance_evaluator Object of class Performance.ServerFarmPE
    #   @param number Number of available VirtualMachine objects of the 
    #                 current type
    def __init__(self, CLname, name, cost, memory, performance_evaluator, 
                 number):
        super().__init__(CLname, name, cost, memory, performance_evaluator)
        self.number = number
    
    ## Method to evaluate the performance of a specific 
    # Graph.Component.Partition object executed onto a Resources.VirtualMachine
    #   @param self The object pointer
    #   @param i Index of the Graph.Component
    #   @param h Index of the Graph.Component.Partition
    #   @param j Index of the Resources.VirtualMachine
    #   @param Y_hat Matix denoting the amount of Resources assigned to each 
    #                Graph.Component object
    #   @param S A System.System object
    #   @return Response time
    def evaluate_partition(self, i, h, j, Y_hat, S):
        return self.performance_evaluator.evaluate(i,h,j,Y_hat,S)
    
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
    
    ## @var number
    # Number of nodes

    ## EdgeNode class constructor
    #   @param self The object pointer
    #   @param CLname the corresponding computational layer
    #   @param name Name of the EdgeNode
    #   @param cost Cost of the EdgeNode
    #   @param memory Amount of available memory on the EdgeNode
    #   @param performance_evaluator Object of class Performance.EdgePE
    #   @param number Number of available EdgeNode objects of the current type
    def __init__(self, CLname, name, cost, memory, performance_evaluator, number):
        super().__init__(CLname, name, cost, memory, performance_evaluator)
        self.number = number
    
    ## Method to evaluate the performance of a specific 
    # Graph.Component.Partition object executed onto a Resources.EdgeNode
    #   @param self The object pointer
    #   @param i Index of the Graph.Component
    #   @param h Index of the Graph.Component.Partition
    #   @param j Index of the Resources.EdgeNode
    #   @param Y Assignment matrix
    #   @param S A System.System object
    #   @return Response time
    def evaluate_partition(self, i, h, j, Y, S):
        return self.performance_evaluator.evaluate(i,h,j,Y,S)
    
    ## Operator to convert an EdgeNode object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '"{}": {{"cost":{}, "memory":{}, "number":{}}}'.\
            format(self.name, self.cost, self.memory, self.number)
        return s


## FaaS
#
# Class to represent a FaaS object (inherits from Resource)
class FaaS(Resource):

    ## @var transition_cost
    # Transition cost of the functions (equal for all objects of this class)

    ## @var idle_time_before_kill
    # Time spent by an idle function before being killed
    
    ## FaaS class constructor
    #   @param self The object pointer
    #   @param CLname Name of the corresponding computational layer
    #   @param name Name of the FaaS instance
    #   @param cost Cost of the FaaS instance
    #   @param memory Amount of available memory on the FaaS instance
    #   @param performance_evaluator Object of class Performance.FunctionPE
    #   @param transition_cost Transition cost
    #   @param idle_time_before_kill How long does the platform keep the 
    #                                servers up after being idle
    def __init__(self, CLname, name, cost, memory, performance_evaluator, 
                 transition_cost, idle_time_before_kill):
        super().__init__(CLname, name, cost, memory, performance_evaluator)
        self.transition_cost = transition_cost
        self.idle_time_before_kill = idle_time_before_kill
    
    ## Method to compute the average response time given the arrival rate of
    # requests
    #   @param self The object pointer
    #   @param arrival_rate Arrival rate of requests
    #   @param warm_service_time Response time for warm start requests
    #   @param cold_service_time Response time for cold start requests
    def get_avg_res_time(self, arrival_rate, warm_service_time, 
                         cold_service_time):
        perf = perfmodel.get_sls_warm_count_dist(arrival_rate,
                                                 warm_service_time, 
                                                 cold_service_time, 
                                                 self.idle_time_before_kill)
        return perf[0]['avg_resp_time']
    
    ## Method to evaluate the performance of a specific 
    # Graph.Component.Partition object executed onto a Resources.FaaS object
    #   @param self The object pointer
    #   @param i Index of the Graph.Component
    #   @param h Index of the Graph.Component.Partition
    #   @param j Index of the Resources.FaaS object
    #   @param S A System.System object
    #   @return Response time
    def evaluate_partition(self, i, h, j, S):
        return self.performance_evaluator.evaluate(i,h,j,S)
    
    ## Operator to convert a FaaS object into a string
    #   @param self The object pointer
    def __str__(self):
        s1 = '"{}": {{"cost":{}, "memory":{}, '.\
             format(self.name, self.cost, self.memory)
        s2 = '"idle_time_before_kill":{}}},'.\
             format(self.idle_time_before_kill)
        s3 = '"transition_cost":{}'.format(self.transition_cost)
        return s1 + s2 + s3

