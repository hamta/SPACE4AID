from classes.Logger import Logger
from abc import ABC, abstractmethod
import numpy as np
import sys



## NetworkPerformanceEvaluator
#
# Class designed to evaluate the performance of a NetworkTechnology object, 
# namely the time required to transfer data between two consecutive 
# Graph.Component or Graph.Component.Partition objects executed on 
# different devices in the same network domain
class NetworkPerformanceEvaluator:

    ## Method to evaluate the performance of a NetworkTechnology object
    #   @param self The object pointer
    #   @param access_delay Access delay characterizing the network domain
    #   @param bandwidth Bandwidth characterizing the network domain
    #   @param data Amount of data transferred
    #   @return Network transfer time
    def evaluate(self, access_delay, bandwidth, data):
        return access_delay + (data / bandwidth)



## QTPerformanceEvaluator
#
# Abstract class used to represent a performance evaluator, namely an object 
# that evaluates the performance of a Graph.Component.Partition executed on 
# different types of resources, exploiting the M/G/1 queue model
class QTPerformanceEvaluator(ABC):
    
    ## @var keyword
    # Keyword identifying the evaluator
    
    ## QTPerformanceEvaluator class constructor
    #   @param self The object pointer
    #   @param keyword Keyword identifying the evaluator
    def __init__(self, keyword):
        self.keyword = keyword

    ## Method to evaluate the performance of a specific 
    # Graph.Component.Partition object executed onto a specific 
    # Resources.Resource
    #   @param self The object pointer
    #   @param i Index of the Graph.Component
    #   @param h Index of the Graph.Component.Partition
    #   @param j Index of the Resources.Resource
    #   @param Y Assignment matrix
    #   @param S A System.System object
    #   @return Response time
    @abstractmethod
    def evaluate(self, i, h, j, Y, S):
        pass
    
    ## Operator to convert a QTPerformanceEvaluator object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '"model":"{}"'.\
            format(self.keyword)
        return s
    

## ServerFarmPE
#
# Class designed to evaluate the performance of a Graph.Component.Partition 
# object executed in a server farm (i.e., a group of Resources.VirtualMachine 
# objects)
class ServerFarmPE(QTPerformanceEvaluator):
    
    ## ServerFarmPE class constructor
    #   @param self The object pointer
    def __init__(self):
        super(ServerFarmPE, self).__init__("QTcloud")
    
    ## Method to compute the utilization of a specific 
    # Resources.VirtualMachine object
    #   @param self The object pointer
    #   @param j Index of the Resources.VirtualMachine object
    #   @param Y_hat Matrix denoting the amount of Resources assigned to each 
    #                Graph.Component.Partition object
    #   @param S A System.System object
    #   @return Utilization of the given Resources.VirtualMachine object
    def compute_utilization(self, j, Y_hat, S):
        utilization = 0
        # loop over all components
        for c in S.components:
            # loop over all partitions in the component
            for p in c.partitions:
                # get the corresponding indices
                i = S.dic_map_part_idx[c.name][p.name][0]
                h = S.dic_map_part_idx[c.name][p.name][1]
                # compute the utilization
                if Y_hat[i][h,j] > 0:
                    utilization += S.demand_matrix[i][h,j] * \
                                    p.part_Lambda / Y_hat[i][h,j]
                                        
        return utilization
    
    ## Method to evaluate the performance of a specific 
    # Graph.Component.Partition object executed onto a Resources.VirtualMachine
    #   @param self The object pointer
    #   @param i Index of the Graph.Component
    #   @param h Index of the Graph.Component.Partition
    #   @param j Index of the Resources.VirtualMachine
    #   @param Y_hat Matrix denoting the amount of Resources assigned to each 
    #                Graph.Component.Partition object
    #   @param S A System.System object
    #   @return Response time
    def evaluate(self, i, h, j, Y_hat, S):
        # compute the utilization
        utilization = self.compute_utilization(j, Y_hat, S)
        # compute the response time
        r = 0.
        if Y_hat[i][h,j] > 0:
            r = S.demand_matrix[i][h,j] / (1 - utilization) 
        return r


## EdgePE
#
# Class designed to evaluate the performance of a Graph.Component.Partition  
# object executed on a Resources.EdgeNode 
class EdgePE(QTPerformanceEvaluator):
    
    ## EdgePE class constructor
    #   @param self The object pointer
    def __init__(self):
        super(EdgePE, self).__init__("QTedge")
    
    ## Method to compute the utilization of a specific 
    # Resources.EdgeNode object
    #   @param self The object pointer
    #   @param j Index of the Resources.EdgeNode object
    #   @param Y Assignment matrix
    #   @param S A System.System object
    #   @return Utilization of the given Resources.EdgeNode object
    def compute_utilization(self, j, Y, S):
        utilization = 0
        # loop over all components
        for c in S.components:
            # loop over all partitions in the component
            for p in c.partitions:
                # get the corresponding indices
                i = S.dic_map_part_idx[c.name][p.name][0]
                h = S.dic_map_part_idx[c.name][p.name][1]
                # compute the utilization
                utilization += S.demand_matrix[i][h,j] * \
                                Y[i][h,j] * p.part_Lambda
        return utilization
    
    ## Method to evaluate the performance of a specific 
    # Graph.Component.Partition object executed onto a Resources.EdgeNode
    #   @param self The object pointer
    #   @param i Index of the Graph.Component
    #   @param h Index of the Graph.Component.Partition
    #   @param j Index of the Resources.EdgeNode
    #   @param Y Assignment matrix
    #   @param S A System.System object
    #   @return Response time
    def evaluate(self, i, h, j, Y, S):
        # compute utilization
        utilization = self.compute_utilization(j, Y, S)
        # compute response time
        return S.demand_matrix[i][h,j] * Y[i][h,j] / (1 - utilization)



## SystemPerformanceEvaluator
#
# Class used to evaluate the performance of a Graph.Component object given 
# the information about the Resources.Resource where it is executed
class SystemPerformanceEvaluator:
    
    ## @var logger
    # Object of Logger type, used to print general messages
    
    
    ## SystemPerformanceEvaluator class constructor
    #   @param self The object pointer
    #   @param log Object of Logger type
    def __init__(self, log=Logger()):
        self.logger = log
    
    
    ## Method to evaluate the response time of the Graph.Component object 
    # identified by the given index
    #   @param self The object pointer
    #   @param S A System.System object
    #   @param Y_hat Matrix denoting the amount of Resources assigned to each 
    #                Graph.Component.Partition object
    #   @param c_idx The index of the current component
    #   @return Response time
    def get_perf_evaluation(self, S, Y_hat, c_idx):
        
        # check if the memory constraints are satisfied
        self.logger.log("Evaluating component {}".format(c_idx), 5)
        
        # initialize response time
        perf_evaluation = 0
        
        # get the indices of the resource where the partitions of the current 
        # component are executed
        j = np.nonzero(Y_hat[c_idx])
        
        # loop over all partitions
        self.logger.level += 1
        self.logger.log("Evaluating partition response times", 6)
        self.logger.level += 1
        for h in range(len(j[0])):
            # evaluate the response time (note: Y_hat and the assignment 
            # matrix Y coincide for Resources.EdgeNode objects)
            p_idx = j[0][h]
            r_idx = j[1][h]
            if r_idx < S.FaaS_start_index:
                p = S.performance_models[c_idx][p_idx][r_idx].evaluate(c_idx,
                                                                       p_idx,
                                                                       r_idx,
                                                                       Y_hat, 
                                                                       S)
            else:
                p = S.demand_matrix[c_idx][p_idx,r_idx]
            self.logger.log("{} --> {}".format(h, p), 7)
            perf_evaluation += p
            # check that the response time is not negative
            if perf_evaluation < 0:
                return float("inf")
        self.logger.level -= 1
        self.logger.log("time --> {}".format(perf_evaluation), 6)
        
        # compute the network transfer time among partitions
        self.logger.log("Evaluating network delay", 6)
        if len(j[0]) > 1:
            self.logger.level += 1
            network_delay = 0
            # loop over all partitions
            for h in range(len(j[0]) - 1):
                # get the data transferred from the partition
                data_size = S.components[c_idx].partitions[j[0][h]].data_size
                # compute the network transfer time
                nd = self.get_network_delay(j[1][h], j[1][h+1], S, data_size)
                self.logger.log("{} --> {}".format(h, nd), 7)
                network_delay += nd
            # update the response time
            perf_evaluation += network_delay
            self.logger.level -= 1
            self.logger.log("time --> {}".format(network_delay), 6)
        
        self.logger.level -= 1
        self.logger.log("time --> {}".format(perf_evaluation), 5)
        
        return perf_evaluation

    
    ## Method to evaluate the response time of the all Graph.Component objects
    #   @param self The object pointer
    #   @param S A System.System object
    #   @param Y_hat Matrix denoting the amount of Resources assigned to each 
    #                Graph.Component.Partition object
    #   @return 1D numpy array with the response times of all components
    def compute_performance(self, S, Y_hat):
        I = len(Y_hat)
        response_times = np.full(I, np.inf)
        for i in range(I):
            response_times[i] = self.get_perf_evaluation(S, Y_hat, i)
        return response_times


    ## Static method to compute the network delay due to data transfer 
    # operations between two consecutive components (or partitions), executed 
    # on different resources in the same network domain
    #   @param self The object pointer
    #   @param cpm1_resource Resource index of first component
    #   @param cpm2_resource Resource index of second component
    #   @param S A System.System object
    #   @param data_size Amount of transferred data
    #   @return Network transfer time
    def get_network_delay(self, cpm1_resource, cpm2_resource, S, data_size):
       
        # get the names of the computational layers where the two resources 
        # are located
        CL1 = S.resources[cpm1_resource].CLname
        CL2 = S.resources[cpm2_resource].CLname
        
        # get the lists of network domains containing the two computational 
        # layers and compute their intersection
        ND1 = list(filter(lambda NT: (CL1 in NT.computationallayers), S.network_technologies))
        ND2 = list(filter(lambda NT: (CL2 in NT.computationallayers), S.network_technologies))
        ND = list(set(ND1).intersection(ND2))
      
        # there must exist a common network domain, otherwise the components
        # cannot communicate with each other
        if len(ND) == 0:
            print("ERROR: no network domain available between two resources "
              + str(cpm1_resource) + " and " + str(cpm2_resource)) 
            sys.exit(1)
        # if only one domain is common to the two layers, evaluate the 
        # network delay on that domain
        elif len(ND) == 1:
            network_delay = ND[0].evaluate_performance(data_size)
        else:
            # else, the network transfer time is the minimum among the times 
            # required with the different network domains
            network_delay = float("inf")
            for nd in ND:
                new_network_delay = nd.evaluate_performance(data_size)
                if new_network_delay < network_delay:
                   network_delay = new_network_delay 
        
        return network_delay

