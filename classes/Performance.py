from abc import ABC, abstractmethod

## PerformanceEvaluator
#
# Class used to evaluate the performance of a specific Resource type
class PerformanceEvaluator(ABC):
    
    ## Method to evaluate the performance of a Resource (abstract)
    #   @param self The object pointer
    @abstractmethod
    def evaluate(self):
        pass


## NetworkPE
#
# Specialization of PerformanceEvaluator designed to evaluate the performance 
# of a NetworkTechnology object
class NetworkPE(PerformanceEvaluator):

    ## Method to evaluate the performance of a NetworkTechnology object
    #
    #   @param self The object pointer
    #   @param access_delay
    #   @param bandwidth
    #   @param data Amount of data transferred
    #   @return Computed performance
    def evaluate(self, access_delay, bandwidth,data):
        return access_delay + (data / bandwidth)
    

## ServerFarmPE
#
# Specialization of PerformanceEvaluator designed to evaluate the performance 
# of a server farm (i.e., a group of VirtualMachine objects)
class ServerFarmPE(PerformanceEvaluator):
    
    
    ## Method to compute the utilization of a specific VirtualMachine object
    #
    #   @param self The object pointer
    #   @param I Number of existing Component objects
    #   @param j Index of the EdgeNode object
    #   @param Y_hat Amount of Resources assigned to each Component, 
    #          returned by Configuration.get_number_matrix()
    #   @param D Demand matrix
    #   @param Lambdas Array of Component.Lambda for all Component objects
    #
    #   @return Utilization of the given VirtualMachine object
    def compute_utilization(self, I, j, Y_hat, D, Lambdas):
        utilization = 0
        
        for i in range(I):
            if Y_hat[i,j]>0:
                 
                utilization += D[i,j] * Lambdas[i] / Y_hat[i,j]
        return utilization
    
    
    ## Method to evaluate the performance of a specific Component deployed 
    #  onto a VirtualMachine object
    #
    #   @param self The object pointer
    #   @param i Component index
    #   @param j Resource index
    #   @param Y_hat Amount of Resources assigned to each Component, 
    #          returned by Configuration.get_number_matrix()
    #   @param D Demand matrix
    #   @param Lambdas Array of Component.Lambda for all Component objects
    #
    #   @return Computed performance
    def evaluate_component(self, i, j, Y_hat, D, Lambdas):
        
        # number of components and number of resources
        I, J = Y_hat.shape
        
        # compute utilization of the given EdgeNode object
        utilization = self.compute_utilization(I, j, Y_hat, D, Lambdas)
              
        #return D[i,j] / (1 - utilization)
        return D[i,j] / (1 - utilization)
            
    
    ## Method to evaluate the performance of all Components deployed onto 
    #  a server farm
    #
    #   @param self The object pointer
    #   @param assignments List of pairs representing assignments; either all 
    #          the assignments using Cloud resources or a subset of 
    #          assignments living onto a specific path
    #   @param Y_hat Amount of Resources assigned to each Component, 
    #          returned by Configuration.get_number_matrix()
    #   @param D Demand matrix
    #   @param Lambdas Array of Component.Lambda for all Component objects
    #
    #   @return Computed performance
    def evaluate(self, assignments, Y_hat, D, Lambdas):
       
        # number of components and number of resources
        I, J = Y_hat.shape
        
        # evaluate performance of all components, given the corresponding
        # assignment
        value = 0
        for assignment in assignments:
            i = assignment[0]
            j = assignment[1]
            value += self.evaluate_component(i, j, Y_hat, D, Lambdas)
              
        return value
                

## FunctionPE
#
# Specialization of PerformanceEvaluator designed to evaluate the performance 
# of a FaaS object
class FunctionPE(PerformanceEvaluator):
    
    
    ## Method to evaluate the performance of a specific Component deployed 
    #  onto a FaaS object
    #
    #   @param self The object pointer
    #   @param i Component index
    #   @param j Resource index
    #   @param D Demand matrix
    #
    #   @return Computed performance
    def evaluate_component(self, i, j, D):
        return D[i,j]
    
    
    ## Method to evaluate the performance of all Components deployed onto 
    #  FaaS objects
    #
    #   @param self The object pointer
    #   @param assignments List of pairs representing assignments; either all 
    #          the assignments using FaaS resources or a subset of 
    #          assignments living onto a specific path
    #   @param D Demand matrix
    #
    #   @return Computed performance
    def evaluate(self, assignments, D): 
        value = 0
        for assignment in assignments:
            i = assignment[0]
            j = assignment[1]
            value += self.evaluate_component(i, j, D)
        return value


## EdgePE
#
# Specialization of PerformanceEvaluator designed to evaluate the performance 
# of all edge Resources
class EdgePE(PerformanceEvaluator):
    
    ## Method to compute the utilization of a specific EdgeNode object
    #
    #   @param self The object pointer
    #   @param I Number of existing Component objects
    #   @param j Index of the EdgeNode object
    #   @param Y Assignment matrix returned by 
    #          Configuration.get_assignment_matrix()
    #   @param D Demand matrix
    #   @param Lambdas Array of Component.Lambda for all Component objects
    #
    #   @return Utilization of the given EdgeNode object
    def compute_utilization(self, I, j, Y, D, Lambdas):
        utilization = 0
        for i in range(I):
            utilization += D[i,j] * Y[i,j] * Lambdas[i]
        return utilization
    
    
    ## Method to evaluate the performance of a specific Component deployed 
    #  onto an EdgeNode object
    #
    #   @param self The object pointer
    #   @param i Component index
    #   @param j Resource index
    #   @param Y Assignment matrix returned by 
    #          Configuration.get_assignment_matrix()
    #   @param D Demand matrix
    #   @param Lambdas Array of Component.Lambda for all Component objects
    #
    #   @return Computed performance
    def evaluate_component(self, i, j, Y, D, Lambdas):
        
        # number of components and number of resources
        I, J = Y.shape
        
        # compute utilization of the given EdgeNode object
        utilization = self.compute_utilization(I, j, Y, D, Lambdas)
              
        return D[i,j] / (1 - utilization)
    
    
    ## Method to evaluate the performance of all Components deployed onto 
    #  edge resources
    #
    #   @param self The object pointer
    #   @param assignments List of pairs representing assignments; either all 
    #          the assignments using Edge resources or a subset of 
    #          assignments living onto a specific path
    #   @param Y Assignment matrix returned by 
    #          Configuration.get_assignment_matrix()
    #   @param D Demand matrix
    #   @param Lambdas Array of Component.Lambda for all Component objects
    #
    #   @return Computed performance
    def evaluate(self, assignments, Y, D, Lambdas):
        
        # number of components and number of resources
        I, J = Y.shape
        
        # evaluate performance of all components, given the corresponding
        # assignment
        value = 0
        for assignment in assignments:
            i = assignment[0]
            j = assignment[1]
            value += self.evaluate_component(i, j, Y, D, Lambdas)
              
        return value

