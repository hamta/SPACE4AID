from classes.Logger import Logger
from classes.Solution import Configuration, Result, EliteResults
import numpy as np
import copy
import sys


## Algorithm
class Algorithm:
    
    ## @var system
    # A System.System object
    
    ## @var logger
    # Object of Logger.Logger type, used to print general messages
    
    ## @var error
    # Object of Logger class, used to print error messages on sys.stderr
    
    ## Algorithm class constructor: initializes the system
    #   @param self The object pointer
    #   @param system A System.System object
    #   @param log Object of Logger.Logger type
    def __init__(self, system, log = Logger()):
        self.logger = log
        self.error = Logger(stream=sys.stderr, verbose=1)
        self.system = system
        
    
    ## Method to create the initial random solution
    #   @param self The object pointer
    #   @return (1) List of 2D numpy matrices denoting the amount of  
    #           Resources.Resource assigned to each 
    #           Graph.Component.Partition object
    #           (2) List of lists that store the random numbers used to select 
    #           the resource assigned to each component partition in the chosen
    #           deployment
    #           (3) List of randomly selected numbers of each used edge/cloud 
    #           resource
    #           (4) List of indices of the resource randomly selected in each 
    #           computational layer
    def create_random_initial_solution(self):
      
        # increase indentation level for logging
        self.logger.level += 1
        
        # initialize the assignments
        self.logger.log("Initialize matrices", 4)
        CL_res_random = []
        res_parts_random = []
        y_hat = []
        y = []
        
        # loop over all components
        I = len(self.system.components)
        for i in range(I):
            # get the number of partitions and available resources
            H, J = self.system.compatibility_matrix[i].shape
            # create the empty matrices
            y_hat.append(np.full((H, J), 0, dtype=int))
            y.append(np.full((H, J), 0, dtype=int))
        
        # generate the list of candidate nodes, selecting one node per each 
        # computational layer (and all nodes in the FaaS layers)
        self.logger.log("Generate candidate resources", 4)
        candidate_nodes = []
        resource_count = 0
        # loop over all computational layers
        for l in self.system.CLs:
            # select all nodes in FaaS layers
            if resource_count >= self.system.FaaS_start_index:
                random_num = l.resources
                candidate_nodes.extend(random_num)
            # randomly select a node in other layers
            else:
                random_num = np.random.choice(l.resources)
                CL_res_random.append(l.resources.index(random_num))
                candidate_nodes.append(random_num)
            resource_count += len(l.resources)
        
        # loop over all components
        self.logger.log("Assign components", 4)
        for comp in self.system.components:
            
            # randomly select a deployment for that component
            random_dep = np.random.choice(comp.deployments)
            h = 0
            rand = []
            # loop over all partitions in the deployment
            for part_idx in random_dep.partitions_indices:
                part = comp.partitions[part_idx]
                # get the indices of the component and the deployment
                i = self.system.dic_map_part_idx[comp.name][part.name][0]
                h_idx = self.system.dic_map_part_idx[comp.name][part.name][1]
                # get the indices of compatible resources and compute the 
                # intersection with the selected resources in each 
                # computational layer
                idx = np.nonzero(self.system.compatibility_matrix[i][h_idx,:])[0]
                index = list(set(candidate_nodes).intersection(idx))
                # randomly extract a resource index in the intersection
                prob = 1/len(index)
                step = 0
                rn = np.random.random()
                rand.append(rn)
                for r in np.arange(0, 1, prob):
                    if rn > r and rn <= r + prob:
                        j = index[step]
                    else:
                        step += 1
                y[i][h_idx,j] = 1
                y_hat[i][h_idx,j] = 1
                # if the partition is the last partition (i.e., its successor 
                # is the successor of the component), update the size of 
                # data transferred between the components
                if self.system.graph.G.succ[comp.name] != {}:
                    if part.Next == list(self.system.graph.G.succ[comp.name].keys())[0]:
                        self.system.graph.G[comp.name][part.Next]["data_size"] = part.data_size
                
            res_parts_random.append(rand)
        
        # loop over edge/cloud resources
        self.logger.log("Set number of resources", 4)
        VM_numbers = [] 
        for j in range(self.system.FaaS_start_index):
            # randomly generate the number of resources that can be assigned 
            # to the partitions that run on that resource
            number = np.random.randint(1, self.system.resources[j].number + 1)
            VM_numbers.append(number - 1)
            # loop over components
            for i in range(I):
                # get the number of partitions
                H = self.system.compatibility_matrix[i].shape[0]
                # loop over the partitions
                for h in range(H):
                    # if the partition runs on the current resource, update 
                    # the number
                    if y[i][h][j] > 0:
                        y_hat[i][h][j] = y[i][h][j] * number
        
        self.logger.level -= 1
        
        return  y_hat, res_parts_random, VM_numbers, CL_res_random

    
    ## Method to increase the number of resources allocated to a partition
    #   @param self The object pointer
    #   @param comp_idx The index of the Graph.Component object
    #   @param part_idx The index of the Graph.Component.Partition object
    #   @param solution The current feasible solution
    #   @return True if the number of resources has been increased
    #
    # TODO: increase also the resources assigned to all the colocated partitions
    #
    def increase_number_of_resource(self, comp_idx, part_idx, solution):
        
        increased = False
        
        # get the index of the resources where the component partition is 
        # allocated
        resource_idx = np.nonzero(solution.Y_hat[comp_idx][part_idx,:])[0]
        
        # check if the number of resources assigned to the partition is 
        # lower than the number of available resources
        assigned_resources = solution.Y_hat[comp_idx][part_idx, resource_idx]
        if assigned_resources < self.system.resources[resource_idx].number:
            # if so, increase the number of assigned resources
            solution.Y_hat[comp_idx][part_idx, resource_idx] += 1
            increased = True

        return increased
    
    
    ## Method to return all the alternative resources of current allocated 
    # resource for the given component 
    #   @param self The object pointer
    #   @param comp_idx The index of the Graph.Component object
    #   @param part_idx The index of the Graph.Component.Partition object
    #   @param solution The current feasible solution
    #   @return The indices of the alternative resources
    def alternative_resources(self, comp_idx, part_idx, solution):
        
        # get the assignment matrix
        Y = solution.get_y()
        
        # get the list of compatible, unused resources
        l = np.greater(self.system.compatibility_matrix[comp_idx][part_idx,:], 
                       Y[comp_idx][part_idx,:])
        resource_idxs = np.where(l)[0]
        
        return resource_idxs
   
  
    ## Method reduce the number of Resources.VirtualMachine objects in a 
    # cluster 
    #   @param self The object pointer
    #   @param resource_idx The index of the Resources.VirtualMachine object
    #   @param result The current Solution.Result object
    #   @return The updated Solution.Result object
    def reduce_cluster_size(self, resource_idx, result):
        
        # initialize the new result
        new_result = copy.deepcopy(result)
        
        # check if the resource index corresponds to an edge/cloud resource
        if resource_idx < self.system.FaaS_start_index:
            
            # check if more than one resource of the given type is available
            if self.system.resources[resource_idx].number > 1:
                
                # get the max number of used resources
                y_bar = new_result.solution.get_y_bar()
                
                # update the current solution, always checking its feasibility
                feasible = True
                while feasible and y_bar[resource_idx].max() > 1:
                    
                    # create a copy of the current Y_hat matrix
                    temp = copy.deepcopy(new_result.solution.Y_hat)
                
                    # loop over all components
                    for i in range(len(new_result.solution.Y_hat)):
                        # loop over all component partitions
                        for h in range(len(new_result.solution.Y_hat[i])):
                            # decrease the number of resources (if > 1)
                            if temp[i][h,resource_idx] > 1:
                                temp[i][h,resource_idx] -= 1
                    
                    # create a new solution with the updated Y_hat
                    new_solution = Configuration(temp)
                    
                    # check if the new solution is feasible
                    new_performance = new_solution.check_feasibility(self.system)
                    
                    # if so, update the result
                    if new_performance[0]:
                        # update the current solution
                        new_result.solution = new_solution
                        new_result.performance = new_performance
                        y_bar = new_result.solution.get_y_bar()
        
        return new_result



## RandomGreedy
#
# Specialization of Algorithm that constructs the optimal solution through a 
# randomized greedy approach
class RandomGreedy(Algorithm):
    
    ## RandomGreedy class constructor
    #   @param self The object pointer
    #   @param system A System.System object
    #   @param log Object of Logger.Logger type
    def __init__(self, system, log = Logger()):
        super().__init__(system, log)
    
    
    ## Single step of the randomized greedy algorithm: it randomly generates 
    # a candidate solution, then evaluates its feasibility. If it is feasible, 
    # it evaluates its cost and updates it by reducing the cluster size
    #   @param self The object pointer
    #   @return Two tuples, storing the solution, its cost and its 
    #           performance results before and after the update, and a tuple 
    #           storing all the random parameters
    def step(self):
        
        # increase indentation level for logging
        self.logger.level += 1
        self.logger.log("Randomized Greedy step", 3)
        
        # initialize results
        result = Result()
        new_result = Result()
        
        # generate random solution and check its feasibility
        self.logger.level += 1
        self.logger.log("Generate random solution", 3)
        y_hat, res_parts_random, VM_numbers_random, CL_res_random = self.create_random_initial_solution()
        result.solution = Configuration(y_hat, self.logger)
        self.logger.log("Check feasibility", 3)
        feasible = result.check_feasibility(self.system)
        
        # if the solution is feasible, compute the corresponding cost 
        # before and after updating the clusters size
        if feasible:
            # compute cost
            self.logger.log("Compute cost", 3)
            result.objective_function(self.system)
            # update the cluster size of cloud resources
            self.logger.log("Update cluster size", 3)
            for j in range(self.system.FaaS_start_index):
                new_result = self.reduce_cluster_size(j, result)
            # compute the updated cost
            self.logger.log("Compute new cost", 3)
            new_result.objective_function(self.system)
            # update the list of VM numbers according to the new solution
            y_bar = new_result.solution.get_y_bar()
            for j in range(self.system.FaaS_start_index):
                if y_bar[j] > 0:
                    VM_numbers_random[j] = copy.deepcopy(min(y_bar[j], VM_numbers_random[j]))
        else:
            new_result = copy.deepcopy(result)
        self.logger.level -= 2
        
        return result, new_result, (res_parts_random, VM_numbers_random, CL_res_random)
            
    
    ## Method to generate a random gready solution 
    #   @param self The object pointer
    #   @param seed Seed for random number generation
    #   @param MaxIt Number of iterations, i.e., number of candidate 
    #                solutions to be generated (default: 1)
    #   @param K Number of elite results to be saved (default: 1)
    #   @return (1) Best Solution.Result before cluster update
    #           (2) Solution.EliteResults object storing the given number of 
    #           Solution.Result objects sorted by minimum cost
    #           (4) List of the random parameters
    def random_greedy(self, seed, MaxIt = 1, K = 1):
              
        # set seed for random number generation
        np.random.seed(seed)
        
        # initialize the elite set, the best result without cluster update 
        # and the lists of random parameters
        elite = EliteResults(K, Logger(self.logger.stream, 
                                       self.logger.verbose,
                                       self.logger.level+1))
        best_result_no_update = Result()
        res_parts_random_list = []
        VM_numbers_random_list = []
        CL_res_random_list = []

        # perform randomized greedy iterations
        self.logger.log("Starting Randomized Greedy procedure", 1)
        self.logger.level += 1
        for iteration in range(MaxIt):
            self.logger.log("#iter {}".format(iteration), 3)
            # perform a step
            result, new_result, random_param = self.step()
            # update the results and the lists of random parameters
            elite.add(new_result)
            if result < best_result_no_update:
                best_result_no_update = copy.deepcopy(result)
            res_parts_random_list.append(random_param[0])
            VM_numbers_random_list.append(random_param[1])
            CL_res_random_list.append(random_param[2])
        
        self.logger.level -= 1
        random_params = [res_parts_random_list, VM_numbers_random_list, 
                         CL_res_random_list]    
        
        return best_result_no_update, elite, random_params



## IteratedLocalSearch  
#
# Specialization of Algorithm      
class IteratedLocalSearch(Algorithm):
    
    pass


