from classes.Logger import Logger
from classes.Solution import Configuration
import numpy as np
import copy
import sys
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.fmin import generate_trials_to_calculate
import time
from datetime import datetime
import random
import pdb

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
    #   @return List of 2D numpy matrices denoting the amount of  
    #           Resources.Resource assigned to each 
    #           Graph.Component.Partition object
    def create_random_initial_solution(self):
        
        # increase indentation level for logging
        self.logger.level += 1
        
        # initialize the assignments
        self.logger.log("Initialize matrices", 4)
        CL_res_random=[]
        res_parts_random=[]

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
            h=0
            rand=[]
            # loop over all partitions in the deployment
            for part in random_dep.partitions:
                # get the indices of the component and the deployment
                
                i = self.system.dic_map_part_idx[comp.name][comp.partitions[part].name][0]
               
                h_idx = self.system.dic_map_part_idx[comp.name][comp.partitions[part].name][1]
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
                    if comp.partitions[part].Next == list(self.system.graph.G.succ[comp.name].keys())[0]:
                        self.system.graph.G[comp.name][comp.partitions[part].Next]["data_size"] = comp.partitions[part].data_size
            res_parts_random.append(rand)
        # loop over edge/cloud resources
        self.logger.log("Set number of resources", 4)
        VM_numbers=[] 
        # check if the system dosent have FaaS  
        if self.system.FaaS_start_index!=float("inf"):
            # get the last index of cloud
            edge_VM=self.system.FaaS_start_index
        else:
            edge_VM=J
        for j in range(edge_VM):
            # randomly generate the number of resources that can be assigned 
            # to the partitions that run on that resource
            number = np.random.randint(0, self.system.resources[j].number )
            VM_numbers.append(number)
            # loop over components
            for i in range(I):
                # get the number of partitions
                H = self.system.compatibility_matrix[i].shape[0]
                # loop over the partitions
                for h in range(H):
                    # if the partition runs on the current resource, update 
                    # the number
                    if y[i][h][j] > 0:
                        y_hat[i][h][j] = y[i][h][j] * (number+1)
        
        self.logger.level -= 1
        
        return  y_hat, res_parts_random, VM_numbers, CL_res_random

    
    ## Method to increase the number of resources allocated to a partition
    #   @param self The object pointer
    #   @param comp_idx The index of the Graph.Component object
    #   @param part_idx The index of the Graph.Component.Partition object
    #   @param solution The current feasible solution
    #   @return True if the number of resources has been increased
    #
    # TODO: this does not sound very reasonable: what if all resources have 
    # already been assigned? Is this number of available resources (which 
    # was originally referred to as S.resource_number) ever updated?
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
    #
    # TODO: this does not sound reasonable, given that Y_hat stores a number
    # of resources while each element of the compatibility matrix is at most
    # equal to 1...I would have used Y instead
    #
    def alternative_resources(self, comp_idx, part_idx, solution): 
        l = np.greater(self.system.compatibility_matrix[comp_idx][part_idx,:], 
                       solution.Y_hat[comp_idx][part_idx,:])
        resource_idxs = np.where(l)[0]
        return resource_idxs
   
  
    ## Method reduce the number of Resources.VirtualMachine objects in a 
    # cluster 
    #   @param self The object pointer
    #   @param resource_idx The index of the Resources.VirtualMachine object
    #   @param solution The current feasible solution 
    #   @return The updated solution
    def reduce_cluster_size(self, resource_idx, solution):
        
        # check if the resource index corresponds to an edge/cloud resource
        if resource_idx < self.system.FaaS_start_index:
            
            # check if more than one resource of the given type is available
            if self.system.resources[resource_idx].number > 1:
                
                # get the max number of used resources
                y_bar = solution.get_y_bar()
                
                # update the current solution, always checking its feasibility
                feasible = True
                # TODO: why max?
                while feasible and y_bar[resource_idx].max() > 1:
                    
                    # create a copy of the current Y_hat matrix
                    temp = copy.deepcopy(solution.Y_hat)
                
                    # loop over all components
                    for i in range(len(solution.Y_hat)):
                        # loop over all component partitions
                        for h in range(len(solution.Y_hat[i])):
                            # decrease the number of resources (if > 1)
                            if temp[i][h,resource_idx] > 1:
                                temp[i][h,resource_idx] -= 1
                    
                    # create a new solution with the updated Y_hat
                    new_solution = Configuration(temp)
                    
                    # check if the new solution is feasible
                    feasible, paths_performance, components_performance = new_solution.check_feasibility(self.system)
                    if feasible:
                        # update the current solution
                        solution = copy.deepcopy(new_solution)
                        y_bar = solution.get_y_bar()
        
        return solution



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
    # a candidate solution, then evaluate its feasibility. If it is feasible, 
    # it evaluates its cost and updates it by reducing the cluster size
    #   @param self The object pointer
    #   @return Two tuples, storing the solution, its cost and its 
    #           performance results before and after the update
    def step(self):
        
        # increase indentation level for logging
        self.logger.level += 1
        self.logger.log("Randomized Greedy step", 3)
        
        # initialize solution, cost and performance results
        solution = None
        cost = np.infty
        performance = (False, None, None)
        new_solution = None
        new_cost = np.infty
        new_performance = (False, None, None)
        
        # generate random solution and check its feasibility
        self.logger.level += 1
        self.logger.log("Generate random solution", 3)
        y_hat, res_parts_random, VM_numbers_random, CL_res_random=self.create_random_initial_solution()
        solution = Configuration(y_hat, self.logger)
        self.logger.log("Check feasibility", 3)
        performance = solution.check_feasibility(self.system)
        
        # if the solution is feasible, compute the corresponding cost 
        # before and after updating the clusters size
        if performance[0]:
            # compute cost
            self.logger.log("Compute cost", 3)
            cost = solution.objective_function(self.system)
            # update the cluster size of cloud resources
            self.logger.log("Update cluster size", 3)
            new_solution = copy.deepcopy(solution)
            J = len(self.system.resources)
            #for j in range(self.system.cloud_start_index, J):
            if self.system.FaaS_start_index!=float("inf"):
                edge_VM=self.system.FaaS_start_index
            else:
                edge_VM=J
            for j in range(edge_VM):
                new_solution = self.reduce_cluster_size(j, new_solution)
            # evaluate the feasibility of the updated solution
            self.logger.log("Check feasibility", 3)
            new_performance = new_solution.check_feasibility(self.system) # we dont need check the feasibility againe 
            # if the new solution is feasible, compute the updated cost
            if new_performance[0]:
                self.logger.log("Compute new cost", 3)
                new_cost = new_solution.objective_function(self.system)
            
            y_bar = new_solution.get_y_bar()
            for j in range(edge_VM):
                if y_bar[j]>0:
                    VM_numbers_random[j]=copy.deepcopy(min(y_bar[j],VM_numbers_random[j]))
        else:
            new_solution=copy.deepcopy(solution)
            new_cost=copy.deepcopy(cost)
        self.logger.level -= 2
        
        return (solution, cost, performance), (new_solution, new_cost, new_performance), (res_parts_random, VM_numbers_random, CL_res_random)
            
    
    ## Method to generate a random gready solution 
    #   @param self The object pointer
    #   @param seed Seed for random number generation
    #   @param MaxIt Number of iterations, i.e., number of candidate 
    #                solutions to be generated (default: 1)
    def random_greedy(self, seed, MaxIt = 1):
        
        # set seed for random number generation
        random.seed(datetime.now())
        
        best_result = [None, np.infty, (False, None, None)]
        new_best_result = [None, np.infty, (False, None, None)]
        solutions=[]
        self.logger.log("Starting Randomized Greedy procedure", 1)
        self.logger.level += 1
        res_parts_random_list=[]
        VM_numbers_random_list=[]
        CL_res_random_list=[]
        for iteration in range(MaxIt):
            self.logger.log("#iter {}".format(iteration), 2)
            self.logger.level += 1
            # perform a step
            result, new_result, random_param = self.step()
            solutions.append(new_result[0])
            res_parts_random_list.append(random_param[0])
            VM_numbers_random_list.append(random_param[1])
            CL_res_random_list.append(random_param[2])
            # check if the new solution improves the current minimum cost  
            if result[1] < best_result[1]:
                best_result[0] = result[0]
                best_result[1] = result[1]
                best_result[2] = result[2]
                self.logger.log("Result improved: {} --> {}".\
                                format(best_result[1], result[1]), 2)
            # check if the new updated solution improves the current updated 
            # minimum cost  
            if new_result[1] < new_best_result[1]:
                new_best_result[0] = new_result[0]
                new_best_result[1] = new_result[1]
                new_best_result[2] = new_result[2]
                self.logger.log("New result improved: {} --> {}".\
                                format(new_best_result[1], new_result[1]), 2)
            self.logger.level -= 1
        
        self.logger.level -= 1
        random_params=[res_parts_random_list,VM_numbers_random_list, CL_res_random_list]    
        return solutions, best_result, new_best_result,random_params


## HyperOpt
#
# Specialization of Algorithm that use HyperOpt library of python to find the optimal solution
class HyperOpt():    
   
    
    ## HyperOpt class constructor
    #   @param self The object pointer
    #   @param system A System.System object
   def __init__(self, system):
        self.system=system
        
   ## The objective function of HyperOpt  
    #   @param self The object pointer
    #   @param args All arguments with their search space     
   def objective(self,args):
       
       # get the search space of all parameters
        resource_random_list, deployment_random_list, VM_number_random_list, prob_res_selection_dep_list= args
        
        # start to create Y_hat from the search space of parameters
        I=len(self.system.components)
        J=len(self.system.resources)
        y_hat=[]
        y=[]
        for i in range(I):
            H,J=self.system.compatibility_matrix[i].shape
            # initialize Y_hat, y
            y_hat.append(np.full((H, J), 0, dtype=int))
            y.append(np.full((H, J), 0, dtype=int))
        
        candidate_nodes=[]
        resource_count = 0
        # loop over all computational layers
        
        for idx,l in enumerate( self.system.CLs):
            # select all nodes in FaaS layers
            if resource_count >= self.system.FaaS_start_index:
                random_num = l.resources
                candidate_nodes.extend(random_num)
            # set a node in other layers based on what HypetOpt selected
            else:
                candidate_nodes.append(l.resources[resource_random_list[idx]])
            resource_count += len(l.resources)
        
        for comp_idx, comp in enumerate(self.system.components):
            # set a deployment for each component based on what HypetOpt selected
            random_dep=comp.deployments[deployment_random_list[comp_idx]]
            
            h=0
            
            for part_idx, part in enumerate(random_dep.partitions):
                
                 # get the indices of the component and the deployment selected by HypetOpt
                i=self.system.dic_map_part_idx[comp.name][comp.partitions[part].name][0]
                # get the indices of compatible resources and compute the 
                # intersection with the selected resources in each 
                # computational layer

                 
                h_idx=self.system.dic_map_part_idx[comp.name][comp.partitions[part].name][1]
                idx=np.nonzero(self.system.compatibility_matrix[i][h_idx,:])[0]
                # extract a resource index in the intersection
                index=list(set(candidate_nodes).intersection(idx))
                prob=1/len(index)
                step=0
                rn=prob_res_selection_dep_list[comp_idx][part_idx]
                
                for r in np.arange(0,1,prob):
                    if rn>r and rn<=r+prob:
                        j= index[step]
                       
                    else:
                        step+=1
                y[i][h_idx][j]=1
                y_hat[i][h_idx][j]=1
                # if the partition is the last partition (i.e., its successor 
                # is the successor of the component), update the size of 
                # data transferred between the components
                if  self.system.graph.G.succ[comp.name]!={}:
                    if comp.partitions[part].Next==list(self.system.graph.G.succ[comp.name].keys())[0]:
                    
                        self.system.graph.G[comp.name][comp.partitions[part].Next]["data_size"]=comp.partitions[part].data_size
               
       
        # check if the system dosent have FaaS 
        if self.system.FaaS_start_index!=float("inf"):
            edge_VM=self.system.FaaS_start_index
        else:
            edge_VM=J
        # randomly generate the number of resources that can be assigned 
        # to the partitions that run on that resource
        for j in range(edge_VM):
             
            # loop over components  
            for i in range(I):
                 H=self.system.compatibility_matrix[i].shape[0]
                 for h in range(H):
                    if y[i][h][j]>0:
                        y_hat[i][h][j] = y[i][h][j]*(VM_number_random_list[j]+1)
        
        solution=Configuration(y_hat)
       
        flag, primary_paths_performance, primary_components_performance =solution.check_feasibility(self.system)
       
        
        if flag:
           
           
            costs=solution.objective_function(self.system)
            return {'loss': costs,
                'time': time.time(),
                'status': STATUS_OK }
        else:
           
            return {'status': STATUS_FAIL,
                    'time': time.time(),
                    'exception': "inf"}
       
             

    ## Random optimization function by HyperOpt  
    #   @param self The object pointer
    #   @param iteration_number The iteration number for HyperOpt
    #   @param vals_list The list of value to feed the result of RandomGreedy to HyperOpt  
   def random_hyperopt(self, iteration_number,vals_list=[]):  
      
        # Create search spaces for all random variable by defining a dictionary for each of them
        resource_random_list=[]
        for idx, l in enumerate(self.system.CLs):
            # Create search spaces for all computational layer except the last one with FaaS
            if l != list(self.system.CLs)[-1]:
              
                resource_random_list.append( hp.randint("res"+str(idx),len(l.resources)))
                
       
        deployment_random_list=[]
        prob_res_selection_dep_list=[]
         # Create search spaces for all random variable which is needed for components
        for idx, comp in enumerate(self.system.components):
            max_part=0
            res_random=[]
             # Create search spaces for deployments
            deployment_random_list.append(hp.randint("dep"+str(idx),len(comp.deployments)))
            #random_dep=comp.deployments[random_num]
            for  dep in comp.deployments:
                if max_part<len(list(dep.partitions)):
                    max_part=len(list(dep.partitions))
             # Create search spaces for resources of partitions
            for i in range( max_part):
                res_random.append( hp.uniform("com_res"+str(idx)+str(i),0,1))
            prob_res_selection_dep_list.append(res_random)    
        VM_number_random_list=[]    
         # Create search spaces for determining VM number 
        for j in range(self.system.FaaS_start_index):
             
            VM_number_random_list.append( hp.randint("VM"+str(j),self.system.resources[j].number))
       
        # creat a list of search space including all variables 
        space1 = resource_random_list
        space2 = deployment_random_list 
        space3 = VM_number_random_list
        space4 = prob_res_selection_dep_list
        space=[space1,space2,space3,space4]     
        # create trails to search in search spaces
        trials = Trials()
        best_cost=0
        # if we need to use Spark for parallelisation
        #trials = SparkTrials(parallelism=4)
        
        # if there is some result from RandomGreedy to feed to HyperOpt
        if len(vals_list)>0:
            trials=generate_trials_to_calculate(vals_list)
            
        # run fmin method to search and find the best solution
        try:
            best = fmin(fn=self.objective, space=space, algo=tpe.suggest, trials=trials,  max_evals=iteration_number)
        except:
            
            best_cost= float("inf")
       
        # check if HyperOpt could find solution
        if best_cost== float("inf"):
            # if HyperOpt cannot find any feasible solution
            return best_cost,None
        else:
           
            best_cost=trials.best_trial["result"]["loss"]
            resource_random_list=[]
            for idx, l in enumerate(self.system.CLs):
                if l != list(self.system.CLs)[-1]:
                  
                    resource_random_list.append( best["res"+str(idx)])
                    #candidate_nodes.append(l.resources[random_num])
            
           
            deployment_random_list=[]
            prob_res_selection_dep_list=[]
           
            for idx, comp in enumerate(self.system.components):
                max_part=0
                res_random=[]
                deployment_random_list.append(best["dep"+str(idx)])
                #random_dep=comp.deployments[random_num]
                for  dep in comp.deployments:
                    if max_part<len(dep.partitions):
                        max_part=len(dep.partitions)
                for i in range( max_part):
                    res_random.append( best["com_res"+str(idx)+str(i)])
                prob_res_selection_dep_list.append(res_random)    
            VM_number_random_list=[]    
            for j in range(self.system.FaaS_start_index):
                 
                VM_number_random_list.append(best["VM"+str(j)])
            I=len(self.system.components)
            J=len(self.system.resources)
            y_hat=[]
            y=[]
            for i in range(I):
                H,J=self.system.compatibility_matrix[i].shape
                y_hat.append(np.full((H, J), 0, dtype=int))
                y.append(np.full((H, J), 0, dtype=int))
            candidate_nodes=[]
            #pdb.set_trace()
            
            for idx, l in enumerate(self.system.CLs):
                if l == list(self.system.CLs)[-1]:
                    random_num=l.resources
                    candidate_nodes.extend(random_num)
                else:
                    #pdb.set_trace()  
                    #resource_random_list.append( hp.randint("res"+str(idx),len(l.resources)))
                    candidate_nodes.append(l.resources[resource_random_list[idx]])
            
           
           
            for comp_idx, comp in enumerate(self.system.components):
               # deployment_random_list.append(hp.randint("dep"+str(idx),len(comp.deployments)))
                random_dep=comp.deployments[deployment_random_list[comp_idx]]
                
                h=0
                
                for part_idx, part in enumerate(random_dep.partitions):
                    
                    
                    i=self.system.dic_map_part_idx[comp.name][comp.partitions[part].name][0]
                    h_idx=self.system.dic_map_part_idx[comp.name][comp.partitions[part].name][1]
                    idx=np.nonzero(self.system.compatibility_matrix[i][h_idx,:])[0]
                   
                    index=list(set(candidate_nodes).intersection(idx))
                    prob=1/len(index)
                    step=0
                    rn=prob_res_selection_dep_list[comp_idx][part_idx]
                    
                    for r in np.arange(0,1,prob):
                        if rn>r and rn<=r+prob:
                            j= index[step]
                           
                        else:
                            step+=1
                    y[i][h_idx][j]=1
                    y_hat[i][h_idx][j]=1
                    
                    if  self.system.graph.G.succ[comp.name]!={}:
                        if comp.partitions[part].Next==list(self.system.graph.G.succ[comp.name].keys())[0]:
                        
                            self.system.graph.G[comp.name][comp.partitions[part].Next]["data_size"]=comp.partitions[part].data_size
                   
            if self.system.FaaS_start_index!=float("inf"):
                edge_VM=self.system.FaaS_start_index
            else:
                edge_VM=J
            for j in range(edge_VM):
            
                 
                #VM_number_random_list.append( hp.randint("VM"+str(idx),self.system.resources[j].number))
                  
                for i in range(I):
                     H=self.system.compatibility_matrix[i].shape[0]
                     for h in range(H):
                        if y[i][h][j]>0:
                            y_hat[i][h][j] = y[i][h][j]*(VM_number_random_list[j]+1)
         
            #A=Algorithm(self.system,y_hat)
            solution=Configuration(y_hat)
            # for j in range(J):
            #     solution= A.reduce_cluster_size(j, solution, self.system)
            costs=solution.objective_function(self.system)
           
            return costs, solution


   def creat_trials_by_RandomGreedy(self, solutions, res_parts_random_list, VM_numbers_random_list, CL_res_random_list):
    
   
        vals_list=[]
        for solution_idx, solution in enumerate(solutions):
           
           flag, primary_paths_performance, primary_components_performance =solution.check_feasibility(self.system)
           if flag:
               
               costs=solution.objective_function(self.system)
           vals={}
           
           com_res={}
           dep={}
           res={}
           VM={}
           for idx, comp in enumerate(self.system.components):
                max_part=0
               
                for  dep in comp.deployments:
                    if max_part<len(list(dep.partitions)):
                        max_part=len(list(dep.partitions))
                for i in range( max_part):
                    vals["com_res"+str(idx)+str(i)]=random.random()
                    com_res["com_res"+str(idx)+str(i)]=random.random()
                    
           for comp_idx, y in enumerate(solution.Y_hat):
               
               
               H,J=y.shape
               for h in range(H):
                  
                   resource_idx=np.nonzero(y[h,:])[0] 
                   if len(resource_idx)>0:
                      
                        
                       for dep_idx, dep in enumerate(self.system.components[comp_idx].deployments):
                           if h in dep.partitions:
                               
                               vals["dep"+str(comp_idx)]=dep_idx
                             
                               vals["com_res"+str(comp_idx)+str(dep.partitions.index(h))]=res_parts_random_list[solution_idx][comp_idx][dep.partitions.index(h)]
                              
           for idx, l in enumerate(CL_res_random_list[solution_idx]):
                           
                        vals["res"+str(idx)]=l
                        
           if self.system.FaaS_start_index!=float("inf"):
                edge_VM=self.system.FaaS_start_index
           else:
                edge_VM=J
           for j in range(edge_VM):
                
                max_number=0
                for i in range(len(solution.Y_hat)):  
                    if max(solution.Y_hat[i][:,j])>max_number:
                          max_number=max(solution.Y_hat[i][:,j])
                       
                if max_number>0: 
                    vals["VM"+str(j)]=max_number-1
                else:
                    vals["VM"+str(j)]=  VM_numbers_random_list[solution_idx][j] 
               
           vals_list.append(vals)
        
        
       
        return  vals_list
## IteratedLocalSearch  
#
# Specialization of Algorithm      
class IteratedLocalSearch(Algorithm):
    
    pass


