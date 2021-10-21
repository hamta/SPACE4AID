from classes.Logger import Logger
from classes.Solution import Configuration, Result, EliteResults
import numpy as np
import copy
import sys
from heurispy.framework import genera_lista_ejecuciones_heuristicas, inicia_exploracion_heuristica
from heurispy.problema import Problema
from heurispy.heuristicas.busqueda_tabu import BusquedaTabu

from random import choice, randint, random
from string import ascii_lowercase
from Solid.Solid.TabuSearch import TabuSearch
from copy import deepcopy
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
    
    ## Method to get all partitions that can be run on FaaS
    #   @param self The object pointer
    #   @param Y_hat Assignment matrix includes all components and partitions assigned to resources 
    #   @return A list of partitions allocated to the FaaS including the components index, partition index and resource index
    def get_partitions_with_FaaS(self,Y_hat):
     
        partitions_with_FaaS=[]
        # loop over components
        for comp_idx, comp in enumerate(Y_hat):
         
           # get the partitions and resources allocated to them
          
           h_idxs,res_idxs=comp.nonzero()
           # get the indexes of FaaS allocated to the partitions
           res_FaaS_idx=res_idxs[res_idxs>=self.system.FaaS_start_index]
           # if the allocated resources are in FaaS, get the index of partitions and FaaS to add to the output list
           if len(res_FaaS_idx)>0:
               
               for i in range(len(res_FaaS_idx)):
                    h_FaaS_idx=comp[:,res_FaaS_idx[i]].nonzero()[0][0]
                  
                    partitions_with_FaaS.append((comp_idx,h_FaaS_idx,res_FaaS_idx[i])) 
                  
        return partitions_with_FaaS
    
    ## Method to get all partitions that can be run on resource j
    #   @param self The object pointer
    #   @param Y_hat Assignment matrix includes all components and partitions assigned to resources 
    #   @param j The index of resource
    #   @return A list of partitions allocated to resource j including the components index and partition index
    def get_partitions_with_j(self,Y_hat,j):
        partitions_with_j=[]
          # loop over components 
        for comp_idx, comp in enumerate(Y_hat):
          # get the partitions that are located in resource j
           comps_parts= np.nonzero(comp[:,j])[0]
           # if some partitions are located in j, add them to output list 
           if len(comps_parts>0):
              
               for comp_part in comps_parts:
                  
                   partitions_with_j.append((comp_idx,comp_part)) 
                  
        return partitions_with_j
    
    ## Method to change the input solution to find some neigbors of the current solution by changing the FaaS assignments
    #   @param self The object pointer
    #   @param solution Current solution 
    #   @return A list neigbors (new solutions) sorted by cost
    def change_FaaS(self,solution):
        new_sorted_results=None
        new_feasible_results=[]
        # call the method to get all partitions located in FaaS 
        partitions_with_FaaS=self.get_partitions_with_FaaS(solution.Y_hat)
        # loop over list of partitions in partitions_with_FaaS list
        for comp_part_j in partitions_with_FaaS:
            # get all the alternative resources of the partition
            res_idx=self.alternative_resources(comp_part_j[0],comp_part_j[1],solution)
            # Extract only the FaaS resources from alternative resources 
            Faas_res_idx = filter(lambda x: x >= self.system.FaaS_start_index, res_idx)
            # loop over alternative FaaS resources
            for j in Faas_res_idx:
                # get a copy of current solution as a new temprary assignment matrix (Y_hat)
                new_temp_Y_hat=copy.deepcopy(solution.Y_hat)
                # assign the current partition to the new alternative FaaS in new Y_hat
                new_temp_Y_hat[comp_part_j[0]][comp_part_j[1]][j]=1
                new_temp_Y_hat[comp_part_j[0]][comp_part_j[1]][comp_part_j[2]]=0
                # create a solution by new Y_hat
                new_temp_solution=Configuration(new_temp_Y_hat)
                # check the feasibility of new solution
                performance=new_temp_solution.check_feasibility(self.system)
                # if the new solution is feasible, add it to the neighbor result list
                if performance[0]:
                    result=Result()
                    result.solution = new_temp_solution
                    result.cost = result.objective_function(self.system)
                    result.performance = performance
                    new_feasible_results.append(result)
        if len(new_feasible_results)>0:
            # sort the list of result      
            new_sorted_results = sorted(new_feasible_results, key=lambda x: x.cost)    
            #new_sorted_solution=[x.solution for x in new_sorted_results]
        # return the list of neibors
        return new_sorted_results
    
    # ## Method to get the resources (nodes) with maximum utilization
    # #   @param self The object pointer
    # #   @param Y_hat Assignment matrix includes all components and partitions assigned to resources 
    # #   @return The index of resource with maximum utilization
    # def get_node_with_max_utilization(self, Y_hat):
        
    #     max_utilization=0
    #     idx_max_U_node=None
    #     # loop over the resources except FaaS
    #     for j in range(self.system.FaaS_start_index):
    #         # compute the utilization of current node
    #         utilization=self.system.resources[j].performance_evaluator.compute_utilization(j, Y_hat,self.system)
    #         # set the resource with maximum utilization
    #         if utilization>max_utilization:
    #             max_utilization=utilization
    #             idx_max_U_node=j
        
    #     return idx_max_U_node
    
    # ## Method to get the resources (nodes) with minimum utilization among alternative resources
    # #   @param self The object pointer
    # #   @param Y_hat Assignment matrix includes all components and partitions assigned to resources 
    # #   @param alternative_res_idxs The list of alternative resource indeces
    # #   @return The index of resource with minimum utilization and  cost
    # def get_node_with_min_U_among_compatible_nodes(self, Y_hat,alternative_res_idxs):
        
    #     #min_utilization=np.inf
    #     idx_min_U_node=[]
    #     # loop over all alternative resources
    #     for j in alternative_res_idxs:
    #         # check if the alternative resource is not FaaS
    #         if j <self.system.FaaS_start_index:
    #             # compute the utilization of current node
    #             utilization=self.system.resources[j].performance_evaluator.compute_utilization(j, Y_hat,self.system)
    #             # add the information of node to the list includes node index, utilization and cost
    #             idx_min_U_node.append((j, utilization, self.system.resources[j].cost))
                
    #     # sort the list based on utilization and cost        
    #     min_U_cost= sorted(idx_min_U_node, key=lambda element: (element[1], element[2]))
    #     # return the index of best alternative
    #     return min_U_cost[0][0]       
    
    
    
    ## Method to get active resources and computationallayers
    #   @param self The object pointer
    #   @param Y_hat Assignment matrix includes all components and partitions assigned to resources 
    #   @return (1) A list of resources that are in used in current Y_hat called active resources
    #           (2) A list includes the computational layers of active resources 
    def get_active_res_computationallayers(self, Y_hat):
        
        act_res_idxs=[]
        # loop over components
        for  comp in Y_hat:
          # get all nodes that are using by the partitions of current component
           h_idxs,res_idxs=comp.nonzero()
           # add the list res to the active resource list
           act_res_idxs.extend(res_idxs)
        # remove the duplicated resource indeces in active resource list
        active_res_idxs=list(set(act_res_idxs))
        
        # initialize active computational layer list
        act_camputationallayers=[]
        # loop over active resource list
        for act_res in active_res_idxs:
            # append the computational layer of current resource to the  active computational layer list
            act_camputationallayers.append(self.system.resources[act_res].CLname)
        # remove the duplicated computational layers in active computational layer list
        active_camputationallayers=list(set(act_camputationallayers))
        return active_res_idxs, active_camputationallayers
            
    ## Method to sort all nodes increasingly except FaaS by utilization and cost respectively
    #   @param self The object pointer
    #   @param Y_hat Assignment matrix includes all components and partitions assigned to resources 
    #   @return The sorted list of resources. 
    #       Each item of list includes the index, utilization and cost of the resource.
    #       The list is sorted by utilization, for the nodes with same utilization, it is sorted by cost
    def sort_nodes_by_U_cost(self, Y_hat):
  
        #min_utilization=np.inf
        idx_min_U_node=[]
        # loop over all alternative resources
        for j in range(self.system.FaaS_start_index):
            # compute the utilization of current node
            utilization=self.system.resources[j].performance_evaluator.compute_utilization(j, Y_hat,self.system)
            # add the information of node to the list includes node index, utilization and cost
            idx_min_U_node.append((j, utilization, self.system.resources[j].cost))
                
        # sort the list based on utilization and cost        
        sorted_node_list= sorted(idx_min_U_node, key=lambda element: (element[1], element[2]))
        # return the index of best alternative
        return sorted_node_list  
        
    ## Method to change the current solution by changing component placement
    #   @param self The object pointer
    #   @param solution Current solution 
    #   @return A list neigbors (new solutions) sorted by cost
    def change_component_placement(self, solution):
        
        neighbors=[]
        # get a sorted list of nodes' index with their utilization and cost (except FaaS)
        nodes_sorted_list= self.sort_nodes_by_U_cost(solution.Y_hat)
        # get resource with maximum utilization as source node
        #idx_max_U_node=self.get_node_with_max_utilization(solution.Y_hat)
        idx_max_U_node=nodes_sorted_list[-1][0]
        # get all partitions located in higest utilization node
        partitions=self.get_partitions_with_j(solution.Y_hat,idx_max_U_node)
        # get the list of nodes and computational layers in used 
        active_res_idxs, active_camputationallayers=self.get_active_res_computationallayers(solution.Y_hat)
        # loop over partitions
        for part in partitions:
            # get all alternative resources of the partitions
            alternative_res_idxs=self.alternative_resources(part[0],part[1],solution)
            # set a boolean variable to break, if best destination is founded
            find=False
            i=0
           
             # search to find the best campatible resource with lowest utilization for current partition
            while not find and i<len(nodes_sorted_list)-1:
                
                des_node_idx=nodes_sorted_list[i][0]
                # Check some conditions to avoid violating the limitation of our problem that says:
                # Only one node can be in used in each computational layer 
                # So, the destination node can be used only if it is one of running node, 
                # or its computational layer is not an active computational layer
                # or if the source and destination node are located in the same computational layer 
                # and the current partition is the only partition running in the computational layer 
                if des_node_idx in active_res_idxs or \
                        self.system.resources[des_node_idx].CLname not in active_camputationallayers or \
                        (len(partitions)==1 and self.system.resources[des_node_idx].CLname==self.system.resources[idx_max_U_node].CLname):
                    if des_node_idx in alternative_res_idxs:
                       
                        # get a copy of current solution as a new temprary assignment matrix (Y_hat)
                        new_temp_Y_hat=copy.deepcopy(solution.Y_hat)
                        # get all partitions running on the destination node
                        partitions_min_U=self.get_partitions_with_j(solution.Y_hat,des_node_idx)
                        # assign the current partition to the new alternative node in new Y_hat with maximume number of its instances
                        new_temp_Y_hat[part[0]][part[1]][idx_max_U_node]=0
                        new_temp_Y_hat[part[0]][part[1]][des_node_idx]=self.system.resources[des_node_idx].number
                       
                        if len(partitions_min_U)>0:
                            # assign the maximume instance number of destination node to the partitions that are running on destination node
                            for part_min in partitions_min_U:
                                new_temp_Y_hat[part_min[0]][part_min[1]][des_node_idx]=self.system.resources[des_node_idx].number
                        # creat a solution by new assignment (Y_hat)
                        new_temp_solution=Configuration(new_temp_Y_hat)
                        # check if new solution is feasible
                        performance=new_temp_solution.check_feasibility(self.system)
                        if performance[0]:
                            # creat new result
                            result=Result()
                            result.solution = new_temp_solution
                            # reduce cluster size of source and destination nodes
                            new_result_1 = self.reduce_cluster_size(des_node_idx, result)
                            new_result = self.reduce_cluster_size(idx_max_U_node, new_result_1)
                            # compute the cost
                            new_result.cost = new_result.objective_function(self.system)
                            new_result.performance = performance
                            # add new result in neigbor list
                            neighbors.append(new_result)
                            find=True
               
                i+=1
               
                    
            if not find:
                 print("There is no alternative node for partition "+str(part[1]) +" of component "+ str(part[0])+" in current solution." )
        # if some neighbors are founded, sort them by cost and return the list    
        if len(neighbors)>0:        
            new_sorted_results = sorted(neighbors, key=lambda x: x.cost)
            #new_sorted_solutions=[x.solution for x in new_sorted_results]
            
        else:
            print("No neighbor could be find by changing component placement of this solution ")
            new_sorted_results=None
        return new_sorted_results    
            
            
            
    ## Method to change the current solution by changing resource type
    #   @param self The object pointer
    #   @param solution Current solution 
    #   @return A list neigbors (new solutions) sorted by cost   
    def change_resource_type(self, solution):
       
        
        neighbors=[]
        new_sorted_results=None
        # get a sorted list of nodes' index with their utilization and cost (except FaaS)
        nodes_sorted_list= self.sort_nodes_by_U_cost(solution.Y_hat)
        # get resource with maximum utilization as source node
        #idx_max_U_node=self.get_node_with_max_utilization(solution.Y_hat)
        idx_max_U_node=nodes_sorted_list[-1][0]
        # get all partitions located in higest utilization node
        partitions=self.get_partitions_with_j(solution.Y_hat,idx_max_U_node)
        alternative_res_idxs_parts=[]
        # get a list of set of alternative nodes for partitions runing on source node
        for part in partitions:
            alternative_res_idxs_parts.append(set(self.alternative_resources(part[0],part[1],solution)))
        # get the intersection of the alternative nodes of all partitions runing on source node
        candidate_nodes=alternative_res_idxs_parts[0].intersection(*alternative_res_idxs_parts)
        
        if len(candidate_nodes)>0:
            # get the list of nodes and computational layers in used 
            active_res_idxs, active_camputationallayers=self.get_active_res_computationallayers(solution.Y_hat)
            # for each candidate nodes, move all partitions on it and create new solution
            for des in candidate_nodes:
                # Check some conditions to avoid violating the limitation of our problem that says:
                # Only one node can be in used in each computational layer 
                # So, the destination node can be used only if it is one of running node, 
                # or its computational layer is not an active computational layer
                # or if the source and destination node are located in the same computational layer 
                if des in active_res_idxs or \
                        self.system.resources[des].CLname not in active_camputationallayers or \
                        self.system.resources[des].CLname==self.system.resources[idx_max_U_node].CLname:
                            new_temp_Y_hat=copy.deepcopy(solution.Y_hat)
                            # get all partitions running on the destination node
                            partitions_on_candidate=self.get_partitions_with_j(solution.Y_hat,des)
                            # assign the maximume instance number of destination node to the partitions that are running on source node
                            for part in partitions:
                                new_temp_Y_hat[part[0]][part[1]][idx_max_U_node]=0
                                new_temp_Y_hat[part[0]][part[1]][des]=self.system.resources[des].number
                            if len(partitions_on_candidate)>0:
                                 # assign the maximume instance number of destination node to the partitions that are running on destination node
                                for part_cand in partitions_on_candidate:
                                    new_temp_Y_hat[part[0]][part[1]][des]=self.system.resources[des].number
                           # create new solution by new assignment
                            new_temp_solution=Configuration(new_temp_Y_hat)
                            # check feasibility
                            
                            performance=new_temp_solution.check_feasibility(self.system)
                            
                            if performance[0]:
                                # create a new result
                                result=Result()
                                result.solution = new_temp_solution
                                # reduce the cluster size of destination node
                                new_result = self.reduce_cluster_size(des, result)
                                new_result.cost = new_result.objective_function(self.system)
                                new_result.performance = performance
                                # add the new result to the neigbor list
                                neighbors.append(new_result)
            
            if len(neighbors)>0:
                # sort neighbor list by cost and return the best one
                new_sorted_results = sorted(neighbors, key=lambda x: x.cost)
                #new_sorted_solutions=[x.solution for x in new_sorted_results]
            else:
                print("No neighbor could be find by changing resource type because no feasible solution exists given the shared compatiblie nodes ")
                
        else:
            print("No neighbor could be find by changing resource type because no shared compatiblie node exists")
        return new_sorted_results 
    
     ## Method to union and sort the set of neighbors came from three methods: change_resource_type, change_component_placement, change_FaaS
    #   @param self The object pointer
    #   @param solution Current solution 
    #   @return A list neigbors (new solutions) sorted by cost   
    def union_neighbors(self, solution):
       
        neighborhood=[]
        # get the neighbors by changing FaaS configuration
        neighborhood1=self.change_FaaS(solution)
        # get the neighbors by changing resource type
        neighborhood2=self.change_resource_type(solution)
        # get the neigbors by change component placement
        neighborhood3=self.change_component_placement(solution)
        # mixe all neigbors
        if neighborhood1 is not None:
            neighborhood.extend(neighborhood1)
        if neighborhood2 is not None:
            neighborhood.extend(neighborhood2)
        if neighborhood3 is not None:
            neighborhood.extend(neighborhood3)
        # sort the neighbors list by cost 
        sorted_neighborhood=sorted(neighborhood, key=lambda x: x.cost)
        # if two solution have the same cost, check if the solutions are the same and drop one of them
        for neighbor_idx in range(len(sorted_neighborhood)-1):
            if sorted_neighborhood[neighbor_idx].cost==sorted_neighborhood[neighbor_idx+1].cost:
                if sorted_neighborhood[neighbor_idx].solution==sorted_neighborhood[neighbor_idx+1].solution:
                    
                    sorted_neighborhood.remove(sorted_neighborhood[neighbor_idx])

        return sorted_neighborhood
            
    ## Method to create the initial solution with largest configuration function (for only FaaS scenario)
    #   @param self The object pointer
    #   @return solution  
    def creat_initial_solution_with_largest_conf_fun(self):
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
                candidate_nodes.append(random_num)
            resource_count += len(l.resources)
        for comp in self.system.components:
            
            # randomly select a deployment for that component
            random_dep = np.random.choice(comp.deployments)
          
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
                max_cost=0
                for j_idx in index:
                    if j_idx in range(self.system.cloud_start_index):
                        cost=self.system.resources[j_idx].cost
                    elif j_idx in range(self.system.cloud_start_index, self.system.FaaS_start_index):
                        cost=self.system.resources[j_idx].cost 
                    #
                    # compute the cost of FaaS and transition cost if not using SCAR
                    elif j_idx in range(self.system.FaaS_start_index, J):
                       
                        cost=self.system.resources[j_idx].cost * self.system.components[i].comp_Lambda * self.system.T 
                    
                    if cost>max_cost:
                        max_cost=copy.deepcopy(cost)
                        j_largest=j_idx
                        
                y[i][h_idx,j_largest] = 1
                y_hat[i][h_idx,j_largest] = 1
                # if the partition is the last partition (i.e., its successor 
                # is the successor of the component), update the size of 
                # data transferred between the components
                if self.system.graph.G.succ[comp.name] != {}:
                    if part.Next == list(self.system.graph.G.succ[comp.name].keys())[0]:
                        self.system.graph.G[comp.name][part.Next]["data_size"] = part.data_size
        solution = Configuration(y_hat, self.logger)
      
        feasible = solution.check_feasibility(self.system)
        
       
        if feasible:
            new_solution=solution
              
        else:
            new_solution=None
        return new_solution
        
        
            
                    
                
                

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



## TabuSearch  
#
# Specialization of Algorithm      
class TabuSearchHeurispy(Algorithm):
    
    ## TabuSearchHeurispy class constructor
    #   @param self The object pointer
    #   @param system A System.System object
    #   @param seed A seed to generate random values
    #   @param Max_It_RG Maximum iterations of random greedy
    #   @param log Object of Logger.Logger type
    def __init__(self, system,seed,Max_It_RG, log = Logger()):
        super().__init__(system, log)
        self.seed=seed
        self.Max_It_RG=Max_It_RG
    
    
    ## Method to create initial solution for tabue search
    #   @param self The object pointer
    #   @return A solution
    def creat_initial_solution(self):
        # create a RandomGreedy object and run random gready method
        GA=RandomGreedy(self.system)
        best_result_no_update, elite, random_params=GA.random_greedy(self.seed,MaxIt = self.Max_It_RG)
      
        #self.initial_solution=elite.elite_results[0].solution
       
        return elite.elite_results[0].solution
        #return self.creat_initial_solution_with_largest_conf_fun()
        
    ## Method to compute the objective function of a solution
    #   @param self The object pointer
    #   @param new_solution The solution to get the objective function
    #   @return The value of objective function
    def objective_function(self, new_solution):
        
        return new_solution.objective_function(self.system)
        
    ## Method to pick one of neighbors 
    #   @param self The object pointer
    #   @param solution The input solution 
    #   @param method The method of choosing a neigbor 
    #   @return One of neighbors
    def get_one_neighbor(self, solution, method="best"):
        neighbor=None
        # compute the neigbors of the solution
        # sorted_solution_list=self.change_FaaS(solution)
        sorted_solution_list=self.change_resource_type(solution)
        if len(sorted_solution_list)>0:
            # if the method is best, pick the first item of sorted list
            if method=="best":
                # pick the best neighbor
                neighbor=sorted_solution_list[0]
            else:
                # pick a neighbor randomly
                idx=np.random.randint(0,len(sorted_solution_list))
                neighbor=sorted_solution_list[idx]
        else:
             print("There is not any neighbors")
             
        return neighbor
    
    ## Method to run tabu search
    #   @param self The object pointer
    #   @param max_iterations The maximum iterations
    #   @param memory_space A list of memory size for tabu 
    #           tabu search will run for each number in the list separately
    #   @param max_search_without_improvement A list of maximum search without improvement
    #           tabu search will run for each number in the list separately
    #   @param repetitions The number of repetitions for tabu search
    #   @return (1): the best result found by tabu search
    #           (2): the list of current solution cost in each iteration
    #           (3): the list of best solution cost found by tabu search until now in each iteration
    def main_tabu_search(self,max_iterations,memory_space,max_search_without_improvement,repetitions):
       
        # create an object of problem class
        coloration_problem = Problema(dominio=self.creat_initial_solution,
                                    funcion_objetivo=self.objective_function,
                                    funcion_variacion_soluciones=self.get_one_neighbor)
        # create an object of tabu search class
        tabu_search = BusquedaTabu(coloration_problem, max_iteraciones=max_iterations)
        # create a dictionary of hyper parameters needed for tabu search
        tabu_search_parameters = dict(espacio_memoria=memory_space, max_busquedas_sin_mejora=max_search_without_improvement)
        # create a list of executions that tabu search will run for each element of the list
        executions_list = genera_lista_ejecuciones_heuristicas(tabu_search_parameters, repeticiones=repetitions)
        # run tabu search and get the results
        best_result, current_cost_list, best_cost_list=inicia_exploracion_heuristica(tabu_search, executions_list, nucleos_cpu=4)
        
        return best_result, current_cost_list, best_cost_list
    
    
## IteratedLocalSearch  
#
# Specialization of Algorithm      
class IteratedLocalSearch(Algorithm):
    
    pass



class TabuSearchSolid(TabuSearch,Algorithm):
    """
    Tries to get a randomly-generated string to match string "clout"
    """
    ## TabuSearchSolid class constructor
    #   @param self The object pointer
    #   @param seed A seed to generate random values
    #   @param Max_It_RG Maximum iterations of random greedy
    #   @param max_steps Maximum steps of tabu search
    #   @param min_score Minimum cost
    #   @param system A System.System object
    #   @param log Object of Logger.Logger type
    def __init__(self,seed,Max_It_RG, tabu_size, max_steps, min_score, system, log = Logger()):
        self.seed=seed
        self.Max_It_RG=Max_It_RG
        Algorithm.__init__(self,system, log)
        # compute initial solution
        initial_state=self.creat_initial_solution()
        TabuSearch.__init__(self,initial_state, tabu_size, max_steps, min_score)
       
    ## Method to get a list of neigbors
    #   @param self The object pointer
    #   @return A list of solutions (neighbors)    
    def _neighborhood(self ):
        #neighborhood=self.change_FaaS(self.current)
        #neighborhood=self.change_resource_type(self.current)
        #neighborhood=self.change_component_placement(self.current)
        #neighborhood=[x.Y_hat for x in neigbors]
        neighborhood=self.union_neighbors(self.current)
        return [x.solution for x in neighborhood]
    
    ## Method to get the cost of current solution
    #   @param self The object pointer
    #   @param solution The current solution
    #   @return The cost of current solution
    def _score(self, solution):
       
        return solution.objective_function(self.system)*(-1)
    
    
    ## Method to create initial solution for tabue search
    #   @param self The object pointer
    #   @return A solution
    def creat_initial_solution(self):
        # create a RandomGreedy object and run random gready method
        GA=RandomGreedy(self.system)
   
        best_result_no_update, elite, random_params=GA.random_greedy(self.seed,MaxIt = self.Max_It_RG)
        initial_solution=elite.elite_results[0].solution
       
        #initial_solution=self.creat_initial_solution_with_largest_conf_fun()
        return initial_solution
    
    # ## Method to get the cost of solution
    # #   @param self The object pointer
    # #   @param state The current solution
    # #   @return The cost of current solution
    # def objective_function(self, solution):
        
    #     return solution.objective_function(self.system)*(-1)
    
    