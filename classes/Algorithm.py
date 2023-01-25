from classes.Logger import Logger
from classes.Solution import Configuration, Result, EliteResults
from classes.PerformanceEvaluators import SystemPerformanceEvaluator, ServerFarmPE, EdgePE
import numpy as np
import copy
import sys
from Solid.Solid.TabuSearch import TabuSearch
from Solid.Solid.SimulatedAnnealing import SimulatedAnnealing
from Solid.Solid.GeneticAlgorithm import GeneticAlgorithm
import math
import time
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, rand
from hyperopt.fmin import generate_trials_to_calculate
import random

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
    def __init__(self, system,seed=1, log = Logger()):
        self.logger = log
        self.error = Logger(stream=sys.stderr, verbose=1)
        self.system = system
        self.seed = seed
        np.random.seed(seed)

     ## Method to create a solution from a solution file in output file format
    #   @param self The object pointer
    #   @param solution_file A solution file
    #   @return result 
    def create_solution_by_file(self, solution_file):
        # read solution json file
         data=self.system.read_solution_file(solution_file)
         if "components" in data.keys():
            # load components info
            C = data["components"]
            Y_hat=[]
         else:
            self.error.log("ERROR: no components available in solution file", 1)
            sys.exit(1)
         # loop over all components
         I = len(self.system.components)
         for i in range(I):
            # get the number of partitions and available resources
            H, J = self.system.compatibility_matrix[i].shape
            # create the empty matrices
            Y_hat.append(np.full((H, J), 0, dtype=int))
        # loop over components
         for c in C:
            if c in self.system.dic_map_com_idx.keys():
                comp_idx=self.system.dic_map_com_idx[c]
            else:
                 self.error.log("ERROR: the commponent does not exist in the system", 1)
                 sys.exit(1)
            dep_included=False
            # loop over deployments
            for s in C[c]:
                if s.startswith("s"):
                    dep_included = True
                    part_included = False
                    # loop over partitions
                    for h in C[c][s]:
                        if h.startswith("h"):
                            part_included=True
                            part_idx = self.system.dic_map_part_idx[c][h][1]

                            CL = list(C[c][s][h].keys())[0]
                            res = list(C[c][s][h][CL].keys())[0]
                            res_idx = self.system.dic_map_res_idx[res]
                            if res_idx < self.system.FaaS_start_index:
                                number = C[c][s][h][CL][res]["number"]
                            else:
                                number = 1
                            Y_hat[comp_idx][part_idx][res_idx]=number
                    if not part_included:
                        self.error.log("ERROR: there is no selected partition for component " + c + " and deployment " + s + " in the solution file", 1)
                        sys.exit(1)
            if not dep_included:
                self.error.log("ERROR: there is no selected deployment for component " + c + " in the solution file", 1)
                sys.exit(1)
         result = Result()
         result.solution = Configuration(Y_hat, self.logger)
         result.check_feasibility(self.system)
         result.objective_function(self.system)
         return result

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
                    #if part.Next == list(self.system.graph.G.succ[comp.name].keys())[0]:
                    if part.Next == list(self.system.graph.G.succ[comp.name].keys()):
                        for next_idx in range(len(part.Next)):
                            self.system.graph.G[comp.name][part.Next[next_idx]]["data_size"] = part.data_size[next_idx]

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
    def create_random_initial_solution_smart(self):

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
        source_nodes= [node[0] for node in self.system.graph.G.in_degree if node[1]==0]
        visited={node:False for node in self.system.graph.G.nodes}
        Queue=copy.deepcopy(source_nodes)
        possible_res=copy.deepcopy(self.system.compatibility_matrix)
        #for node in Queue:
         #   visited[node]=True
        while Queue:
            pred_is_visited=True
            current_node=Queue.pop(0)
            comp_idx=self.system.dic_map_com_idx[current_node]
            comp=self.system.components[comp_idx]
            random_dep = np.random.choice(comp.deployments)
            comp_pred_ist=list(self.system.graph.G.pred[current_node])
            if len(comp_pred_ist)>0:
                for comp_pred in comp_pred_ist:
                    comp_pred_idx=self.system.dic_map_com_idx[comp_pred]
                    if len(np.nonzero(y_hat[comp_pred_idx])[0])>0:
                        last_h_idx=np.nonzero(y_hat[comp_pred_idx])[0][-1]
                        last_h_res = np.nonzero(y_hat[comp_pred_idx][last_h_idx,:])[0][0]
                        if last_h_res >= self.system.cloud_start_index:
                            possible_res[comp_idx][:, :self.system.cloud_start_index]=0
                    else:
                        pred_is_visited=False
                        if comp_pred in Queue:
                            Queue.remove(comp_pred)
                        Queue.insert(0, comp_pred)
                        if current_node in Queue:
                            Queue.remove(current_node)
                        Queue.insert(1, current_node)
            if pred_is_visited:
                last_part_res = -1
                h = 0
                rand = []
                # loop over all partitions in the deployment
                for part_idx in random_dep.partitions_indices:
                    part = comp.partitions[part_idx]
                    # get the indices of the component and the deployment
                    i = self.system.dic_map_part_idx[comp.name][part.name][0]
                    h_idx = self.system.dic_map_part_idx[comp.name][part.name][1]
                    if last_part_res >= self.system.cloud_start_index:
                        possible_res[comp_idx][:, :self.system.cloud_start_index]=0
                    idx = np.nonzero(possible_res[i][h_idx,:])[0]
                    index = list(set(candidate_nodes).intersection(idx))
                    prob = 1/len(index)
                    step = 0
                    rn = np.random.random()
                    rand.append(rn)
                    for r in np.arange(0, 1, prob):
                        if rn > r and rn <= r + prob:
                            j = index[step]
                        else:
                            step += 1
                    last_part_res=j
                    y[i][h_idx,j] = 1
                    y_hat[i][h_idx,j] = 1
                    if self.system.graph.G.succ[comp.name] != {}:
                        #if part.Next == list(self.system.graph.G.succ[comp.name].keys())[0]:
                        if part.Next == list(self.system.graph.G.succ[comp.name].keys()):
                            for next_idx in range(len(part.Next)):
                                self.system.graph.G[comp.name][part.Next[next_idx]]["data_size"] = part.data_size[next_idx]
                visited[current_node]=True
                for node in self.system.graph.G.neighbors(current_node):
                    if not visited[node]:
                        if node not in Queue:
                            Queue.append(node)



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
    def increase_number_of_resource(self, comp_idx, part_idx, solution):

        increased = False

        # get the index of the resources where the component partition is 
        # allocated
        resource_idx = np.nonzero(solution.Y_hat[comp_idx][part_idx,:])[0]
        # get all components' index and partitions' index that are running j
        partitions_with_j=self.get_partitions_with_j(solution.Y_hat,resource_idx)
        # check if the number of resources assigned to the partition is 
        # lower than the number of available resources
        assigned_resources = solution.Y_hat[comp_idx][part_idx, resource_idx]
        if assigned_resources < self.system.resources[resource_idx].number:
            # if so, increase the number of assigned resources in all partitions
            for comp_part in partitions_with_j:
                solution.Y_hat[comp_part[0]][comp_part[1], resource_idx] += 1
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

        self.logger.level += 1
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

                    self.logger.log("y_bar[{}] = {}".\
                                    format(resource_idx,y_bar[resource_idx].max()),
                                    7)

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
                    feasible = new_performance[0]
                    if feasible:
                        # update the current solution
                        new_result.solution = new_solution
                        new_result.performance = new_performance
                        y_bar = new_result.solution.get_y_bar()
                        self.logger.log("feasible", 7)

        self.logger.level -= 1
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

    ## Method to get all partitions that is running on resource j
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
        counter_obj_evaluation=0
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
                    counter_obj_evaluation+=1
                    result.performance = performance
                    new_feasible_results.append(result)
        if len(new_feasible_results)>0:
            # sort the list of result      
            new_sorted_results = sorted(new_feasible_results, key=lambda x: x.cost)
            #new_sorted_solution=[x.solution for x in new_sorted_results]
        # return the list of neibors
        return new_sorted_results, counter_obj_evaluation


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

    ## Method to sort all nodes increasingly except FaaS by utilization and cost 
    #   @param self The object pointer
    #   @param Y_hat Assignment matrix includes all components and partitions assigned to resources 
    #   @return 1) The sorted list of resources by utilization and cost, respectively. 
    #           Each item of list includes the index, utilization and cost of the resource.
    #           The list is sorted by utilization, but for the nodes with same utilization, it is sorted by cost
    #           2) The sorted list of resources by cost and utilization, respectively.  
    #           Each item of list includes the index, utilization and cost of the resource.
    #           The list is sorted by utilization, but for the nodes with same utilization, it is sorted by cost
    def sort_nodes(self, Y_hat):

        #min_utilization=np.inf
        idx_min_U_node=[]
        # loop over all alternative resources
        edge=EdgePE()
        cloud=ServerFarmPE()
        for j in range(self.system.cloud_start_index):
              # compute the utilization of current node
            utilization=edge.compute_utilization(j, Y_hat, self.system)
            if not math.isnan(utilization) and utilization>0:
                # add the information of node to the list includes node index, utilization and cost
                idx_min_U_node.append((j, utilization, self.system.resources[j].cost))

        for j in range(self.system.cloud_start_index,self.system.FaaS_start_index):
             # compute the utilization of current node
            utilization=cloud.compute_utilization(j, Y_hat, self.system)
            if not math.isnan(utilization) and utilization>0:
                # add the information of node to the list includes node index, utilization and cost
                idx_min_U_node.append((j, utilization, self.system.resources[j].cost))

        # sort the list based on utilization and cost respectively     
        sorted_node_by_U_cost= sorted(idx_min_U_node, key=lambda element: (element[1], element[2]))
        # sort the list based on cost and utilization respectively
        sorted_node_by_cost_U= sorted(idx_min_U_node, key=lambda element: (element[2], element[1]))
        # return the index of best alternative
        return sorted_node_by_U_cost, sorted_node_by_cost_U

    ## Method to change the current solution by changing component placement
    #   @param self The object pointer
    #   @param solution Current solution 
    #   @param sorting_method indicate the sorting order of nodes. 
    #           If sorting_method=0, the list of nodes are sorted by utilization and cost respectively
    #           otherwise the list of nodes are sorted by cost and utilization respectively.
    #   @return A list neigbors (new solutions) sorted by cost
    def change_component_placement(self, solution, sorting_method=0):
        counter_obj_evaluation=0
        neighbors=[]
        new_sorted_results=None
        # get a sorted list of nodes' index with their utilization and cost (except FaaS)
        if sorting_method==0:
            nodes_sorted_list= self.sort_nodes(solution.Y_hat)[0]
        else:
            nodes_sorted_list= self.sort_nodes(solution.Y_hat)[1]
        # get resource with maximum utilization as source node 
        # first while loop is used for the case that if we cannot find any neighbors by starting with last node of the sorted list as a source node,
        # we try to search to find neigbors by starting from second last node in the list and etc. Continue untile finding neigbors
        selected_node=len(nodes_sorted_list)
        j=1
        # explore nodes_sorted_list until finding at least one neighbor
        while len(neighbors)<1 and j<=len(nodes_sorted_list):
            # get resource with maximum utilization/cost as source node
            selected_node=len(nodes_sorted_list)-j
            idx_source_node=nodes_sorted_list[selected_node][0]
            # get all partitions located in higest utilization node
            partitions=self.get_partitions_with_j(solution.Y_hat,idx_source_node)
            # get the list of nodes and computational layers in used 
            active_res_idxs, active_camputationallayers = self.get_active_res_computationallayers(solution.Y_hat)
            # loop over partitions
            for part in partitions:
                # get all alternative resources of the partitions
                alternative_res_idxs=self.alternative_resources(part[0],part[1],solution)
                # # set a boolean variable to break, if best destination is founded
                # find=False
                i=0

                 # search to find the best campatible resource with lowest utilization for current partition
                #while not find and i<len(nodes_sorted_list)-1 and i<j:
                while  i<len(nodes_sorted_list)-1 and i<j:

                    des_node_idx=nodes_sorted_list[i][0]
                    # Check some conditions to avoid violating the limitation of our problem that says:
                    # Only one node can be in used in each computational layer 
                    # So, the destination node can be used only if it is one of running (active) node, 
                    # or its computational layer is not an active computational layer

                    if des_node_idx in active_res_idxs or \
                            self.system.resources[des_node_idx].CLname not in active_camputationallayers:
                        if des_node_idx in alternative_res_idxs:

                            # get a copy of current solution as a new temprary assignment matrix (Y_hat)
                            new_temp_Y_hat=copy.deepcopy(solution.Y_hat)
                            # get all partitions running on the destination node
                            partitions_min_U = self.get_partitions_with_j(solution.Y_hat,des_node_idx)
                            # assign the current partition to the new alternative node in new Y_hat with maximume number of its instances
                            new_temp_Y_hat[part[0]][part[1]][idx_source_node]=0
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
                                new_result_1 = self.reduce_cluster_size(idx_source_node, result)
                                new_result = self.reduce_cluster_size(des_node_idx, new_result_1)
                                # compute the cost
                                new_result.cost = new_result.objective_function(self.system)
                                counter_obj_evaluation+=1
                                new_result.performance = performance
                                # add new result in neigbor list
                                neighbors.append(new_result)


                    i+=1


                # if not find:
                #      print("There is no alternative node for partition "+str(part[1]) +" of component "+ str(part[0])+" in current solution." )
            # if some neighbors are founded, sort them by cost and return the list    
            if len(neighbors)>0:
                new_sorted_results = sorted(neighbors, key=lambda x: x.cost)
                #new_sorted_solutions=[x.solution for x in new_sorted_results]

            else:
               # print("No neighbor could be find by changing component placement of source node " +str(idx_source_node))
                new_sorted_results = None
            j += 1

       # if new_sorted_results is None:
        #     print("Any neighbors could not be found by changing component placement for the current solution")
        return new_sorted_results, counter_obj_evaluation



    ## Method to change the current solution by changing resource type
    #   @param self The object pointer
    #   @param solution Current solution 
    #   @param sorting_method indicate the sorting order of nodes. 
    #           If sorting_method=0, the list of nodes are sorted by utilization and cost respectively
    #           otherwise the list of nodes are sorted by cost and utilization respectively.
    #   @return A list neigbors (new solutions) sorted by cost   
    def change_resource_type(self, solution, sorting_method=0):

        counter_obj_evaluation=0
        neighbors=[]
        new_sorted_results=None
        # get a sorted list of nodes' index with their utilization and cost (except FaaS)
        if sorting_method==0:
            nodes_sorted_list= self.sort_nodes(solution.Y_hat)[0]
        else:
            nodes_sorted_list= self.sort_nodes(solution.Y_hat)[1]
        selected_node=len(nodes_sorted_list)
        i=1
        while len(neighbors)<1 and i<=len(nodes_sorted_list):
            # get resource with maximum utilization/cost as source node
            selected_node=len(nodes_sorted_list)-i
            idx_source_node=nodes_sorted_list[selected_node][0]

            # get all partitions located in higest utilization node
            partitions=self.get_partitions_with_j(solution.Y_hat,idx_source_node)
            alternative_res_idxs_parts=[]
            # get a list of set of alternative nodes for partitions runing on source node
            for part in partitions:
                alternative_res_idxs_parts.append(set(self.alternative_resources(part[0],part[1],solution)))
            # get the intersection of the alternative nodes of all partitions runing on source node
            candidate=set.intersection(*alternative_res_idxs_parts)
            candidate_nodes = [i for i in candidate if i < self.system.FaaS_start_index]

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
                            self.system.resources[des].CLname==self.system.resources[idx_source_node].CLname:
                                new_temp_Y_hat=copy.deepcopy(solution.Y_hat)
                                # get all partitions running on the destination node
                                partitions_on_candidate=self.get_partitions_with_j(solution.Y_hat,des)
                                # assign the maximume instance number of destination node to the partitions that are running on source node
                                for part in partitions:
                                    new_temp_Y_hat[part[0]][part[1]][idx_source_node]=0
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
                                    counter_obj_evaluation+=1
                                    new_result.performance = performance
                                    # add the new result to the neigbor list
                                    neighbors.append(new_result)

                if len(neighbors)>0:
                    # sort neighbor list by cost and return the best one
                    new_sorted_results = sorted(neighbors, key=lambda x: x.cost)
                    #new_sorted_solutions=[x.solution for x in new_sorted_results]
            #     else:
            #         print("No neighbor could be find by changing resource "+str(idx_source_node)+" because no feasible solution exists given the shared compatiblie nodes ")

            # else:
            #     print("No neighbor could be find by changing resource "+str(idx_source_node)+" because no shared compatiblie node exists")
            i+=1
        #if  new_sorted_results==None:
        #    print("Any neighbors could not be found by changing resource type for the current solution")
        return new_sorted_results, counter_obj_evaluation

    ## Method to change the current solution by moveing partitions from edge or cloud toFaaS
    #   @param self The object pointer
    #   @param solution Current solution 
    #   @param sorting_method indicate the sorting order of nodes. 
    #           If sorting_method=0, the list of nodes are sorted by utilization and cost respectively
    #           otherwise the list of nodes are sorted by cost and utilization respectively.
    #   @return A list neigbors (new solutions) sorted by cost
    def move_to_FaaS(self, solution, sorting_method=0):
        counter_obj_evaluation=0
        neighbors=[]
        new_sorted_results=None
        # get a sorted list of nodes' index with their utilization and cost (except FaaS)
        if sorting_method==0:
            nodes_sorted_list= self.sort_nodes(solution.Y_hat)[0]
        else:
            nodes_sorted_list= self.sort_nodes(solution.Y_hat)[1]
        selected_node=len(nodes_sorted_list)

        # sort FaaS by memory and cost respectively
        sorted_FaaS=self.system.sorted_FaaS_by_memory_cost
        # if we need to sort FaaS by cost and memory respectively, we use self.system.sorted_FaaS_by_cost_memory
        sorted_FaaS_idx=[i[0] for i in sorted_FaaS]
        i=1
        while len(neighbors)<1 and i<=len(nodes_sorted_list):
            # get resource with maximum utilization/cost as source node
            selected_node=len(nodes_sorted_list)-i
            idx_source_node=nodes_sorted_list[selected_node][0]

            # get all partitions located in higest utilization node
            partitions=self.get_partitions_with_j(solution.Y_hat,idx_source_node)
            alternative_res_idxs_parts=[]
            all_FaaS_compatible=True
            # get a list of alternative FaaS for partitions runing on source node
            for part in partitions:
                x=self.alternative_resources(part[0],part[1],solution)
                FaaS_alternatives=x[x>=self.system.FaaS_start_index]
                if len(FaaS_alternatives)<1:
                    all_FaaS_compatible=False
                alternative_res_idxs_parts.append(FaaS_alternatives)




            if all_FaaS_compatible:
                new_temp_Y_hat=copy.deepcopy(solution.Y_hat)
                for idx, part in enumerate(partitions):
                    idx_alternative=[sorted_FaaS_idx.index(j) for j in alternative_res_idxs_parts[idx]]
                    # get the FaaS with maximum memory/cost
                    des=sorted_FaaS_idx[max(idx_alternative)]
                    #alternative_res_idxs_parts[idx] sorted_FaaS
                    new_temp_Y_hat[part[0]][part[1]][idx_source_node]=0

                    new_temp_Y_hat[part[0]][part[1]][des]=1
                # create new solution by new assignment
                new_temp_solution=Configuration(new_temp_Y_hat)
                # check feasibility
                performance=new_temp_solution.check_feasibility(self.system)

                if performance[0]:
                    # create a new result
                    new_result=Result()
                    new_result.solution = new_temp_solution
                    new_result.cost = new_result.objective_function(self.system)
                    counter_obj_evaluation+=1
                    new_result.performance = performance
                    # add the new result to the neigbor list
                    neighbors.append(new_result)
            else:

                for idx, part in enumerate(partitions):
                    if len(alternative_res_idxs_parts[idx])>0:
                        new_temp_Y_hat=copy.deepcopy(solution.Y_hat)

                        idx_alternative = [sorted_FaaS_idx.index(j) for j in alternative_res_idxs_parts[idx]]
                        des = sorted_FaaS_idx[max(idx_alternative)]
                        #alternative_res_idxs_parts[idx] sorted_FaaS
                        new_temp_Y_hat[part[0]][part[1]][idx_source_node]=0

                        new_temp_Y_hat[part[0]][part[1]][des]=1
                    # create new solution by new assignment
                        new_temp_solution=Configuration(new_temp_Y_hat)
                        # check feasibility
                        performance=new_temp_solution.check_feasibility(self.system)

                        if performance[0]:
                            # create a new result
                            result=Result()
                            result.solution = new_temp_solution
                            # reduce the cluster size of destination node
                            new_result = self.reduce_cluster_size(idx_source_node, result)
                            new_result.cost = new_result.objective_function(self.system)
                            counter_obj_evaluation+=1
                            new_result.performance = performance
                            # add the new result to the neigbor list
                            neighbors.append(new_result)
            i+=1

        if len(neighbors)>0:
            # sort neighbor list by cost and return the best one
            new_sorted_results = sorted(neighbors, key=lambda x: x.cost)
            #new_sorted_solutions=[x.solution for x in new_sorted_results]
        #else:
        #    print("Any neighbors could not be found by moveing to FaaS for the current solution")

        return new_sorted_results, counter_obj_evaluation


    ## Method to move the partitions running on FaaS to the edge/cloud
    #   @param self The object pointer
    #   @param solution Current solution 
    #   @param sorting_method indicate the sorting order of nodes. 
    #           If sorting_method=0, the list of nodes are sorted by utilization and cost respectively
    #           otherwise the list of nodes are sorted by cost and utilization respectively.
    #   @return A list neigbors (new solutions) sorted by cost
    def move_from_FaaS(self, solution, sorting_method=0):
        counter_obj_evaluation=0
        neighbors=[]
        new_sorted_results=None
        # get a sorted list of nodes' index with their utilization and cost (except FaaS)
        if sorting_method==0:
            nodes_sorted_list= self.sort_nodes(solution.Y_hat)[0]
        else:
            nodes_sorted_list= self.sort_nodes(solution.Y_hat)[1]
        # call the method to get all partitions located in FaaS
        partitions_with_FaaS=self.get_partitions_with_FaaS(solution.Y_hat)
         # get the list of nodes and computational layers in used
        active_res_idxs, active_camputationallayers=self.get_active_res_computationallayers(solution.Y_hat)
        # loop over partitions
        for part in partitions_with_FaaS:
                # get all alternative resources of the partitions
                alternative_res_idxs=self.alternative_resources(part[0],part[1],solution)
                alternative_res_idxs=alternative_res_idxs[alternative_res_idxs<self.system.FaaS_start_index]
                i=0
                find=False
                 # search to find the best campatible resource with lowest utilization for current partition
                #while not find and i<len(nodes_sorted_list)-1 and i<j:
                while i < len(nodes_sorted_list)-1 and not find :

                    des_node_idx=nodes_sorted_list[i][0]
                    # Check some conditions to avoid violating the limitation of our problem that says:
                    # Only one node can be in used in each computational layer
                    # So, the destination node can be used only if it is one of running (active) node,
                    # or its computational layer is not an active computational layer

                    if des_node_idx in active_res_idxs or \
                            self.system.resources[des_node_idx].CLname not in active_camputationallayers:
                        if des_node_idx in alternative_res_idxs:

                            # get a copy of current solution as a new temprary assignment matrix (Y_hat)
                            new_temp_Y_hat=copy.deepcopy(solution.Y_hat)
                            # get all partitions running on the destination node
                            partitions_on_des=self.get_partitions_with_j(solution.Y_hat,des_node_idx)
                            # assign the current partition to the new alternative node in new Y_hat with maximume number of its instances
                            new_temp_Y_hat[part[0]][part[1]][part[2]]=0
                            new_temp_Y_hat[part[0]][part[1]][des_node_idx]=self.system.resources[des_node_idx].number

                            if len(partitions_on_des)>0:
                                # assign the maximume instance number of destination node to the partitions that are running on destination node
                                for part_des in partitions_on_des:
                                    new_temp_Y_hat[part_des[0]][part_des[1]][des_node_idx]=self.system.resources[des_node_idx].number
                            # creat a solution by new assignment (Y_hat)
                            new_temp_solution=Configuration(new_temp_Y_hat)
                            # check if new solution is feasible
                            performance=new_temp_solution.check_feasibility(self.system)
                            if performance[0]:
                                # creat new result
                                result=Result()
                                result.solution = new_temp_solution
                                # reduce cluster size of the destination node
                                new_result = self.reduce_cluster_size(des_node_idx, result)
                                # compute the cost
                                new_result.cost = new_result.objective_function(self.system)
                                counter_obj_evaluation+=1
                                new_result.performance = performance
                                # add new result in neigbor list
                                neighbors.append(new_result)
                                find=True

                    i+=1

        if len(neighbors) > 0:
            new_sorted_results = sorted(neighbors, key=lambda x: x.cost)
       # if new_sorted_results is None:
        #     print("Any neighbors could not be found by moving from FaaS to edge/cloud for the current solution")
        return new_sorted_results, counter_obj_evaluation

    ## Method to union and sort the set of neighbors came from three methods: change_resource_type, change_component_placement, change_FaaS
    #   @param self The object pointer
    #   @param solution Current solution 
    #   @return A list neigbors (new solutions) sorted by cost   
    def union_neighbors(self, solution):

        neighborhood=[]
        # get the neighbors by changing FaaS configuration
        neighborhood1, counter_obj_evaluation1=self.change_FaaS(solution)
        # get the neighbors by changing resource type
        neighborhood2, counter_obj_evaluation2=self.change_resource_type(solution)
        # get the neigbors by changing component placement
        neighborhood3, counter_obj_evaluation3=self.change_component_placement(solution)
        # get the neigbors by moveing to FaaS
        neighborhood4, counter_obj_evaluation4=self.move_to_FaaS(solution)
        # get the neigbors by moveing from FaaS to edge/cloud
        neighborhood5, counter_obj_evaluation5=self.move_from_FaaS(solution)
        counter_obj_evaluation = counter_obj_evaluation1 + counter_obj_evaluation2 + counter_obj_evaluation3 +\
                               counter_obj_evaluation4 + counter_obj_evaluation5
        # mixe all neigbors
        if neighborhood1 is not None:
            neighborhood.extend(neighborhood1)
        if neighborhood2 is not None:
            neighborhood.extend(neighborhood2)
        if neighborhood3 is not None:
            neighborhood.extend(neighborhood3)
        if neighborhood4 is not None:
            neighborhood.extend(neighborhood4)
        if neighborhood5 is not None:
            neighborhood.extend(neighborhood5)
        # sort the neighbors list by cost 
        sorted_neighborhood=sorted(neighborhood, key=lambda x: x.cost)
        # if two solution have the same cost, check if the solutions are the same and drop one of them
        new_sorted_neighborhood=copy.deepcopy(sorted_neighborhood)
        for neighbor_idx in range(len(sorted_neighborhood)-1):
            if sorted_neighborhood[neighbor_idx].cost==sorted_neighborhood[neighbor_idx+1].cost:
                if sorted_neighborhood[neighbor_idx].solution==sorted_neighborhood[neighbor_idx+1].solution:

                    new_sorted_neighborhood.remove(new_sorted_neighborhood[neighbor_idx])

        return new_sorted_neighborhood, counter_obj_evaluation

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
    def __init__(self, system,seed, log = Logger()):
        super().__init__(system,seed, log)


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

        # generate random solution and check its feasibility
        self.logger.level += 1
        self.logger.log("Generate random solution", 3)
        y_hat, res_parts_random, VM_numbers_random, CL_res_random = self.create_random_initial_solution()
        result.solution = Configuration(y_hat, self.logger)
        self.logger.log("Start check feasibility: {}".format(time.time()), 3)
        feasible = result.check_feasibility(self.system)
        self.logger.log("End check feasibility: {}".format(time.time()), 3)

        # if the solution is feasible, compute the corresponding cost 
        # before and after updating the clusters size
        if feasible:
            self.logger.log("Solution is feasible", 3)
            # compute cost
            self.logger.log("Compute cost", 3)
            result.objective_function(self.system)
            # update the cluster size of cloud resources
            self.logger.log("Update cluster size", 3)
            new_result= copy.deepcopy(result)
            for j in range(self.system.FaaS_start_index):
                new_result = self.reduce_cluster_size(j, new_result)
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
    #   @param MaxTime Time duration that random greedy should run (if we set time instead of iteration)
    #   @return (1) Best Solution.Result before cluster update
    #           (2) Solution.EliteResults object storing the given number of 
    #           Solution.Result objects sorted by minimum cost
    #           (4) List of the random parameters
    def random_greedy(self, MaxIt = 1, K = 1, MaxTime=0):

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
        iteration=0
        start=time.time()
        while iteration<MaxIt or time.time()-start<MaxTime:

            self.logger.log("#iter {}: {}".format(iteration, time.time()), 3)
            # perform a step
            result, new_result, random_param = self.step()
            # update the results and the lists of random parameters
            elite.add(new_result)
            if result < best_result_no_update:
                best_result_no_update = copy.deepcopy(result)
            #res_parts_random_list.append(random_param[0])
            #VM_numbers_random_list.append(random_param[1])
            #CL_res_random_list.append(random_param[2])
            iteration += 1
        self.logger.level -= 1
        random_params = [res_parts_random_list, VM_numbers_random_list,
                         CL_res_random_list]

        return best_result_no_update, elite, random_params


## IteratedLocalSearch  
#
# Specialization of Algorithm      
class IteratedLocalSearch(Algorithm):

    pass


## Tabu Search
#
# Specialization of Algorithm that constructs the optimal solution through a
# Tabu Search approach
class Tabu_Search(TabuSearch,RandomGreedy):

    ## TabuSearchSolid class constructor
    #   @param self The object pointer
    #   @param seed A seed to generate random values
    #   @param Max_It_RG Maximum iterations of random greedy
    #   @param system A System.System object
    #   @param Max_time_RG Maximum time of running RG
    #   @param K The number of elite solutions obtained by RG as the starting point of SA
    #   @param besties_RG A list of starting points
    #   @param log Object of Logger.Logger type
    def __init__(self,Max_It_RG,seed,  system,Max_time_RG=0,K=1, besties_RG=None, log = Logger()):

        self.Max_It_RG=Max_It_RG
        self.Max_time_RG=Max_time_RG
        self.counter_obj_evaluation=0
        RandomGreedy.__init__(self,system,seed, log)
        # compute initial solution
        if besties_RG is None:
            best_result_no_update, elite, random_params=self.random_greedy(K=K,MaxIt = self.Max_It_RG, MaxTime=self.Max_time_RG)
            if len(elite.elite_results)<K:
                Len=len(elite.elite_results)
            else:
                Len=K
            for i in range(Len):
                self.starting_points.append(elite.elite_results[i].solution)
        else:
            self.starting_points=besties_RG


    ## Method to get a list of neigbors
    #   @param self The object pointer
    #   @return A list of solutions (neighbors)
    def _neighborhood(self ):

        neighborhood, counter_obj_evaluation =self.union_neighbors(self.current)
        self.counter_obj_evaluation += counter_obj_evaluation
        return [x.solution for x in neighborhood]

    ## Method to get the cost of current solution
    #   @param self The object pointer
    #   @param solution The current solution
    #   @return The cost of current solution
    def _score(self, solution):
        self.counter_obj_evaluation += 1
        return solution.objective_function(self.system)

        ## Method to run SA
    #   @param self The object pointer
    #   @param tabu_size Maximum size of tabu list
    #   @param method The method to select one of neghbors in each iteration ("best" or "random")
    #   @param max_steps Maximum steps of tabu search
    #   @param min_score Minimum cost
    #   @param Max_time Maximum running time of Simulated Annealing in second
    def run_TS (self,method, tabu_size, min_score, max_steps,Max_time=None):

        Max_time_starting_point=Max_time/len(self.starting_points)
        best_best_solution=None
        best_best_cost=np.inf
        best_current_cost_list=[]
        best_best_cost_list=[]
        best_time_list=[]
        Starting_point_best_costs=[]
        Starting_point_solutions=[]
        Starting_point_current_cost_list=[]
        Starting_point_best_cost_list=[]
        Starting_point_time_list=[]


        for starting_point in self.starting_points:
            TabuSearch.__init__(self,starting_point, tabu_size, max_steps,Max_time_starting_point, min_score)
            best_solution, best_cost,current_cost_list,best_cost_list, time_list=self.run(method)
            if best_cost<np.inf:
                Starting_point_best_costs.append(best_cost)
                Starting_point_solutions.append(best_solution)
                Starting_point_current_cost_list.append(current_cost_list)
                Starting_point_best_cost_list.append(best_cost_list)
                Starting_point_time_list.append(time_list)
                if best_cost<best_best_cost:
                    best_best_cost=best_cost
                    best_best_solution=copy.deepcopy(best_solution)
                    best_current_cost_list= copy.deepcopy(current_cost_list)
                    best_best_cost_list= copy.deepcopy(best_cost_list)
                    best_time_list= copy.deepcopy(time_list)
        result = Result()
        result.solution = best_best_solution
        feasible = result.check_feasibility(self.system)
        if feasible:
            result.objective_function(self.system)

        best_sol_info = [best_best_cost, best_best_solution, best_current_cost_list, best_best_cost_list, best_time_list]
        Starting_points_info=[Starting_point_solutions, Starting_point_best_costs, Starting_point_current_cost_list, Starting_point_best_cost_list, Starting_point_time_list]
        return result, best_sol_info, Starting_points_info




## Simulated Annealing

class Simulated_Annealing(SimulatedAnnealing, RandomGreedy):
     ## SimulatedAnnealingSolid class constructor
    #   @param self The object pointer
    #   @param seed A seed to generate random values
    #   @param Max_It_RG Maximum iterations of random greedy
    #   @param system A System.System object
    #   @param Max_time_RG Maximum running time of Random Greedy in second
    #   @param K The number of elite solutions obtained by RG as the starting point of SA
    #   @param log Object of Logger.Logger type
    def __init__(self, Max_It_RG ,seed, system, Max_time_RG=0, K=1, besties_RG=None, log = Logger()):

        self.Max_It_RG=Max_It_RG
        self.Max_time_RG=Max_time_RG
        self.starting_points=[]
        self.counter_obj_evaluation=0
        RandomGreedy.__init__(self,system,seed, log)
        # compute initial solution
        if besties_RG is None:
            best_result_no_update, elite, random_params=self.random_greedy(K=K,MaxIt = self.Max_It_RG, MaxTime=self.Max_time_RG)
            if len(elite.elite_results)<K:
                Len=len(elite.elite_results)
            else:
                Len=K
            for i in range(Len):
                self.starting_points.append(elite.elite_results[i].solution)
        else:
            self.starting_points=besties_RG


     ## Method to get a list of neigbors
    #   @param self The object pointer
    #   @return A solution (neighbor)
    def _neighbor(self ):

        neighborhood, counter_obj_evaluation = self.union_neighbors(self.current_state)
        self.counter_obj_evaluation += counter_obj_evaluation
        if len(neighborhood)>0:
            x=neighborhood[np.argmin([self._energy(x) for x in neighborhood])]
            sol=x.solution
        else:
            sol=None
        return sol

    ## Method to get the cost of current solution
    #   @param self The object pointer
    #   @param solution The current solution
    #   @return The cost of current solution
    def _energy(self, solution):
        self.counter_obj_evaluation += 1
        return solution.objective_function(self.system)

    ## Method to run SA
    #   @param self The object pointer
    #   @param schedule_constant Constant value in annealing schedule function
    #   @param temp_begin Beginning temperature
    #   @param max_steps Maximum steps of tabu search
    #   @param min_energy Minimum cost
    #   @param schedule Temperature decay function ("linear" or "exponential")
    #   @param Max_time Maximum running time of Simulated Annealing in second
    def run_SA (self,temp_begin, schedule_constant, max_steps, min_energy, schedule,Max_time=None):
        Max_time_starting_point=Max_time/len(self.starting_points)
        best_best_solution=None
        best_best_cost=np.inf
        best_current_cost_list=[]
        best_best_cost_list=[]
        best_time_list=[]
        Starting_point_best_costs=[]
        Starting_point_solutions=[]
        Starting_point_current_cost_list=[]
        Starting_point_best_cost_list=[]
        Starting_point_time_list=[]

        for starting_point in self.starting_points:
            SimulatedAnnealing.__init__(self, starting_point, temp_begin, schedule_constant, max_steps,
                                        Max_time_starting_point, min_energy, schedule)
            best_solution, best_cost,current_cost_list,best_cost_list, time_list=self.run()
            if best_cost<np.inf:
                Starting_point_best_costs.append(best_cost)
                Starting_point_solutions.append(best_solution)
                Starting_point_current_cost_list.append(current_cost_list)
                Starting_point_best_cost_list.append(best_cost_list)
                Starting_point_time_list.append(time_list)
                if best_cost<best_best_cost:
                    best_best_cost=best_cost
                    best_best_solution=copy.deepcopy(best_solution)
                    best_current_cost_list= copy.deepcopy(current_cost_list)
                    best_best_cost_list= copy.deepcopy(best_cost_list)
                    best_time_list= copy.deepcopy(time_list)
        result = Result()
        result.solution = best_best_solution
        feasible = result.check_feasibility(self.system)
        if feasible:
            result.objective_function(self.system)

        best_sol_info = [best_best_cost, best_best_solution, best_current_cost_list, best_best_cost_list, best_time_list]
        Starting_points_info=[Starting_point_solutions, Starting_point_best_costs, Starting_point_current_cost_list, Starting_point_best_cost_list, Starting_point_time_list]
        return result, best_sol_info, Starting_points_info

## Genetic algorithm

class Genetic_algorithm(GeneticAlgorithm, RandomGreedy):
     ## GeneticAlgorithm class constructor
    #   @param self The object pointer
    #   @param seed A seed to generate random values
    #   @param Max_It_RG Maximum iterations of random greedy
    #   @param max_steps Maximum steps of tabu search
    #   @param min_score Minimum cost
    #   @param system A System.System object
    #   @param log Object of Logger.Logger type
    def __init__(self,Max_It_RG,seed, crossover_rate, mutation_rate, max_steps, system,Max_time_RG=0,Max_time=None,max_fitness=None, besties_RG=None, log = Logger()):

        self.Max_It_RG=Max_It_RG
        self.Max_time_RG=Max_time_RG
        self.counter_obj_evaluation=0
        RandomGreedy.__init__(self,system,seed, log)
        GeneticAlgorithm.__init__(self,  crossover_rate, mutation_rate, max_steps,Max_time, max_fitness)
        if besties_RG is None:
            self.besties_RG=None
        else:
            self.besties_RG=besties_RG

    def _initial_population(self, K):
        if self.besties_RG is None:
            best_result_no_update, elite, random_params = self.random_greedy(MaxIt = self.Max_It_RG, K=K,MaxTime=self.Max_time_RG)
            population=[]
            if len(elite.elite_results)<K:
                Len=len(elite.elite_results)
            else:
                Len=K
            for i in range(Len):

                population.append(elite.elite_results[i].solution)
        else:
            population=self.besties_RG
        return population

    def _fitness(self, member):
        
        self.counter_obj_evaluation += 1
        return member.objective_function(self.system)

    def _mutate(self, member):
        """
        Randomly mutates a member

        :param member: a member
        :return: mutated member
        """
        if self.mutation_rate >= np.random.random():
            results = None
            fns = [self.change_FaaS, self.change_resource_type, self.change_component_placement,
                   self.move_to_FaaS, self.move_from_FaaS]
            np.random.shuffle(fns)
            while results is None and len(fns) > 0:
                fn = fns.pop()
                results, counter_obj_evaluation = fn(member)
                self.counter_obj_evaluation += counter_obj_evaluation
            if results is not None:
                member = list([result.solution for result in results])
            else:
                member=list([member])
        else:
            member=list([member])
        return member

    def _crossover(self, parent1, parent2):
        # get a partition point randomly
        partition = np.random.randint(0, len(self.population[0].Y_hat) - 1)
        part1 = copy.deepcopy(parent1.Y_hat[0:partition])
        part2 = copy.deepcopy(parent2.Y_hat[partition:])
        children = self.mix_parts(partition, part1, part2)

        part1 = copy.deepcopy(parent2.Y_hat[0:partition])
        part2 = copy.deepcopy(parent1.Y_hat[partition:])
        children.extend(self.mix_parts(partition, part1, part2))
        solutions=[]
        for child in children:
            # creat a solution by new child (Y_hat)
            new_solution=Configuration(child)
            # check if new solution is feasible
            performance=new_solution.check_feasibility(self.system)
            if performance[0]:
                solutions.append(new_solution)

        return solutions

    def mix_parts(self,partition,part1,part2):
        act_res1, act_CL1 = self.get_active_res_computationallayers(part1)
        act_res2, act_CL2 = self.get_active_res_computationallayers(part2)

        intersec_CL = set(act_CL1).intersection(act_CL2)
        child1=copy.deepcopy(part1+ part2)
        children=[]
        if intersec_CL is not None:
            two_child=False
            for cl in intersec_CL:
                res_idxs= [compLayer.resources for compLayer in self.system.CLs if compLayer.name==cl][0]
                if res_idxs[0] >= self.system.FaaS_start_index:
                    continue
                res_part1=set(res_idxs).intersection(act_res1).pop()
                res_part2=set(res_idxs).intersection(act_res2).pop()
                if res_part1 == res_part2:
                    continue
                else:
                    if not two_child:
                        two_child=True
                        child2=copy.deepcopy(part1+ part2)
                    # get all partitions of part2 that are running on res_part2
                    partitions2 = self.get_partitions_with_j(part2,res_part2)
                    # get all partitions that are running on res_part1
                    partitions1 = self.get_partitions_with_j(part1,res_part1)
                    # to create child1,
                    for part in partitions2:
                        child1[part[0]+partition][part[1]][res_part2]=0
                        child1[part[0]+partition][part[1]][res_part1]=part1[partitions1[0][0]][partitions1[0][1]][res_part1]
                    for part in partitions1:
                        child2[part[0]][part[1]][res_part1]=0

                        child2[part[0]][part[1]][res_part2]=part2[partitions2[0][0]][partitions2[0][1]][res_part2]

        children.append(child1)
        if two_child:
            children.append(child2)

        return children

    def run_GA(self, K):

         # increase indentation level for logging
        self.logger.level += 1
        self.logger.log("Run Genetic Algorithm", 3)
        best_member, best_fitness, best_sol_cost_list_GA, time_list_GA, population=self.run(K)
        # initialize results
        result = Result()
        result.solution = best_member
        self.logger.log("Start check feasibility: {}".format(time.time()), 3)
        feasible = result.check_feasibility(self.system)
        self.logger.log("End check feasibility: {}".format(time.time()), 3)

        # if the solution is feasible, compute the corresponding cost
        # before and after updating the clusters size
        if feasible:
            self.logger.log("Solution is feasible", 3)
            # compute cost
            self.logger.log("Compute cost", 3)
            result.objective_function(self.system)
            self.counter_obj_evaluation += 1
        else:
            new_result = copy.deepcopy(result)
        self.logger.level -= 2
        return result,  best_sol_cost_list_GA, time_list_GA, population

class HyperOpt():


    ## HyperOpt class constructor
    #   @param self The object pointer
    #   @param system A System.System object
   def __init__(self, system):
        self.system=system

   ## The objective function of HyperOpt
    #   @param self The object pointer
    #   @param args All arguments with their search space
    #   @return a solution found by HyperOpt
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
                if resource_random_list[idx] != np.inf :
                    candidate_nodes.append(l.resources[resource_random_list[idx]])
            resource_count += len(l.resources)

        for comp_idx, comp in enumerate(self.system.components):
            # set a deployment for each component based on what HypetOpt selected
            random_dep=comp.deployments[deployment_random_list[comp_idx]]

            h=0

            #for part_idx, part in enumerate(random_dep.partitions_indices):
            for part_idx in random_dep.partitions_indices:
                part = comp.partitions[part_idx]
                 # get the indices of the component and the deployment selected by HypetOpt
                i=self.system.dic_map_part_idx[comp.name][part.name][0]
                # get the indices of compatible resources and compute the
                # intersection with the selected resources in each
                # computational layer


                h_idx=self.system.dic_map_part_idx[comp.name][part.name][1]
                if  max(max(prob_res_selection_dep_list))<=1:
                    idx=np.nonzero(self.system.compatibility_matrix[i][h_idx,:])[0]
                    # extract a resource index in the intersection
                    index=list(set(candidate_nodes).intersection(idx))
                    prob=1/len(index)
                    step=0
                    h= random_dep.partitions_indices.index(h_idx)
                    rn=prob_res_selection_dep_list[comp_idx][h]

                    for r in np.arange(0,1,prob):
                        if rn>r and rn<=r+prob:
                            j= index[step]

                        else:
                            step+=1
                else:
                    h= random_dep.partitions_indices.index(h_idx)
                    j=int(prob_res_selection_dep_list[comp_idx][h])
                y[i][h_idx][j]=1
                y_hat[i][h_idx][j]=1
                # if the partition is the last partition (i.e., its successor
                # is the successor of the component), update the size of
                # data transferred between the components
                if self.system.graph.G.succ[comp.name] != {}:

                    #if part.Next == list(self.system.graph.G.succ[comp.name].keys())[0]:
                    if part.Next == list(self.system.graph.G.succ[comp.name].keys()):
                        for next_idx in range(len(part.Next)):
                            self.system.graph.G[comp.name][part.Next[next_idx]]["data_size"] = part.data_size[next_idx]


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
                        if max(max(prob_res_selection_dep_list))<=1:
                             y_hat[i][h][j] = y[i][h][j]*(VM_number_random_list[j]+1)
                        else:
                             y_hat[i][h][j] = y[i][h][j]*(VM_number_random_list[j])
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
    #   @param seed Seed for random number generation
    #   @param iteration_number The iteration number for HyperOpt
    #   @param vals_list The list of value to feed the result of RandomGreedy to HyperOpt
    #   @return the best cost and the best solution found by HyperOpt
   def random_hyperopt(self,seed, max_time,vals_list=[]):
        #np.random.seed(int(time.time()))
        # set the seed for replicability
        #os.environ['HYPEROPT_FMIN_SEED'] = "1"#str(np.random.randint(1,1000))
        # Create search spaces for all random variable by defining a dictionary for each of them
        resource_random_list=[]
        for idx, l in enumerate(self.system.CLs):
            # Create search spaces for all computational layer except the last one with FaaS
            if l != list(self.system.CLs)[-1]:

                resource_random_list.append(hp.randint("res"+str(idx),len(l.resources)))


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
                if max_part<len(list(dep.partitions_indices)):
                    max_part=len(list(dep.partitions_indices))
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

       # set seed
        rstate = np.random.default_rng(seed)
        try:
             # run fmin method to search and find the best solution
            best = fmin(fn=self.objective, space=space, algo=rand.suggest, trials=trials,  timeout=max_time,rstate=rstate)

        except:

            best_cost= float("inf")

        # check if HyperOpt could find solution
        if best_cost== float("inf"):
            # if HyperOpt cannot find any feasible solution
             solution=None
        else:
             # if HyperOp find a feasible solution extract the solution from fmin output
            best_cost=trials.best_trial["result"]["loss"]
            cost, solution=self.extract_HyperOpt_result(best)

        return best_cost, solution

   ## Method to extract the best solution of HperOpt and converting it to Y_hat
    #   @param self The object pointer
    #   @param best The output of fmin method
    #   @return the best cost and the best solution found by HyperOpt
   def extract_HyperOpt_result(self,best ):

            I=len(self.system.components)
            J=len(self.system.resources)
            y_hat=[]
            y=[]
            # initialize Y_hat and y by 0
            for i in range(I):
                H,J=self.system.compatibility_matrix[i].shape
                y_hat.append(np.full((H, J), 0, dtype=int))
                y.append(np.full((H, J), 0, dtype=int))

            # initialize the list of selected resources by best solution of HyperOpt
            resource_random_list=[]
            # extract the selected resources by best solution of HyperOpt in CLs
            candidate_nodes=[]
            for idx, l in enumerate(self.system.CLs):
                if l == list(self.system.CLs)[-1]:
                    random_num=l.resources
                    candidate_nodes.extend(random_num)
                else:
                    if str(best["res"+str(idx)]).isdigit():
                         candidate_nodes.append(l.resources[best["res"+str(idx)]])
                         resource_random_list.append( best["res"+str(idx)])

            resources=[ v for k,v in best.items() if 'com_res' in k]
             # extract the selected deployments of components by best solution of HyperOpt
            for comp_idx, comp in enumerate(self.system.components):

                random_dep=comp.deployments[best["dep"+str(comp_idx)]]
                #deployment_random_list.append(best["dep"+str(idx)])
                h=0
                for part_idx in random_dep.partitions_indices:
                    part = comp.partitions[part_idx]
                     # pick the selected compatible resources by the best solution of HperOpt for each partition
                     #  and set Y_hat according to it
                    i=self.system.dic_map_part_idx[comp.name][part.name][0]
                    h_idx=self.system.dic_map_part_idx[comp.name][part.name][1]
                    if max(resources)<=1:
                        idx=np.nonzero(self.system.compatibility_matrix[i][h_idx,:])[0]
                        index=list(set(candidate_nodes).intersection(idx))
                        prob=1/len(index)
                        step=0
                        h= random_dep.partitions_indices.index(h_idx)
                        rn=best["com_res"+str(comp_idx)+str(h)]
                        for r in np.arange(0,1,prob):
                            if rn>r and rn<=r+prob:
                                j= index[step]

                            else:
                                step+=1
                    else:
                        h= random_dep.partitions_indices.index(h_idx)
                        j=int(best["com_res"+str(comp_idx)+str(h)])
                    y[i][h_idx][j]=1
                    y_hat[i][h_idx][j]=1

                    if  self.system.graph.G.succ[comp.name]!={}:
                        if part.Next == list(self.system.graph.G.succ[comp.name].keys()):

                            for next_idx in range(len(part.Next)):
                                self.system.graph.G[comp.name][part.Next[next_idx]]["data_size"] = part.data_size[next_idx]
            # pick the number of VM selected by best solution of HyperOpt and set it in Y_hat
            if self.system.FaaS_start_index!=float("inf"):
                edge_VM=self.system.FaaS_start_index
            else:
                edge_VM=J
            for j in range(edge_VM):
                for i in range(I):
                     H=self.system.compatibility_matrix[i].shape[0]
                     for h in range(H):
                        if y[i][h][j]>0:
                            if max(resources)<=1:
                                 y_hat[i][h][j] = y[i][h][j]*(best["VM"+str(j)]+1)
                            else:
                                 y_hat[i][h][j] = y[i][h][j]*(best["VM"+str(j)])

            # create the solution by new Y_hat extracted by the best solution of HyperOpt
            solution=Configuration(y_hat)
            # compute cost
            cost=solution.objective_function(self.system)

            return cost, solution

   ## Method to create the trials according to solutions of random greedy to feed HyperOpt
    #   @param self The object pointer
    #   @param solutions The solutions of random greedy
    #   @param res_parts_random_list The lists of random parameters needed to select compatible resources assigned to partitions
    #   @param VM_numbers_random_list The list of random number selected by random greedy
    #   @param CL_res_random_list The list of randomly selected resources in CLs by random greedy
    #   @return a list of values to feed HyperOpt
   def creat_trials_by_RandomGreedy(self, solutions, res_parts_random_list,
                                    VM_numbers_random_list, CL_res_random_list):

        # Initialize the list of values to feed HyperOpt
        vals_list=[]
        # For all solutions found by random greedy, create the parameters that HyperOpt needs to create the same solutions
        for solution_idx, solution in enumerate(solutions):

           vals={}
           com_res={}
           dep={}
           # initialize the resources assigned to parts by random value,
           # it is necessary for HyperOpt to has the space of all parameters even if the deployment of the partition
           # is not selected by random greedy. The deployments selected by random greedy
           # will assigned to the same resources as random greedy while the others will assign to resources randomly.
           for idx, comp in enumerate(self.system.components):
                 max_part=0

                 for  dep in comp.deployments:
                     if max_part<len(list(dep.partitions_indices)):
                         max_part=len(list(dep.partitions_indices))
                 for i in range( max_part):
                     vals["com_res"+str(idx)+str(i)]=random.random()

           # for each component and deployment, set the random parameters selected by random greedy
           for comp_idx, y in enumerate(solution.Y_hat):


               H,J=y.shape
               for h in range(H):

                   resource_idx=np.nonzero(y[h,:])[0]
                   if len(resource_idx)>0:
                       if solution_idx==2 and comp_idx==2:
                            x=1
                       for dep_idx, dep in enumerate(self.system.components[comp_idx].deployments):
                           if h in dep.partitions_indices:

                               vals["dep"+str(comp_idx)]=dep_idx
                               try:
                                     vals["com_res"+str(comp_idx)+str(dep.partitions_indices.index(h))]=res_parts_random_list[solution_idx][comp_idx][dep.partitions_indices.index(h)]
                               except:
                                   print("solution_idx: "+ str(solution_idx)+" comp_idx:" + str(comp_idx) + " part_idx:" + str(dep_idx) )
           # set the selected node selected by random greedy
           for idx, l in enumerate(CL_res_random_list[solution_idx]):

                        vals["res"+str(idx)]=l
           # set the number of VM selected by random greedy
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
                    vals["VM"+str(j)]= VM_numbers_random_list[solution_idx][j]

           vals_list.append(vals)
        return  vals_list

 ## Method to create the trials according to solutions of a heuristic to feed HyperOpt
    #   @param self The object pointer
    #   @param solutions The solutions of random greedy
    #   @param res_parts_random_list The lists of random parameters needed to select compatible resources assigned to partitions
    #   @param VM_numbers_random_list The list of random number selected by random greedy
    #   @param CL_res_random_list The list of randomly selected resources in CLs by random greedy
    #   @return a list of values to feed HyperOpt
   def creat_trials_by_Heuristic(self, solutions):

        # Initialize the list of values to feed HyperOpt
        vals_list=[]
        # For all solutions found by random greedy, create the parameters that HyperOpt needs to create the same solutions
        for solution_idx, solution in enumerate(solutions):

           vals={}
           com_res={}
           dep={}
           selected_res_list=[]
           number_of_vms_list=[]
           y_bar = solution.get_y_bar()
           for idx, comp in enumerate(self.system.components):
                 max_part=0

                 for dep in comp.deployments:
                     if max_part<len(list(dep.partitions_indices)):
                         max_part=len(list(dep.partitions_indices))
                 for i in range( max_part):
                     vals["com_res"+str(idx)+str(i)]=1
           for comp_idx, y in enumerate(solution.Y_hat):
               H,J=y.shape
               for h in range(H):

                   resource_idx=np.nonzero(y[h,:])[0]
                   if len(resource_idx)>0:
                       selected_res_list.append(resource_idx[0])
                       number_of_vms_list.append(y_bar[resource_idx[0]])
                       for dep_idx, dep in enumerate(self.system.components[comp_idx].deployments):
                           if h in dep.partitions_indices:

                                vals["dep"+str(comp_idx)]=dep_idx
                                compatible_res_idx=np.nonzero(self.system.compatibility_matrix[comp_idx][h,:])[0]

                                h_idx= dep.partitions_indices.index(h)
                                vals["com_res"+str(comp_idx)+str(h_idx)]=resource_idx[0]

           if self.system.FaaS_start_index!=float("inf"):
                edge_VM=self.system.FaaS_start_index
           else:
                edge_VM=J
           for j in range(edge_VM):

                vals["VM"+str(j)]=np.random.randint(0, self.system.resources[j].number )



           for l in range(len(self.system.CLs)):
                    flag=False
                    res_list=self.system.CLs[l].resources
                    for j in res_list:#zip(selected_res_list,number_of_vms_list):
                        if j in selected_res_list:
                            flag=True
                            vals["res"+str(l)]=self.system.CLs[l].resources.index(j)
                            idx=selected_res_list.index(j)
                            vals["VM"+str(j)]=number_of_vms_list[idx]
                            break
                    if flag==False:
                        vals["res"+str(l)]= np.inf
           # initialize the resources assigned to parts by random value,
           # it is necessary for HyperOpt to has the space of all parameters even if the deployment of the partition
           # is not selected by random greedy. The deployments selected by random greedy
           # will assigned to the same resources as random greedy while the others will assign to resources randomly.


           vals_list.append(vals)
        return vals_list
