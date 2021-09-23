from classes.Logger import Logger
from classes.Performance import SystemPerformanceEvaluator
import numpy as np
import itertools
import json


## Configuration
class Configuration:
    
    ## @var Y_hat
    # List of 2D numpy arrays storing the number of Resources.Resource 
    # assigned to each Graph.Component.Partition
    
    ## @var local_slack_value
    # Slack values related to Constraints.LocalConstraints
    
    ## @var global_slack_value
    # Slack value related to Constraints.GlobalConstraints
    
    ## @var logger
    # Object of Logger type, used to print general messages
   

    ## Configuration class constructor
    #   @param self The object pointer
    #   @param Y_hat List of 2D numpy arrays storing the number of 
    #                Resources.Resource assigned to each 
    #                Graph.Component.Partition
    #   @param log Object of Logger type
    def __init__(self, Y_hat, log=Logger()):
        self.Y_hat = Y_hat
        self.local_slack_value = np.full(len(self.Y_hat), np.inf, 
                                         dtype = float)
        self.global_slack_value = None
        self.logger = log
    
    
    ## Method to get information about the used resources
    #   @param self The object pointer
    #   @return 1D numpy array whose j-th element is 1 if resource j is used
    def get_x(self):
        J = self.Y_hat[0].shape[1]
        x = np.full(J, 0, dtype = int)
        for i in range(len(self.Y_hat)):
            x[self.Y_hat[i].sum(axis=0) > 0] = 1
        return x
    
    
    ## Method to get the maximum number of used resources of each type
    #   @param self The object pointer
    #    @return 1D numpy array whose j-th element denotes the maximum number 
    #            of used resources of type j
    def get_y_bar(self):
        y_max = []
        for i in range(len(self.Y_hat)):
            y_max.append(np.array(self.Y_hat[i].max(axis=0), dtype=int))
        y_bar = [max(i) for i in itertools.zip_longest(*y_max, fillvalue=0)]
        return np.array(y_bar)
    
   
    ## Method to check if the preliminary constraints are satisfied
    #   @param self The object pointer
    #   @param compatibility_matrix Compatibility matrix
    #   @param resource_number 1D numpy array storing the number of each 
    #                          resource
    #   @return True if the preliminary constraints are satisfied
    def preliminary_constraints_check_assignments(self, compatibility_matrix, 
                                                  resource_number):
        feasible = True
        i = 0
        I = len(self.Y_hat)
        
        # loop over all components until an infeasible assignment is found
        while i < I and feasible:
            
            # check that each component partition is assigned to exactly one 
            # resource
            if all(np.count_nonzero(row) == 1 for row in self.Y_hat[i]):
                # convert y_hat to y (binary)
                y = np.array(self.Y_hat[i] > 0, dtype = int)
                
                # check that only compatible resources are assigned to the 
                # component partitions
                if np.all(np.less_equal(y, compatibility_matrix[i])):
                    
                    # check that the number of resources assigned to each 
                    # component partition is at most equal to the number of 
                    # available resources of that type
                    if any(self.Y_hat.max(axis=0)[0:resource_number.shape[0]]>resource_number):
                        feasible = False
                else:
                    feasible = False
            else:
                    feasible = False
            
            # increment the component index
            i += 1
        
        return feasible       
        
    
    ## Method to check if memory constraints of all Resources.Resource 
    # objects are satisfied
    #   @param self The object pointer
    #   @param S A System.System object
    #   @return True if the constraints are satisfied
    def memory_constraints_check(self, S):
        
        # create y from y_hat
        I = len(S.components)
        J = len(S.resources)
        y = []
        for i in range(I):
            y.append(np.array(self.Y_hat[i] > 0, dtype = int))
       
        # for each resource, check if the sum of the memory requirements of 
        # all component partitions assigned to it is greater than the maximum 
        # capacity of the resource
        feasible = True
        j = 0
        while j < J and feasible:
            memory = 0
            for i, c in zip(y, S.components):
                memory += (i[:,j] * np.array(list(h.memory for h in c.partitions))).sum(axis=0)
                if memory > S.resources[j].memory:
                    feasible = False
            j += 1
        
        return feasible
    
    
    ## Method to check that, if a Graph.Component.Partition object is executed
    # on a Resources.VirtualMachine or a Resources.FaaS, all its successors
    # are not executed on Resources.EdgeNode objects (assignments cannot move
    # back from cloud to edge)
    #   @param self The object pointer
    #   @param S A System.System object
    #   @return True if the constraint is satisfied
    def move_backward_check(self, S):
        
        feasible = True
        last_part_res = -1

        # loop over all components
        for node in S.graph.G:
            # get the component index
            i = S.dic_map_com_idx[node]
            # get the indices of resources where the component partitions are 
            # executed
            for y in self.Y_hat[i]:
                h = np.nonzero(y)
                if np.size(h) > 0:
                    # if the last partition was executed on cloud, check 
                    # that the current is not executed on edge
                    if last_part_res >= S.cloud_start_index:
                        if h[0][0] < S.cloud_start_index:
                            feasible = False
                    last_part_res = h[0][0]
        
        return feasible     


    ## Method to check the feasibility of the current configuration
    #   @param self The object pointer
    #   @param S A System.System object
    def check_feasibility(self, S):
        
        # increase indentation level for logging
        self.logger.level += 1
        
        # define status of components and paths response times and constraints
        I = len(S.components)
        components_performance = [[True, np.infty]] * I
        paths_performance = []
        
        # check if the memory constraints are satisfied
        self.logger.log("Memory constraints check", 4)
        feasible = self.memory_constraints_check(S)
        
        if feasible:
            # check if the cloud placement constraint is satisfied
            self.logger.log("Cloud placement constraint check", 4)
            feasible = self.move_backward_check(S)
            
            if feasible:
                # check if all local constraints are satisfied
                self.logger.log("Local constraints check", 4)
                for LC in S.local_constraints:
                    i = LC.component_idx
                    components_performance[i] = LC.check_feasibility(S, self)
                    feasible = feasible and components_performance[i][0]
                
                if feasible:
                    self.logger.log("Global constraints check", 4)
                    # check global constraints
                    for GC in S.global_constraints:
                        paths_performance.append(GC.check_feasibility(S, self))
                        feasible = feasible and paths_performance[-1][0]

        if not feasible:
            self.logger.level += 1
            self.logger.log("Unfeasible", 4)
            self.logger.level -= 1
        
        self.logger.level -= 1
        
        return feasible, paths_performance, components_performance


    ## Method to compute the cost of a feasible solution
    #   @param self The object pointer
    #   @param S A System.System object
    #   @return total cost
    def objective_function(self, S):
        
        J = len(S.resources)
        
        # get information about the used resources and the max number of 
        # used resources of each type
        x = self.get_x()   
        y_bar = self.get_y_bar()
        
        # compute costs
        costs = []
        # compute cost of edge
        for j in range(S.cloud_start_index):
            costs.append(S.resources[j].cost * x[j])
        #
        # compute cost of VMs
        for j in range(S.cloud_start_index, S.FaaS_start_index):
            costs.append(S.resources[j].cost * y_bar[j])
        #
        # compute the cost of FaaS and transition cost if not using SCAR
        for j in range(S.FaaS_start_index, J):
            for i in range(len(self.Y_hat)):
                part_indexes = np.nonzero(S.compatibility_matrix[i][:,j])[0]
                for part_idx in part_indexes:
                    costs.append(S.resources[j].cost * self.Y_hat[i][part_idx][j] * S.components[i].comp_Lambda * S.T )
        
        total_cost = sum(costs)
        
        return total_cost
    
    
    ## Method to convert the solution description into a json object
    #   @param self The object pointer
    #   @param S A System.System object
    #   @param response_times Response times of all Graph.Component objects
    #                         (default: None)
    #   @param path_response_times Response times of all paths involved in 
    #                              Constraints.GlobalConstraint objects
    #                              (default: None)
    #   @param cost Cost of the Solution (default: None)
    #   @return Json object storing the solution description
    def to_json(self, S, response_times = None, path_response_times = None, 
                cost = None):
        
        # compute response times of all components
        if not response_times:
            PE = SystemPerformanceEvaluator(Logger(self.logger.stream,
                                             self.logger.verbose,
                                             self.logger.level + 1))
            response_times = PE.compute_performance(S, self.Y_hat)
        
        solution_string = '{"Lambda": ' + str(S.Lambda)
        
        # write components deployments and response times
        solution_string += ',  "components": {'
        I = len(self.Y_hat)
        for i in range(I):
            component = S.components[i].name
            component_string = ' "' + component + '": {'
            allocation = np.nonzero(self.Y_hat[i])
            # loop over partitions
            for idx in range(len(allocation[0])):
                # get partition and resource indices
                h = allocation[0][idx]
                j = allocation[1][idx]
                # get partition name
                partition = [key for key, (value1, value2) in \
                             S.dic_map_part_idx[component].items() \
                                 if value1 == i and value2 == h][0]
                component_string += ' "' + partition + '": {'
                # get computational layer name
                CL = S.resources[j].CLname
                component_string += '"' + CL + '": {'
                # get resource name and description
                resource = S.resources[j].name
                description = S.description[resource]
                component_string += ('"' + resource + \
                                     '": {"description": "' + \
                                    description + '"')
                # get cost and memory
                res_cost = S.resources[j].cost * self.Y_hat[i][h,j]
                memory = S.resources[j].memory
                component_string += (', "cost": ' + str(res_cost) + \
                                     ', "memory": ' + str(memory))
                # get number of FaaS-related information
                if j < S.FaaS_start_index:  
                    number = int(self.Y_hat[i][h,j])
                    component_string += ', "number": ' + str(number) + '}}},'
                else:
                    idle_time_before_kill = S.resources[j].idle_time_before_kill
                    transition_cost = S.resources[j].transition_cost
                    component_string += (', "idle_time_before_kill": ' + \
                                        str(idle_time_before_kill) + \
                                        ', "transition_cost": ' + \
                                        str(transition_cost) + '}}},')
            # get response time and corresponding threshold
            component_string += ' "response_time": ' + str(response_times[i])
            component_string += ', "response_time_threshold": '
            threshold = [LC.max_res_time for LC in S.local_constraints \
                         if LC.component_idx == i]
            if len(threshold) > 0:
                component_string += str(threshold[0]) + '},'
            else:
                component_string += '"inf"},'
            solution_string += component_string
        
        # write global constraints
        solution_string = solution_string[:-1] + '},  "global_constraints": {'
        for GCidx in range(len(S.global_constraints)):
            solution_string += S.global_constraints[GCidx].__str__(S.components)
            # write response time of the path
            solution_string = solution_string[:-1] + ', "path_response_time": '
            if path_response_times:
                solution_string += str(path_response_times[GCidx][1])
            else:
                f, time = S.global_constraints[GCidx].check_feasibility(S,self)
                solution_string += str(time)
            solution_string += '},'
        
        # write total cost
        solution_string = solution_string[:-1] + '},  "total_cost": '
        if cost:
            solution_string += str(cost)
        else:
            solution_string += str(self.objective_function(S))
        solution_string += '}'
        
        # load string as json
        solution_string = solution_string.replace('0.,', '0.0,')
        jj = json.dumps(json.loads(solution_string), indent = 2)
        
        return jj
    
    
    ## Method to print the solution in json format, either on screen or on 
    # the file whose name is passed as parameter
    #   @param self The object pointer
    #   @param S A System.System object
    #   @param response_times Response times of all Graph.Component objects
    #                         (default: None)
    #   @param path_response_times Response times of all paths involved in 
    #                              Constraints.GlobalConstraint objects
    #                              (default: None)
    #   @param cost Cost of the Solution (default: None)
    #   @param solution_file Name of the file where the solution should be 
    #                        printed (default: "")
    def print_solution(self, S, response_times = None, 
                       path_response_times = None, 
                       cost = None, solution_file = ""):

        # get solution description in json format
        jj = self.to_json(S, response_times, path_response_times, cost)
        
        # print
        if solution_file:
            with open(solution_file, "w") as f:
                f.write(jj)
        else:
            print(jj)

