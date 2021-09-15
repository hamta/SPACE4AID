from classes.Logger import Logger
from classes.Graph import DAG, Component
from classes.Resources import ComputationalLayer, VirtualMachine, EdgeNode, FaaS
from classes.NetworkTechnology import NetworkDomain
from classes.Performance import NetworkPE, ServerFarmPE, FunctionPE, EdgePE
from classes.Constraints import LocalConstraint, GlobalConstraint
import json
import sys
import numpy as np
import copy   

## System
#
# Class to store the system description with all the relevant information
class System:
    
    ## @var cloud_start_index
    # Index of the first Resources.VirtualMachine object available in 
    # System.resources

    ## @var CLs
    # List of all the available Resources.ComputationalLayer objects
    
    ## @var compatibility_dict
    # Dictionary representing the compatibility of each 
    # Graph.Component.Deployment.Partition object in a given Graph.Component 
    # with the available Resources.Resource objects
    
    ## @var compatibility_matrix 
    # List of 2D numpy arrays representing the compatibility between all 
    # Graph.Component.Deployment.Partition objects in each Graph.Component 
    # and the available Resources.Resource
    
    ## @var components 
    # List of all Graph.Component objects
            
    ## @var demand_dict 
    # Dictionary representing the demand matrix of the available 
    # Graph.Component.Deployment.Partition objects in a given 
    # Graph.Component when they are deployed on different  
    # Resources.Resource objects
    
    ## @var demand_matrix 
    # List of 2D numpy arrays representing the demand to run all 
    # Graph.Component.Deployment.Partition objects in each Graph.Component 
    # on the available Resources.Resource
    
    ## @var description
    # Dictionary associating to each Resources.Resource object the 
    # corresponding description
    
    ## @var dic_map_com_idx
    # Dictionary associating to the name of each Graph.Component object the 
    # corresponding index in System.components
    
    ## @var dic_map_part_idx
    # Nested dictionary associating to the name of each Graph.Component and 
    # each Graph.Component.Deployment.Partition in the Graph.Component a 
    # tuple whose first element is the index of the Graph.Component object 
    # in System.components and whose second element is the index of 
    # the Graph.Component.Deployment.Partition
    
    ## @var dic_map_res_idx
    # Dictionary associating to the name of each Resources.Resource object 
    # the corresponding index in System.resources
    
    ## @var error
    # Object of Logger.Logger class, used to print error messages on sys.stderr
    
    ## @var FaaS_start_index
    # Index of the first Resources.FaaS object available in System.resources
    
    ## @var global_constraints
    # List of Constraints.GlobalConstraint objects
    
    ## @var graph 
    # Object of Graph.DAG type
    
    ## @var local_constraints
    # List of Constraints.LocalConstraint objects
    
    ## @var Lambda
    # Incoming load
    
    ## @var logger
    # Object of Logger.Logger type, used to print general messages
    
    ## @var network_technologies
    # List of NetworkTechnology objects, characterized by a given access 
    # delay and bandwidth
    
    ## @var resources 
    # List of all the available Resources.Resource objects
    
    ## @var T
    # Time
  
    
    ## System class constructor: initializes all the System class members 
    # starting either from the configuration file or the json object passed 
    # as parameters
    #   @param self The object pointer
    #   @param system_file Configuration file describing the system
    #   @param system_json Json object describing the system
    #   @param log Object of Logger.Logger type
    def __init__(self, system_file = "", system_json = None, log = Logger()):
        self.logger = log
        self.error = Logger(stream=sys.stderr, verbose=1)
        if system_file != "":
            self.logger.log("Loading system from configuration file", 1)
            self.read_configuration_file(system_file)
        elif system_json:
            self.logger.log("Loading system from json object", 1)
            self.load_json(system_json)
        else:
            self.error.log("ERROR: no configuration file or json specified",1)
            sys.exit(1)
    
    
    ## Method to read a configuration file providing the system description 
    # (in json format) and populate the class members accordingly
    #   @param self The object pointer
    #   @param system_file Configuration file describing the system 
    #                      (json format)
    def read_configuration_file(self, system_file):
        
        # load json file
        with open(system_file) as f:
            data = json.load(f)
        
        self.load_json(data)
        
    
    ## Method to load the system description provided in json format and 
    # initialize all the class members accordingly
    #   @param self The object pointer
    #   @param data Json object describing the system 
    def load_json(self, data):
        
        # increase indentation level for logging
        self.logger.level += 1
        
        # initialize system DAG
        if "DirectedAcyclicGraph" in data.keys():
            self.logger.log("Initializing DAG", 2)
            DAG_dict = data["DirectedAcyclicGraph"]
            self.graph = DAG(graph_dict=DAG_dict, 
                             log=Logger(stream=self.logger.stream,
                                        verbose=self.logger.verbose,
                                        level=self.logger.level+1))
        else:
            self.error.log("ERROR: no DAG available in configuration file", 1)
            sys.exit(1)
        
        # initialize lambda
        if "Lambda" in data.keys():
            self.logger.log("Initializing Lambda", 2)
            self.Lambda = float(data["Lambda"])
        else:
            self.error.log("ERROR: no Lambda available in configuration file", 1)
            sys.exit(1)
               
        # initialize components and local constraints
        if "Components" in data.keys():
            # load components info
            C = data["Components"]
            LC = None
            # load local constraints info
            if "LocalConstraints" in data.keys():
                self.logger.log("Initializing components and local constraints", 
                                2)
                LC = data["LocalConstraints"]
            else:
                self.logger.log("No local constraints specified", 3)
            # perform initialization
            self.initialize_components(C, LC)
        else:
            self.error.log("ERROR: no components available in configuration file", 1)
            sys.exit(1)
        
        # get global constraints to initialize the maximun response time of 
        # application paths
        if "GlobalConstraints" in data.keys():
            self.logger.log("Initializing global constraints", 2)
            GC = data["GlobalConstraints"]
            self.convert_GCdic_to_list(GC)
        else:
            self.logger.log("No global constraints specified", 3)
        
        # initialize resources, together with their description and the 
        # dictionary that maps their names to their indices, and 
        # computational layers
        self.logger.log("Initializing resources and computational layers", 2)
        self.initialize_resources(data)
        
        # load Network Technology
        if "NetworkTechnology" in data.keys():
            self.logger.log("Initializing network technology", 2)
            self.network_technologies = []
            NT = data["NetworkTechnology"]
            # load Network Domains
            for ND in NT:
                if "computationallayers" in NT[ND].keys() \
                    and "AccessDelay" in NT[ND].keys() \
                        and "Bandwidth" in NT[ND].keys():
                    network_domain = NetworkDomain(ND,
                                          list(NT[ND]["computationallayers"]),
                                          float(NT[ND]["AccessDelay"]),
                                          float(NT[ND]["Bandwidth"]),
                                          NetworkPE())
                    self.network_technologies.append(network_domain)
                else:
                    self.error.log("ERROR: missing field in {} description".\
                                   format(ND), 1)
                    sys.exit(1)
        else:
            self.error.log("ERROR: no NetworkTechnology available in configuration file", 1)
            sys.exit(1)
       
        # load dictionary of component-to-node compatibility 
        self.logger.log("Initializing compatibility and demand matrices", 2)
        if "CompatibilityMatrix" in data.keys():
            self.compatibility_dict = data["CompatibilityMatrix"]
        else:
            self.error.log("ERROR: no CompatibilityMatrix available in configuration file", 1)
            sys.exit(1)
        
        # variable to check, for each component, if all resources mentioned 
        # in the demand matrix are compatible with the component itself
        is_compatible = True
     
        # load demand matrix
        if "DemandMatrix" in data.keys():
            self.demand_dict = data["DemandMatrix"]
            # check if, for each component, all resources mentioned in the 
            # demand matrix are compatible with the component itself
            for c in self.demand_dict:
                for r in self.demand_dict[c].keys():
                    if not r in self.compatibility_dict[c]:
                        is_compatible = False
                        self.error.log("ERROR: demand and compatibility matrix are not consistent", 1)
                        sys.exit(1)    
        else:
            is_compatible = False
            self.error.log("ERROR: no DemandMatrix available in configuration file", 1)
            sys.exit(1)
       
        # if demand matrix is available and it is consistent with 
        # compatibility matrix, convert both these dictionaries to arrays
        if is_compatible:
            self.convert_dic_to_matrix()
         
        # initialize time
        self.logger.log("Initializing time", 2)
        if "Time" in data.keys():
            self.T = float(data["Time"])
        
        # restore indentation level for logging
        self.logger.level -= 1
       
 
    ## Method to initialize the components based on the dictionary of 
    # component extracted from config file and graph, and to compute the input 
    # lambda (workload) of each component. The input graph can be a general 
    # case with some parallel branches.
    # @param self The object pointer
    # @param C Dictionary of components came from configuration file.
    # @param LC Dictionary of LocalConstraints came from configuration file.
    def initialize_components(self, C, LC):
        self.components = []
        self.dic_map_com_idx = {}
        self.dic_map_part_idx = {}
        localconstraints = {}
        self.local_constraints = []
        comp_idx = 0
        # loop over components
        for c in C :
            # check if the component is in the graph
            if self.graph.G.has_node(c):
                if len(C[c]) > 0:
                    deployments = []
                    partitions = []
                    part_idx = 0
                    temp = {}
                # check if the node c has any input edge
                if self.graph.G.in_edges(c):
                    Sum = 0
                    # if the node c has some input edges, its Lambda is equal 
                    # to the sum of products of lambda and weight of its 
                    # input edges.
                    for n, c, data in self.graph.G.in_edges(c, data=True):
                        prob = float(data["transition_probability"])
                        ll = self.components[self.dic_map_com_idx[n]].comp_Lambda
                        Sum += prob * ll
                        # loop over all candidate deployments
                        for s in C[c]:
                            part_Lambda = -1
                            part_idx_list=[]
                            if len(C[c][s]) > 0:
                               
                                # loop over all partitions
                                for h in C[c][s]:
                                    if part_Lambda > -1:
                                        prob = float(C[c][s][h]["early_exit_probability"])
                                        part_Lambda *= (1 - prob)
                                    else:
                                        part_Lambda=copy.deepcopy(Sum)
                                    temp[h] = (comp_idx, part_idx)
                                    partitions.append(Component.Partition(h,float(C[c][s][h]["memory"]),part_Lambda,
                                                                                     float(C[c][s][h]["early_exit_probability"]),
                                                                                     C[c][s][h]["next"],float(C[c][s][h]["data_size"])))
                                    part_idx_list.append(part_idx)
                                    part_idx += 1
                            deployments.append(Component.Deployment(s, part_idx_list))    
                        self.dic_map_part_idx[c] = temp
                    comp = Component(c, deployments,partitions, Sum)
                    self.components.append(comp)
                else:
                    # if the node c does not have any input edge, it is the 
                    # first node of a path and its Lambda is equal to input 
                    # lambda
                    for s in C[c]:
                        part_Lambda = -1
                        part_idx_list=[]
                        if len(C[c][s]) > 0:
                            partitions = []
                            # loop over all partitions
                            for h in C[c][s]:
                                if part_Lambda > -1:
                                    prob = float(C[c][s][h]["early_exit_probability"])
                                    part_Lambda *= (1 - prob)
                                else:
                                    part_Lambda = copy.deepcopy(self.Lambda)
                                temp[h] = (comp_idx, part_idx)
                                partitions.append(Component.Partition(h,float(C[c][s][h]["memory"]),part_Lambda,
                                                                                 float(C[c][s][h]["early_exit_probability"]),
                                                                                 C[c][s][h]["next"],float(C[c][s][h]["data_size"])))
                                part_idx_list.append(part_idx)
                                part_idx += 1
                        deployments.append(Component.Deployment(s, part_idx_list))    
                    self.dic_map_part_idx[c] = temp
                    self.components.append(Component(c, deployments,partitions, self.Lambda))
            else:
                self.error.log("ERROR: no match between components in DAG and system input file", 1)
                sys.exit(1)
            self.dic_map_com_idx[c] = comp_idx
    
            # initialize local constraint
            if LC and c in LC:
                self.local_constraints.append(LocalConstraint(self.dic_map_com_idx[c],
                                                              float(LC[c]["local_res_time"])))
                localconstraints[c] = self.local_constraints[-1]
            
            comp_idx += 1
       
    
    ## Method to initialize resources, together with their description and the 
    # dictionary that maps their names to their indices, and 
    # computational layers
    #   @param self The object pointer
    #   @param data Json object storing all the relevant information
    def initialize_resources(self, data):
        
        # increase indentation level for logging
        self.logger.level += 1
        
        self.resources = []
        self.description = {}
        self.dic_map_res_idx = {}
        self.CLs = []
        resource_idx = 0
        #
        # edge resources
        if "EdgeResources" in data.keys():
            self.logger.log("Edge resources", 3)
            ER = data["EdgeResources"]
            # loop over computational layers
            for CL in ER:
                cl = ComputationalLayer(CL)
                # loop over nodes and add them to the corresponding layer
                for node in ER[CL]:
                    temp = ER[CL][node]
                    if "number" in temp.keys() and "cost" in temp.keys() \
                        and "memory" in temp.keys():
                        new_node = EdgeNode(CL, node, float(temp["cost"]),
                                            float(temp["memory"]),EdgePE(),
                                            int(temp["number"]))
                        self.resources.append(new_node)
                        self.dic_map_res_idx[node] = resource_idx
                        cl.add_resource(resource_idx)
                        resource_idx += 1
                    else:
                        self.error.log("ERROR: missing field in {} description".\
                                       format(node), 1)
                        sys.exit(1)
                    # add the resource description to the corresponding 
                    # dictionary
                    if "description" in temp.keys():
                        self.description[node] = temp["description"]
                    else:
                        self.description[node] = "No description"
                # add the new computational layer
                self.CLs.append(cl)    
        #
        # cloud resources
        if "CloudResources" in data.keys():
            self.logger.log("Cloud resources", 3)
            # initialize the index corresponding to the first cloud resource
            self.cloud_start_index = resource_idx
            CR = data["CloudResources"]
            # loop over computational layers
            for CL in CR:
                cl = ComputationalLayer(CL)
                # loop over VMs and add them to the corresponding layer
                for VM in CR[CL]:
                    temp = CR[CL][VM]
                    if "number" in temp.keys() and "cost" in temp.keys() \
                        and "memory" in temp.keys():
                        new_vm = VirtualMachine(CL, VM, float(temp["cost"]), 
                                                float(temp["memory"]), 
                                                ServerFarmPE(),
                                                int(temp["number"]))
                        self.resources.append(new_vm)
                        self.dic_map_res_idx[VM] = resource_idx
                        cl.add_resource(resource_idx)
                        resource_idx += 1
                    else:
                        self.error.log("ERROR: missing field in {} description".\
                                       format(VM), 1)
                        sys.exit(1)
                    # add the resource description to the corresponding 
                    # dictionary
                    if "description" in temp.keys():
                        self.description[VM] = temp["description"]
                    else:
                        self.description[VM] = "No description"
                # add the new computational layer
                self.CLs.append(cl)
        #
        # faas resources
        if "FaaSResources" in data.keys():
            self.logger.log("FaaS resources", 3)
            # initialize the index corresponding to the first FaaS resource
            self.FaaS_start_index = resource_idx
            FR = data["FaaSResources"]
            # initialize transition cost
            if "transition_cost" in FR.keys():
                transition_cost = float(FR["transition_cost"])
            else:
                self.error.log("ERROR: missing transition cost in FaaSResources", 1)
                sys.exit(1)
            # loop over computational layers
            for CL in FR:
                if CL.startswith("computationallayer"):
                    cl = ComputationalLayer(CL)
                    # loop over functions and add them to the corresponding 
                    # layer
                    for func in FR[CL]:
                        temp = FR[CL][func]
                        if "cost" in temp.keys() and "memory" in temp.keys() \
                            and "idle_time_before_kill" in temp.keys():
                            new_f = FaaS(CL, func, float(temp["cost"]), 
                                         float(temp["memory"]), FunctionPE(),
                                         transition_cost, 
                                         float(temp["idle_time_before_kill"]))
                            self.resources.append(new_f)
                            self.dic_map_res_idx[func]=resource_idx
                            cl.add_resource(resource_idx)
                            resource_idx += 1
                        else:
                            self.logger.log("ERROR: missing field in {} description".\
                                            format(func), 1)
                            sys.exit(1)
                        # add the resource description to the corresponding 
                        # dictionary
                        if "description" in temp.keys():
                            self.description[func] = temp["description"]
                        else:
                            self.description[func] = "No description"
                    # add the new computational layer
                    self.CLs.append(cl)
        else:
                self.FaaS_start_index=float("inf")
        # restore indentation level for logging
        self.logger.level -= 1
    
    
    ## Method to convert the dictionary of global constraints to a list
    # @param self The object pointer   
    # @param GC Dictionary of global constraints
    def convert_GCdic_to_list(self, GC):
        self.global_constraints = []
        # loop over paths
        for p in GC:
            C_list = []
            # loop over components in the path
            for c in GC[p]["components"]:
                if c in self.dic_map_com_idx.keys():
                    C_list.append(list(self.dic_map_com_idx.keys()).index(c))
                else:
                    self.error.log("ERROR: no match between components and path in global constraints", 1)
                    sys.exit(1)
            self.global_constraints.append(GlobalConstraint(C_list, 
                                                            GC[p]["global_res_time"],
                                                            p))
    
    
    ## Method to convert the compatibility and demand dictionaries into 
    # lists of 2D numpy arrays such that M[i][h][j] represents either the 
    # compatibility of partition h in component i with resource j or the 
    # demand to run such partition on the given resource
    #    @param self The object pointer
    def convert_dic_to_matrix(self):
        self.compatibility_matrix = []
        self.demand_matrix = []
        comp_idx = 0
        # loop over components
        for comp in self.components:
            # loop over candidate deployments and count the total number of 
            # partitions
            p = 0
            for dep in comp.deployments:
                p += len(dep.partitions)
            # define and initialize the matrices to zero
            self.compatibility_matrix.append(np.full((p, len(self.resources)), 
                                                     0, dtype = int))
            self.demand_matrix.append(np.full((p, len(self.resources)), 
                                              0, dtype=float))
        # loop over components to assign the values to the matrices
        for c in self.compatibility_dict:
            # loop over partitions
            for part in self.compatibility_dict[c]: 
                # loop over resources
                for res in self.compatibility_dict[c][part]:
                    comp_idx = self.dic_map_part_idx[c][part][0]
                    part_idx = self.dic_map_part_idx[c][part][1]
                    self.compatibility_matrix[comp_idx][part_idx][self.dic_map_res_idx[res]] = 1
                    # for Edge and Cloud resources, the demand is taken 
                    # directly from the dictionary
                    if self.dic_map_res_idx[res] < self.FaaS_start_index:
                        self.demand_matrix[comp_idx][part_idx][self.dic_map_res_idx[res]] = self.demand_dict[c][part][res]
                    # for FaaS resources, it should be computed accordingly
                    else:
                        arrival_rate = self.components[self.dic_map_com_idx[c]].comp_Lambda
                        warm_service_time = self.demand_dict[c][part][res][0]
                        cold_service_time = self.demand_dict[c][part][res][1]
                        self.demand_matrix[comp_idx][part_idx][self.dic_map_res_idx[res]]  = self.resources[self.dic_map_res_idx[res]].get_avg_res_time(
                            arrival_rate, 
                            warm_service_time, 
                            cold_service_time)
    
    
    ## Method to convert the system description into a json object
    #   @param self The object pointer
    #   @return Json object storing the system description
    def to_json(self):
        
        # components
        system_string = '{"Components": {'
        for c in self.components:
            system_string += (str(c) + ',')
        system_string = system_string[:-1] + '}'
        
        # resources
        system_string += ', '
        edge_string = '"EdgeResources": {'
        cloud_string = '"CloudResources": {'
        faas_string = '"FaaSResources": {'
        for CL in self.CLs:
            last_idx = CL.resources[-1]
            if last_idx < self.cloud_start_index:
                edge_string += (CL.__str__(self.resources) + ',')
            elif last_idx < self.FaaS_start_index:
                cloud_string += (CL.__str__(self.resources) + ',')
            else:
                faas_string += (CL.__str__(self.resources) + ',')
        system_string += (edge_string[:-1] + '}, \n' + \
                          cloud_string[:-1] + '}, \n' + \
                          faas_string[:-1] + '}')
                
        # compatibility matrix
        system_string += (', \n"CompatibilityMatrix": ' + \
                          str(self.compatibility_dict).replace("\'", "\""))
        
        # demand matrix
        system_string += (', \n"DemandMatrix": ' + \
                          str(self.demand_dict).replace("\'", "\""))
        
        # lambda
        system_string += (', \n"Lambda": ' + str(self.Lambda))
        
        # local constraints
        
        # global constraints
        
        # network technology
        system_string += ', \n"NetworkTechnology": {'
        for d in self.network_technologies:
            system_string += (str(d) + ',')
        system_string = system_string[:-1] + '}'
        
        # DAG
        system_string += (', \n"DirectedAcyclicGraph": {' + \
                          str(self.graph) + '}')
                          
        system_string += (', \n"Time": ' + str(self.T) + '}')
        
        # load string as json
        jj = json.dumps(json.loads(system_string), indent = 2)
        
        return jj

        
    ## Method to print the system description (in json format), either on 
    # stdout or onto a given file
    #   @param self The object pointer
    #   @param system_file File where to print the system description (optional)
    def print_system(self, system_file = ""):
        
        # get system description in json format
        jj = self.to_json()
        
        # print
        if system_file:
            with open(system_file, "w") as f:
                f.write(jj)
        else:
            print(jj)


    ## Method to print the graph onto a gml file (see Graph.DAG.write_DAG)
    #   @param self The object pointer
    #   @param graph_file File where to print the graph (gml format)
    def print_graph(self, graph_file):
        self.graph.write_DAG(graph_file)


    ## Method to plot the graph (see Graph.DAG.plot_DAG)
    #   @param self The object pointer
    #   @param plot_file File where to plot the graph (optional)
    def plot_graph(self, plot_file = ""):
        self.graph.plot_DAG(plot_file)


