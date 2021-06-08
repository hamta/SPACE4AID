from classes.Graph import DAG, Component
from classes.Resources import ComputationalLayer, VirtualMachine, EdgeNode, FaaS
from classes.NetworkTechnology import NetworkDomain
from classes.Performance import NetworkPE, ServerFarmPE, FunctionPE, EdgePE
from classes.Constraints import LocalConstraint, GlobalConstraint
from classes.Solution import Configuration
import json
import sys
import numpy as np
import pdb    

## System
#
# Class to store the system description with all the relevant information
class System:

    ## @var graph 
    # Object of Graph.DAG type
    
    ## @var components 
    # Dictionary of the available Resources.Component objects (indexed by name)
    
    ## @var resources 
    # Dictionary of Resources.ComputationalLayer objects (indexed by type)
    
    ## @var compatibility_dict
    # Dictionary representing the compatibility matrix of the available 
    # components (i.e., for each Graph.Component object, the list of 
    # Resources.Resource objects where it can be deployed)
    
    ## @var compatibility_matrix 
    # compatibility matrix with 2-dimensional 0-1 array equal to Dictionary 
    # compatibility matrix in which each row denotes a component and each 
    # column denotes a resource with integer indexes
            
    ## @var demand_dict 
    # Dictionary representing the demand matrix of the available components 
    # (i.e., the demand of each Graph.Component object when it is deployed on 
    # the different Resources.Resource objects)
    
    ## @var demand_matrix 
    # Demand matrix with 2-dimensional integer array equal to Dictionary 
    # demand matrix in which each row denotes a component and each column 
    # denotes a resource with integer indexes
    
    ## @var performance_constraints
    # TODO
    
    ## @var network
    # NetworkTechnology object, characterized by a given access delay and 
    # bandwidth

    ## @var Lambda
    # Incoming load
  
    ## @var Max_res_time
    # Maximum response time of application

    ## System class constructor
    #   @param self The object pointer
    #   @param system_file Configuration file describing the system
    #   @param graph_file File describing the graph (gml or text format)
    #   @param performance_constraints
    #   @param performance_evaluators
    #   @param networkTechnology
    #   @param initial_solution_file includes initial random decision variable
    def __init__(self, system_file, graph_file):
        
        # build graph from the corresponding file
        self.graph = DAG(graph_file)
        
        
        # read configuration file to store information about components,
        # resources and compatibility/demand matrices
       
        self.read_configuration_file(system_file)
       
        self.Lambdas=np.array(list(c.Lambda for c in list(self.components.values()))).T
                
        # read random initial solution file to initialize decesion variables
        #self.read_random_initial_solution_file(initial_solution_file)
        
               
        # creat a list of resource with corresponding computational layer
        self.resources_CL=self.creat_resource_CL_list()
        
        self.CL_resources=self.create_LC_resources_Dict()
        # creat a list of computational layer with corresponding Network domain
        self.CL_NDs=self.creat_CL_ND_list()
        
        # initialize performance evaluators
        self.performance_evaluators = None  # TODO: where do we save this?
        
        
    
    ## Method to read a configuration file providing the system description 
    # (in json format) and populate the class members accordingly
    #   @param self The object pointer
    #   @param system_file Configuration file describing the system (json format)
    def read_configuration_file(self, system_file):
        
        # load json file
        with open(system_file) as f:
            data = json.load(f)
        
        # initialize lambda
        if "Lambda" in data.keys():
            self.Lambda = float(data["Lambda"])
        else:
            print("ERROR: no Lambda available in configuration file")
            sys.exit(1)
               
        # initialize NetworkTechnology object
       
          
          
        # initialize dictionary of components and local constraints
        if "Components" in data.keys():
            if "LocalConstraints" in data.keys():
              
                    C = data["Components"]
                    LC = data["LocalConstraints"]
                    
                    self.initialize_components(C,LC)
                
            else:
                print("ERROR: no LocalConstraints available in configuration file")
                sys.exit(1)
        else:
            print("ERROR: no components available in configuration file")
            sys.exit(1)
        
        # get Global Constraints to initialize Maximun response time of application
        if "GlobalConstraints" in data.keys():
            GC = data["GlobalConstraints"]
            self.convert_GCdic_to_list(GC)
        
        # initialize dictionary of resources
        #
        # cloud resources
        if "CloudResources" in data.keys():
            
            self.resources = {}
            CR = data["CloudResources"]
            self.resources["CloudResources"] = ComputationalLayer(ServerFarmPE())
            for CL in CR:
                
                # loop over VMs and add them to the corresponding layer
                for VM in CR[CL]:
                    if "number" in CR[CL][VM].keys() and \
                        "cost" in CR[CL][VM].keys() and \
                            "memory" in CR[CL][VM].keys():
                        new_vm = VirtualMachine(CL, VM, float(CR[CL][VM]["cost"]), 
                                                float(CR[CL][VM]["memory"]),
                                                int(CR[CL][VM]["number"]))
                        self.resources["CloudResources"].add_resource(new_vm)
                    else:
                        print("ERROR: missing field in ", VM, "  description")
                        sys.exit(1)
     #
        # edge resources
        
        if "EdgeResources" in data.keys():
            ER = data["EdgeResources"]
            self.resources["EdgeResources"] = ComputationalLayer(EdgePE())
            for CL in ER:
                
                # loop over nodes and add them to the corresponding layer
                for node in ER[CL]:
                  if "number" in ER[CL][node].keys() and \
                      "cost" in ER[CL][node].keys() and "memory" in ER[CL][node].keys():
                        new_node = EdgeNode(CL, node, float(ER[CL][node]["cost"]),
                                            float(ER[CL][node]["memory"]),
                                            int(ER[CL][node]["number"]))
                        self.resources["EdgeResources"].add_resource(new_node)
                  else:
                        print("ERROR: missing field in ", node, " description")
                        sys.exit(1)
        #
        # faas resources
       
        if "FaaSResources" in data.keys():
            FR = data["FaaSResources"]
            self.resources["FaaSResources"] = ComputationalLayer(FunctionPE())
            for CL in FR:
                if "transition_cost" in FR.keys():
                        transition_cost = float(FR["transition_cost"])
                else:
                        print("ERROR: missing field in FaaSResources description")
                        sys.exit(1)
                if CL.startswith("computationallayer"):
                   
                    # read transition cost
                    
                    
                    # loop over functions and add them to the corresponding layer
                    for func in FR[CL]:
                        if func != "transition_cost":
                            if "cost" in FR[CL][func].keys() and \
                                "memory" in FR[CL][func].keys() and \
                                 "idle_time_before_kill" in FR[CL][func].keys():
                                new_f = FaaS(CL, func, float(FR[CL][func]["cost"]), 
                                             float(FR[CL][func]["memory"]),
                                             transition_cost, 
                                             float(FR[CL][func]["idle_time_before_kill"]))
                                self.resources["FaaSResources"].add_resource(new_f)
                            else:
                                print("ERROR: missing field in ",func," description")
                                sys.exit(1)
        
       # load Network Technology
        if "NetworkTechnology" in data.keys():
            self.network_technologies=[]
            NT=data["NetworkTechnology"]
             # load Network Domains
            for ND in NT:
                if "computationallayers" in NT[ND].keys() and \
                      "AccessDelay" in NT[ND].keys() and "Bandwidth" in NT[ND].keys():
                        network_domain= NetworkDomain(ND,list(NT[ND]["computationallayers"]),float(NT[ND]["AccessDelay"]),
                                            float(NT[ND]["Bandwidth"]),NetworkPE())
                        self.network_technologies.append(network_domain)
                        
                else:
                        print("ERROR: missing field in ", node, " description")
                        sys.exit(1)
        else:
            print("ERROR: no Network Technology available in configuration file")
            sys.exit(1)
       
       
        
       # load compatibility matrix
        if "CompatibilityMatrix" in data.keys():
            self.compatibility_dict = data["CompatibilityMatrix"]
           
        else:
            print("ERROR: no compatibility matrix available in configuration file")
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
                        print("ERROR: demand and compatibility matrix are not consistent")
                        sys.exit(1)    
        else:
            is_compatible = False
            print("ERROR: no demand matrix available in configuration file")
            sys.exit(1)
        
        # if demand matrix is available and it is consistent with 
        # compatibility matrix, convert both these dict to arrays
        if is_compatible:
            self.convert_dic_to_matrix()
         
          
        if "DataTransfer" in data.keys():
            self.data_transfer_dict=data["DataTransfer"]
            self.create_datasize_matrix()
            
        if "Time" in data.keys():
            self.T=float(data["Time"])
##############################################################################################

    ## Method creats a square matrix to show the data transfer between each two components
    # @param self The object pointer
    def create_datasize_matrix(self):
        self.data_sizes=np.full((len(self.components), len(self.components)), 0, dtype=float)
        for c in self.data_transfer_dict:
                
                if "next" in self.data_transfer_dict[c] and \
                    "data_size" in self.data_transfer_dict[c].keys():
                    comps=list(self.data_transfer_dict[c]["next"])
                    sizes=list(self.data_transfer_dict[c]["data_size"])
                    if len(comps)==len(sizes):
                        for component in comps:
                            if component in self.components and \
                                c in self.components:
                               x1= list(self.components.keys()).index(c)
                               x2=list(self.components.keys()).index(component)
                               self.data_sizes[x1][x2]=sizes[self.data_transfer_dict[c]["next"].index(component)]
                            else:
                               print("ERROR: no match between components dict and data transfer dict")
                               sys.exit(1) 
                    else:
                         print("ERROR: no match between components list and data size list")
                         sys.exit(1)
    
    ## Method creates a list of tuple to show each resource belong to which computational layer
    # @param self The object pointer
    def creat_resource_CL_list(self):
        
        resource_CL_list=[]
        resource_idx=0
        for res in self.resources["EdgeResources"].resources:
            resource_CL_list.append((resource_idx,res.CLname))
            resource_idx+=1
        for res in self.resources["CloudResources"].resources:
            resource_CL_list.append((resource_idx,res.CLname))
            resource_idx+=1
        for res in self.resources["FaaSResources"].resources:
            resource_CL_list.append((resource_idx,res.CLname))
            resource_idx+=1
        return resource_CL_list
    
    ## Method creates a list of tuple to show each computational layer belong to which Network domains
    # @param self The object pointer
    def creat_CL_ND_list(self):
        
        CL_NDs=[]
        # get a list of all computational layers
        computational_layers=list(sorted((set(list(c[1] for c in self.resources_CL)))))
        # for each computational layer determine that the current layer is located in which network domains
        for layer in computational_layers:
            li=list(filter(lambda network_technology: (layer in network_technology.computationallayers)
                           , self.network_technologies))
            NDs=[]
            for l in li:
                NDs.append(l.ND_name)
            CL_NDs.append((layer,NDs))
       
        return CL_NDs
   
    ## Method initialize the components based on the dictionary of component 
    # extracted from config file and graph, compute the input lambda 
    # (workload) of each component. The input graph can be a general case 
    # with some parallel branches.
    # @param self The object pointer
    # @param C Dictionary of components came from configuration file.
    # @param LC Dictionary of LocalConstraints came from configuration file.
    def initialize_components(self, C,LC):
        
        self.components = {}
        localconstraints={}
        for c in C :
            # check if the components' names in Components and LocalConstraints are matched
            
                
                # check if the dict component of input file is in the graph
                if self.graph.G.has_node(c):
                    # check if the node c has any input edge
                    if self.graph.G.in_edges(c):
                        Sum = 0
                        # if the node c has some input edges, its Lambda is equal 
                        # to the sum of products of lambda and weight of its 
                        # input edges.
                        for n, c, data in self.graph.G.in_edges(c, data=True):
                            Sum += float(data["weight"]) * self.components[n].Lambda
                        self.components[c] = Component(c, float(C[c]["memory"]),
                                                      Sum)
                    else:
                        # if the node c does not have any input edge, it is the 
                        # first node of a path and its Lambda is equal to input 
                        # lambda.
                        self.components[c] = Component(c, float(C[c]["memory"]), 
                                                      self.Lambda)
                        
                if c  in LC:
                    localconstraints[c]=LocalConstraint(self.components[c],
                                                                 float(LC[c]["local_res_time"]))
                    
        # create a list of local constraints, each row belong to a component 
        # and each component has max_res_time
        
        self.LC=np.full((len(localconstraints),2), 0, dtype=float)
        LC_idx=0
        for c in localconstraints:
            if c in self.components:
                self.LC[LC_idx][0]=list(self.components.keys()).index(c)
                self.LC[LC_idx][1]=localconstraints[c].max_res_time
                LC_idx+=1
            else:
                print("ERROR: no match between current component in local constraint and components in the system ")
                sys.exit(1)
        # creat a list of components memory
        self.component_memory=np.array(list(c.memory for c in list(self.components.values())))
        
    ## Method to convert dictionaries of global constraints to a numpy array  
    # @param self The object pointer   
    # @param GC dict of global constraint  
    def convert_GCdic_to_list(self,GC):
        
        self.GC=[]
        
        for p in GC:
            C_list=[]
            for c in GC[p]["components"]:
                if c in self.components.keys():
                    C_list.append(list(self.components.keys()).index(c))
                else:
                    print("ERROR: no match between components and path in global constraints")
                    sys.exit(1)
           
            self.GC.append((C_list, GC[p]["global_res_time"]))
            
    
          
    
            
    ## Method to convert dictionaries of demand and compatibility to 
    # 2_D arrays of int in which each row denotes a component and each column 
    # this method creat a matrix of resource memory and resource cost
    # denotes a resource with integer indexes
    # @param self The object pointer
    def convert_dic_to_matrix(self):
      
        # obtain the number of type of resources
        resource_count = len(self.resources["CloudResources"].resources) + \
                         len(self.resources["EdgeResources"].resources) + \
                         len(self.resources["FaaSResources"].resources)
        
        # determin that there are how many devices for each type of resources
        # compute for edge and cloud, for function, it is unlimited 
        edge_cloud_count=resource_count-len(self.resources["FaaSResources"].resources)
        self.resource_number = np.full(edge_cloud_count, 
                                            1, dtype=int)
        
        #obtain the number of components 
        component_count = len(self.compatibility_dict)
        
        # define and initialize the matrices to zero
        self.compatibility_matrix = np.full((component_count, resource_count), 
                                            0, dtype=int)
        self.demand_matrix = np.full((component_count, resource_count), 
                                     0, dtype=float)
        
        # Initialize resource memory matrix with infinitive value, 
        # FaaS resources will have "inf" value
        self.resource_memory = np.full(resource_count, 
                                            float("inf"), dtype=float)
        # Initialize resource cost with zero
        self.resource_cost = np.full(resource_count, 
                                            0, dtype=float)
       
       
        
        component_index = 0
        edge_seen=False
        cloud_seen=False
        FaaS_seen=False
      
        # Loop over components and resources to compute the integer matrix 
        # from the dictionary
        for c in self.compatibility_dict:
            resource_index = 0
           
            # edge resources
            for E in self.resources["EdgeResources"].resources:
                # check if the current resource is compatible with the 
                # current component
                if E.name in self.compatibility_dict[c]:
                    self.compatibility_matrix[component_index][resource_index] = 1
                    self.demand_matrix[component_index][resource_index] = self.demand_dict[c][E.name]
                
                if not edge_seen:
                    self.resource_number[resource_index]=E.number
                    self.resource_memory[resource_index]=E.memory
                    self.resource_cost[resource_index]=E.cost
                    
                
                resource_index += 1
            edge_seen=True        
            
                  
            # cloud resources
            if not cloud_seen:
                self.cloud_start_index=resource_index
            for C in self.resources["CloudResources"].resources:
                # check if the current resource is compatible with the 
                # current component
                if C.name in self.compatibility_dict[c]:
                    self.compatibility_matrix[component_index][resource_index] = 1
                    self.demand_matrix[component_index][resource_index] = self.demand_dict[c][C.name]
                
                
                if not cloud_seen:
                    self.resource_number[resource_index]=C.number
                    self.resource_memory[resource_index]=C.memory
                    self.resource_cost[resource_index]=C.cost
                    
                resource_index += 1
            cloud_seen=True        

            # faas resources
            
            if not FaaS_seen:
                self.FaaS_start_index=resource_index
            for F in self.resources["FaaSResources"].resources:   
                # check if the current resource is compatible with the 
                # current component
                if F.name in self.compatibility_dict[c]:
                    arrival_rate = self.components[c].Lambda
                    self.compatibility_matrix[component_index][resource_index] = 1
                 
                    
                    warm_service_time=self.demand_dict[c][F.name][0]
                    cold_service_time=self.demand_dict[c][F.name][1]
                    self.demand_matrix[component_index][resource_index] = F.get_avg_res_time(arrival_rate, warm_service_time, cold_service_time)
                  
                if not FaaS_seen:
                    self.resource_cost[resource_index]=F.cost
                    self.FaaS_trans_cost = F.transition_cost  
                resource_index += 1
            FaaS_seen=True
            component_index += 1
       
        
    
      
    ## Method to get the system description as json object
    #   @param self The object pointer
    #   @return Json object storing the system description
    def to_json(self):
        
        # components
        system_string = '{"Components": {'
        for c in self.components:
            system_string += (str(self.components[c]) + ',')
        system_string = system_string[:-1] + '}'
        
        # resources
        system_string += ', '
        for resource_type in self.resources:
            system_string += ('\n"' + resource_type + '":' + 
                              str(self.resources[resource_type]) + ",")
        system_string = system_string[:-1]
                
        # compatibility matrix
        system_string += (', \n"CompatibilityMatrix": ' + 
                          str(self.compatibility_dict).replace("\'", "\""))
        
        # demand matrix
        system_string += (', \n"DemandMatrix": ' + 
                          str(self.demand_dict).replace("\'", "\""))
        
        # lambda, access delay, bandwidth and maximum response time
        system_string += (', \n"Lambda": ' + str(self.Lambda))
        system_string += (', ' + str(self.network))
        system_string += (', \n"Max_res_time": ' + str(self.Max_res_time) + '}')
        
        # load string as json
        jj = json.dumps(json.loads(system_string), indent = 2)
        
        return jj
        
    def create_LC_resources_Dict(self):
        Output = {}
        for x, y in self.resources_CL:
            if y in Output:
                Output[y].append(x)
            else:
                Output[y] = [x]
        return Output  
        
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

    