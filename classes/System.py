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
import copy   

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
    def __init__(self, system_file):
        
        # build graph from the corresponding file
        #self.graph = DAG(graph_file)
        
        
        # read configuration file to store information about components,
        # resources and compatibility/demand matrices
       
        self.read_configuration_file(system_file)
       
      
      
        
    
    ## Method to read a configuration file providing the system description 
    # (in json format) and populate the class members accordingly
    #   @param self The object pointer
    #   @param system_file Configuration file describing the system (json format)
    def read_configuration_file(self, system_file):
        self.description={}
        # load json file
        with open(system_file) as f:
            data = json.load(f)
        
        if "DirectedAcyclicGraph" in data.keys():
            DAG_dict=data["DirectedAcyclicGraph"]
            self.graph = DAG(DAG_dict)
            #self.create_datasize_matrix(data_transfer_dict)
            
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
        # edge resources
        self.dic_map_res_idx={}
        resource_idx=0
        self.CLs=[]
        self.resources = []
       
        if "EdgeResources" in data.keys():
            ER = data["EdgeResources"]
           
            for CL in ER:
                cl=ComputationalLayer(CL)
                # loop over nodes and add them to the corresponding layer
                for node in ER[CL]:
                  if "number" in ER[CL][node].keys() and \
                      "cost" in ER[CL][node].keys() and "memory" in ER[CL][node].keys():
                        new_node = EdgeNode(CL, node, float(ER[CL][node]["cost"]),
                                            float(ER[CL][node]["memory"]),EdgePE(),
                                            int(ER[CL][node]["number"]))
                        self.resources.append(new_node)
                        self.dic_map_res_idx[node]=resource_idx
                        cl.add_resource(resource_idx)
                        resource_idx+=1
                 
                  else:
                        print("ERROR: missing field in ", node, " description")
                        sys.exit(1)
                  if "description" in ER[CL][node].keys():
                        self.description[node]=ER[CL][node]["description"]
                  else:
                        self.description[node]="No description"
                self.CLs.append(cl)
                
        #
        # cloud resources
        if "CloudResources" in data.keys():
            self.cloud_start_index=resource_idx
           
            CR = data["CloudResources"]
            
            for CL in CR:
                cl=ComputationalLayer(CL)
                # loop over VMs and add them to the corresponding layer
                for VM in CR[CL]:
                    if "number" in CR[CL][VM].keys() and \
                        "cost" in CR[CL][VM].keys() and \
                            "memory" in CR[CL][VM].keys():
                        new_vm = VirtualMachine(CL, VM, float(CR[CL][VM]["cost"]), 
                                                float(CR[CL][VM]["memory"]), ServerFarmPE(),
                                                int(CR[CL][VM]["number"]))
                        self.resources.append(new_vm)
                        self.dic_map_res_idx[VM]=resource_idx
                        cl.add_resource(resource_idx)
                        resource_idx+=1
                    else:
                        print("ERROR: missing field in ", VM, "  description")
                        sys.exit(1)
                    if "description" in CR[CL][VM].keys():
                        self.description[VM]=CR[CL][VM]["description"]
                    else:
                        self.description[VM]="No description"
                self.CLs.append(cl)
     #
        
        # faas resources
       
        if "FaaSResources" in data.keys():
            FR = data["FaaSResources"]
            self.FaaS_start_index=resource_idx
            for CL in FR:
               
                if "transition_cost" in FR.keys():
                        transition_cost = float(FR["transition_cost"])
                else:
                        print("ERROR: missing field in FaaSResources description")
                        sys.exit(1)
                if CL.startswith("computationallayer"):
                    cl=ComputationalLayer(CL)
                    # read transition cost
                    
                    
                    # loop over functions and add them to the corresponding layer
                    for func in FR[CL]:
                        if func != "transition_cost":
                            if "cost" in FR[CL][func].keys() and \
                                "memory" in FR[CL][func].keys() and \
                                 "idle_time_before_kill" in FR[CL][func].keys():
                                new_f = FaaS(CL, func, float(FR[CL][func]["cost"]), 
                                             float(FR[CL][func]["memory"]), FunctionPE(),
                                             transition_cost, 
                                             float(FR[CL][func]["idle_time_before_kill"]))
                                self.resources.append(new_f)
                                self.dic_map_res_idx[func]=resource_idx
                                cl.add_resource(resource_idx)
                                resource_idx+=1
                            else:
                                print("ERROR: missing field in ",func," description")
                                sys.exit(1)
                            if "description" in FR[CL][func].keys():
                                self.description[func]=FR[CL][func]["description"]
                            else:
                                self.description[func]="No description"
                    self.CLs.append(cl)
        
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
         
         
        
            
        if "Time" in data.keys():
            self.T=float(data["Time"])
       
##############################################################################################

    # ## Method creats a square matrix to show the data transfer between each two components
    # # @param self The object pointer
    # def create_datasize_matrix(self, data_transfer_dict):
    #     self.data_sizes=np.full((len(self.components), len(self.components)), 0, dtype=float)
    #     for c in data_transfer_dict:
                
    #             if "next" in data_transfer_dict[c] and \
    #                 "data_size" in data_transfer_dict[c].keys():
    #                 comps=list(data_transfer_dict[c]["next"])
    #                 sizes=list(data_transfer_dict[c]["data_size"])
    #                 if len(comps)==len(sizes):
    #                     for component in comps:
    #                         if component in self.dic_map_com_idx and \
    #                             c in self.dic_map_com_idx:
    #                            x1= list(self.dic_map_com_idx.keys()).index(c)
    #                            x2=list(self.dic_map_com_idx.keys()).index(component)
    #                            self.data_sizes[x1][x2]=sizes[data_transfer_dict[c]["next"].index(component)]
    #                         else:
    #                            print("ERROR: no match between components dict and data transfer dict")
    #                            sys.exit(1) 
    #                 else:
    #                      print("ERROR: no match between components list and data size list")
    #                      sys.exit(1)
    
    
    
   
   
    ## Method initialize the components based on the dictionary of component 
    # extracted from config file and graph, compute the input lambda 
    # (workload) of each component. The input graph can be a general case 
    # with some parallel branches.
    # @param self The object pointer
    # @param C Dictionary of components came from configuration file.
    # @param LC Dictionary of LocalConstraints came from configuration file.
    def initialize_components(self, C,LC):
        
        self.components = []
        self.dic_map_com_idx={}
        self.dic_map_part_idx={}
        localconstraints={}
        comp_idx=0
      
        for c in C :
                
            # check if the components' names in Components and LocalConstraints are matched
            
                
                # check if the dict component of input file is in the graph
                if self.graph.G.has_node(c):
                    # check if the node c has any input edge
                    if len(C[c])>0:
                            deployments=[]
                            part_idx=0
                            temp={}
                    if self.graph.G.in_edges(c):
                        Sum = 0
                        # if the node c has some input edges, its Lambda is equal 
                        # to the sum of products of lambda and weight of its 
                        # input edges.
                        
                        for n, c, data in self.graph.G.in_edges(c, data=True):
                            Sum += float(data["transition_probability"]) * self.components[self.dic_map_com_idx[n]].comp_Lambda
                        
                       
                            
                            for s in C[c]:
                                part_Lambda=-1
                                if  len(C[c][s])>0:
                                    partitions=[]
                                    for h in C[c][s]:
                                        if part_Lambda>-1:
                                            part_Lambda=part_Lambda*(1-float(C[c][s][h]["stop_probability"]))
                                        else:
                                            part_Lambda=copy.deepcopy(Sum)
                                        temp[h]=(comp_idx,part_idx)
                                        partitions.append(Component.Deployment.Partition(h,float(C[c][s][h]["memory"]),part_Lambda,
                                                                                         float(C[c][s][h]["stop_probability"]),
                                                                                         C[c][s][h]["next"],float(C[c][s][h]["data_size"])))
                                        part_idx+=1
                                deployments.append(Component.Deployment(s,partitions))    
                            self.dic_map_part_idx[c]=temp
                        comp=Component(c,deployments,Sum)
                        
                        self.components.append(comp)
                        
                    else:
                        # if the node c does not have any input edge, it is the 
                        # first node of a path and its Lambda is equal to input 
                        # lambda.
                         
                         
                            
                            for s in C[c]:
                                part_Lambda=-1
                                if  len(C[c][s])>0:
                                    partitions=[]
                                    for h in C[c][s]:
                                        if part_Lambda>-1:
                                            part_Lambda=part_Lambda*(1-float(C[c][s][h]["stop_probability"]))
                                        else:
                                            part_Lambda=copy.deepcopy(self.Lambda)
                                        temp[h]=(comp_idx,part_idx)
                                        partitions.append(Component.Deployment.Partition(h,float(C[c][s][h]["memory"]),part_Lambda,
                                                                                         float(C[c][s][h]["stop_probability"]),
                                                                                         C[c][s][h]["next"],float(C[c][s][h]["data_size"])))
                                        part_idx+=1
                                        
                                deployments.append(Component.Deployment(s,partitions))    
                            self.dic_map_part_idx[c]=temp
                       
                            self.components.append(Component(c,deployments, self.Lambda))
                else:
                    print("ERROR: no match between components in DAG and system input file")
                    sys.exit(1)
                self.dic_map_com_idx[c]=comp_idx
                        
                if c  in LC:
                    localconstraints[c]=LocalConstraint(self.components[self.dic_map_com_idx[c]],
                                                                 float(LC[c]["local_res_time"]))
                comp_idx+=1
                    
        # create a list of local constraints, each row belong to a component 
        # and each component has max_res_time
        
        self.LC=np.full((len(localconstraints),2), 0, dtype=float)
        LC_idx=0
        for c in localconstraints:
            if c in self.dic_map_com_idx:
                self.LC[LC_idx][0]=self.dic_map_com_idx[c]
                self.LC[LC_idx][1]=localconstraints[c].max_res_time
                LC_idx+=1
            else:
                print("ERROR: no match between current component in local constraint and components in the system ")
                sys.exit(1)
        # creat a list of components memory
        #self.component_memory=np.array(list(c.memory for c in list(self.components.values())))
       
    ## Method to convert dictionaries of global constraints to a numpy array  
    # @param self The object pointer   
    # @param GC dict of global constraint  
    def convert_GCdic_to_list(self,GC):
        
        self.GC=[]
        
        for p in GC:
            C_list=[]
            for c in GC[p]["components"]:
                if c in self.dic_map_com_idx.keys():
                    C_list.append(list(self.dic_map_com_idx.keys()).index(c))
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
        
        self.compatibility_matrix=[]
        self.demand_matrix=[]
        comp_idx=0
        for comp in self.components:
            p=0
            for dep in comp.deployments:
                p+=len(dep.partitions)
            self.compatibility_matrix.append( np.full((p, len(self.resources)), 
                                            0, dtype=int))
        # define and initialize the matrices to zero
        
            self.demand_matrix.append( np.full((p, len(self.resources)), 
                                     0, dtype=float))
        
       

      
        # Loop over components and resources to compute the integer matrix 
        # from the dictionary
        
        for c in self.compatibility_dict:
           
            for part in self.compatibility_dict[c]: 
                for res in self.compatibility_dict[c][part]:
                    comp_idx=self.dic_map_part_idx[c][part][0]
                    part_idx=self.dic_map_part_idx[c][part][1]
                    self.compatibility_matrix[comp_idx][part_idx][self.dic_map_res_idx[res]] = 1
                    if self.dic_map_res_idx[res] < self.FaaS_start_index:
                        self.demand_matrix[comp_idx][part_idx][self.dic_map_res_idx[res]] = self.demand_dict[c][part][res]
                    else:
                        
                        arrival_rate = self.components[self.dic_map_com_idx[c]].comp_Lambda
                        warm_service_time=self.demand_dict[c][part][res][0]
                        cold_service_time=self.demand_dict[c][part][res][1]
                        self.demand_matrix[comp_idx][part_idx][self.dic_map_res_idx[res]]  = self.resources[self.dic_map_res_idx[res]].get_avg_res_time(
                            arrival_rate, warm_service_time, cold_service_time)

               
      
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
    
    #def create_dic_map_res_com():
        
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
    def print_graph(self ):
        self.graph.write_DAG()


    ## Method to plot the graph (see Graph.DAG.plot_DAG)
    #   @param self The object pointer
    #   @param plot_file File where to plot the graph (optional)
    def plot_graph(self, plot_file = ""):
        self.graph.plot_DAG(plot_file)

    