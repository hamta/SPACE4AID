from classes.Logger import Logger
import networkx as nx
import matplotlib.pyplot as plt
import sys
import pdb


## DAG
#
# Class to represent a Directed Acyclic Graph. The nodes represent the names 
# of existing components, while edges represent precedence relations between 
# components. Each edge is characterized by a number in [0, 1], representing 
# the probabiliy of moving between two adjacent nodes
class DAG:

    ## @var G
    # Object of type nx.DiGraph used to represent the directed acyclic graph
    
    ## @var logger
    # Object of Logger class, used to print general messages
    
    ## @var error
    # Object of Logger class, used to print error messages on sys.stderr

    ## DAG class constructor
    #   @param self The object pointer
    #   @param graph_file File describing the graph (gml or text format)
    #   @param graph_dict Dictionary describing the graph
    #   @param log Object of Logger class
    def __init__(self, graph_file = "", graph_dict = None, log = Logger()):
        self.logger = log
        self.error = Logger(stream=sys.stderr, verbose=1)
        if graph_file != "":
            self.logger.log("Loading DAG from file", 2)
            self.read_DAG_file(graph_file)
        elif graph_dict:
            self.logger.log("Loading DAG dictionary", 2)
            self.read_DAG(graph_dict)
        else:
            self.error.log("ERROR: no DAG provided", 1)
            sys.exit(1)
          
    
    ## Method to generate a DAG object starting from the description provided
    # in a text file (which should be in gml or text format).
    #   @param self The object pointer
    #   @param graph_file File with the graph description (gml or text format)
    def read_DAG_file(self, graph_file):        
        
        # read gml file
        if graph_file.endswith(".gml"):
            self.G = nx.read_gml(graph_file)
        # read text file
        else:
            with open(graph_file) as f:
                # read file lines
                lines = f.read().splitlines()
                # initialize directed graph
                self.G = nx.DiGraph()
                # populate graph
                for line in lines:
                    tokens = line.split(" ")
                    n1 = tokens[0]
                    n2 = tokens[1]
                    probability = float(tokens[2])
                    if len(tokens) > 3:
                        data_size = float(tokens[3])
                    else:
                        data_size = 0.0
                    if n1 not in self.G:
                        self.G.add_node(n1)
                    if n2 not in self.G:
                        self.G.add_node(n2)
                    self.G.add_edge(n1, n2, 
                                    transition_probability=probability, 
                                    data_size=data_size)
    
    
    ## Method to generate a DAG object starting from the description provided
    # in the given dictionary
    #   @param self The object pointer
    #   @param graph_dict Dictionary with the graph description
    def read_DAG(self, graph_dict):
        
        # initialize graph
        self.G = nx.DiGraph()
        
        # loop over components
        for c in graph_dict:
            # get the list of successors
            if "next" in graph_dict[c]:
                comps = list(graph_dict[c]["next"])
                if len(comps)>0:
                #sizes = list(graph_dict[c]["data_size"])
                    probabilities = list(graph_dict[c]["transition_probability"])
                    if len(probabilities) == len(comps):
                        # add component to the graph
                        if c not in self.G:
                            self.G.add_node(c)
                        # add edges and corresponding weights
                        for (next_c, probability) in zip(comps, probabilities):
                            self.G.add_edge(c, next_c, transition_probability=probability, data_size=0)
                    else:
                         self.error.log("ERROR: no match between components list and probabilities list", 1)
                         sys.exit(1)
                else:
                        if len( graph_dict)>0:
                            self.G.add_node(c)
                        else:
                             print("ERROR: there is not any component in DAG")
                             sys.exit(1)
    
    
    ## Method to write the graph object onto a gml file
    #   @param self The object pointer
    #   @param graph_file File where to print the graph (gml format)
    def write_DAG(self, graph_file):
        nx.write_gml(self.G, graph_file)
    
    
    ## Method to plot the graph object. Edges have different widths according 
    # to the corresponding probability
    #   @param self The object pointer
    #   @param plot_file File where to plot the graph (optional)
    def plot_DAG(self, plot_file = ""):
        
        # weight edges according to the attached probability
        elarge = [(u, v) for (u, v, d) in self.G.edges(data=True) \
                        if d["transition_probability"] >= 1.0]
        esmall = [(u, v) for (u, v, d) in self.G.edges(data=True) \
                        if d["transition_probability"] < 1.0]
        
        # define position of nodes
        pos = nx.shell_layout(self.G)
        
        # draw nodes
        if "start" in list(self.G) and "end" in list(self.G):
            node_list = list(self.G)
            node_color = ["k" if x=="start" or x=="end" else "gold" \
                          for x in node_list]
        else:
            node_color = "gold"
        nx.draw_networkx_nodes(self.G, pos, node_size=500, 
                               node_color=node_color)
        
        # edges
        nx.draw_networkx_edges(self.G, pos, edgelist=elarge, width=3, 
                               edge_color="k")
        nx.draw_networkx_edges(self.G, pos, edgelist=esmall, width=2, 
                               alpha=0.5, edge_color="k")
        
        # add node labels
        nx.draw_networkx_labels(self.G, pos, font_size=10, 
                                font_family="sans-serif")
        
        # add edge labels (data size)
        e_labels = {(u, v): d["data_size"] for (u, v, d) in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=e_labels, 
                                     font_size=10, font_family="sans-serif")
        
        # other graph properties
        ax = plt.gca()
        ax.margins(0.1)
        plt.axis("off")
        plt.tight_layout()
        
        # show or save plot
        if plot_file:
            plt.savefig(plot_file)
        else:
            plt.show()
    
    
    ## Operator to convert a Graph object into a string
    #   @param self The object pointer
    def __str__(self):
        
        s = ''
        
        # loop over nodes
        for node in self.G.nodes:
            # get the successors and the transition probabilities for the 
            # current node
            successors = []
            probabilities = []
            for successor in self.G.successors(node):
                successors.append(str(successor))
                p = self.G.get_edge_data(node, successor)["transition_probability"]
                probabilities.append(float(p))
            # append to the string if the list of successors is not empty
            if len(successors) > 0:
                s += ('"' + str(node) + '": {"next": ')
                s += (str(successors).replace("'", '"') + \
                      ', "transition_probability":' + \
                      str(probabilities) + '},')
        s = s[:-1]
        
        return s



## Component
#
# Class to represent AI applications components
class Component():
    
    ## @var name
    # Component name (used to uniquely identify it)
    
    ## @var deployments
    # List of all the candidate Graph.Component.Deployment objects
    
    ## @var comp_Lambda
    # Load factor

    ## Component class constructor
    #   @param self The object pointer
    #   @param name Component name
    #   @param deployments List of Graph.Component.Deployment objects
    #   @param comp_Lambda Load factor
    def __init__(self, name, deployments,partitions, comp_Lambda):
        self.name = name
        self.deployments = deployments
        self.partitions=partitions
        self.comp_Lambda = comp_Lambda
        
    ## Operator used to check if two Graph.Component objects are equal, 
    # comparing the corresponding names
    #   @param self The object pointer
    #   @param other The rhs Component
    def __eq__(self, other):
        return self.name == other.name
    
    ## Operator to convert a Graph.Component object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '"' + self.name + '": {'
        for deployment in self.deployments:
            s += (str(deployment) + ',')
        s = s[:-1] + '}'
        return s
    
    
    ## Deployment
    #
    # Class to represent a candidate deployment for the corresponding 
    # Graph.Component
    class  Deployment():
        
        ## @var name
        # Deployment name (used to uniquely identify it)
        
        ## @var partitions
        # List of Graph.Component.Deployment.Partition objects characterizing 
        # the Deployment
       
        ## Deployment class constructor
        #   @param self The object pointer
        #   @param name Name of the Deployment
        #   @param partitions List of Graph.Component.Deployment.Partition 
        #                     objects
        def __init__(self, name, partitions):
             self.name = name
             self.partitions = partitions
             
        ## Operator to convert a Graph.Component.Deployment object into a 
        # string
        #   @param self The object pointer
        def __str__(self):
            s = '"' + self.name + '": {'
            for partition in self.partitions:
                s += (str(partition) + ',')
            s = s[:-1] + '}'
            return s
             
       
        ## Partition
        #
        # Class to represent a partition in a candidate deployment
    class Partition():
            
            ## @var name
            # Partition name (used to uniquely identify it)
            
            ## @var memory
            # Memory requirement of the partitions
            
            ## @var part_Lambda
            # Load factor
            
            ## @var early_stopping_probability
            # Probability of early stopping
            
            ## @var Next
            # Name of subsequent partition
            
            ## @var data_size
            # Amount of data transferred to the subsequent partition
            
            ## Partition class constructor
            #   @param name Partition name (used to uniquely identify it)
            #   @param memory Memory requirement of the partitions
            #   @param part_Lambda Load factor
            #   @param early_stopping_probability Probability of early stopping
            #   @param Next Name of subsequent partition
            #   @param data_size Amount of data transferred to the subsequent 
            #                    partition
            def __init__(self, name, memory, part_Lambda, early_exit_probability, 
                         Next, data_size):
                self.name = name
                self.memory = memory
                self.part_Lambda = part_Lambda
                self.early_exit_probability = early_exit_probability
                self.Next = Next
                self.data_size = data_size
                 
            ## Operator to convert a Graph.Component.Deployment.Partition 
            # object into a string
            #   @param self The object pointer
            def __str__(self):
                s = '"{}": {{"memory":{}, "next":{}, "early_stopping_probability":{}, "data_size":{}}}'.\
                    format(self.name, self.memory, '"'+self.Next+'"', 
                           self.early_stopping_probability, self.data_size)
                return s
                
