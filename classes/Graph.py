import networkx as nx
import matplotlib.pyplot as plt
import sys

## DAG
#
# Class to represent a Directed Acyclic Graph. The nodes represent the names 
# of existing components, while edges represent precedence relations between 
# components. Each edge is characterized by a number in [0, 1], representing 
# the probabiliy of moving between two adjacent nodes
class DAG():

    ## @var G
    # Object of type nx.DiGraph used to represent the directed acyclic graph

    ## DAG class constructor
    #   @param self The object pointer
    #   @param graph_file File with the graph description (gml or text format)
    def __init__(self, graph_dict):
        self.read_DAG(graph_dict)
          
    
    # ## Method to generate a DAG object starting from the description provided
    # # in a text file (which should be in gml or text format).
    # #   @param self The object pointer
    # #   @param graph_file File with the graph description (gml or text format)
    # def read_DAG(self, graph_file):
        
    #     if graph_file.endswith(".gml"):
    #         self.G = nx.read_gml(graph_file)
    #     else:
    #         with open(graph_file) as f:
    #             # read file lines
    #             lines = f.read().splitlines()
    #             # initialize directed graph
    #             self.G = nx.DiGraph()
    #             # populate graph
    #             make_tuple = lambda x : (x[0], x[1], float(x[2]))
    #             graph_list = list((make_tuple(el.split()) for el in lines))
    #             self.G.add_weighted_edges_from(graph_list)
     ## Method to generate a DAG object starting from the description provided
    # in a text file (which should be in gml or text format).
    #   @param self The object pointer
    #   @param graph_file File with the graph description (gml or text format)
    def read_DAG(self, graph_dict):
        
        self.G=nx.DiGraph()
    
        for c in graph_dict:
                
                if "next" in graph_dict[c] and \
                    "data_size" in graph_dict[c].keys():
                    comps=list(graph_dict[c]["next"])
                    sizes=list(graph_dict[c]["data_size"])
                    probabilities=list(graph_dict[c]["transition_probability"])
                    if len(comps)==len(sizes) and len(probabilities)==len(comps):
                        if c not in self.G:
                            self.G.add_node(c)
                        for (next_c, size, probability) in zip(comps, sizes, probabilities):
                            self.G.add_edge( c,next_c, data_size=size, transition_probability=probability)
                        
                            
                    else:
                         print("ERROR: no match between components list, probabilities list and data size list")
                         sys.exit(1)
       
    ## Method to write the graph object onto a gml file
    #   @param self The object pointer
    #   @param graph_file File where to print the graph (gml format)
    def write_DAG(self):
        nx.write_gml(self.G)
    
    
    ## Method to plot the graph object. Edges have different widths according 
    # to the corresponding probability
    #   @param self The object pointer
    #   @param plot_file File where to plot the graph (optional)
    def plot_DAG(self, plot_file = ""):
        
        # weight edges according to the attached probability
        elarge = [(u, v) for (u, v, d) in self.G.edges(data=True) \
                        if d["weight"] >= 1.0]
        esmall = [(u, v) for (u, v, d) in self.G.edges(data=True) \
                        if d["weight"] < 1.0]
        
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
        
        # ladd node abels
        nx.draw_networkx_labels(self.G, pos, font_size=10, 
                                font_family="sans-serif")
        
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


## Component
class Component():
    
    ## @var name
    # Component name (used to uniquely identify it)
    
    ## @var memory
    # Component memory requirement
    
    ## @var Lambda
    # load factor
    
    ## @var max_res_time
    # Maximum response time for the component

    ## Component class constructor
    #   @param self The object pointer
    #   @param name
    #   @param memory
    #   @param Lambda

    def __init__(self,name, memory, Lambda):
        self.name = name
        self.memory = memory
        self.Lambda = Lambda
        

    ## Operator used to check if two Component objects are equal
    #
    # It compares the corresponding names
    #   @param self The object pointer
    #   @param other The rhs Component
    def __eq__(self, other):
        return self.name == other.name
    
    ## Operator to convert a Component object into a string
    #   @param self The object pointer
    def __str__(self):
        s = '"{}": {{"memory":{}, "lambda":{}'.\
            format(self.name, self.memory, self.Lambda )
        return s
    