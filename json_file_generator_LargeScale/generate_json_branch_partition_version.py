
import numpy as np
import copy
import json
import networkx as nx
from collections import defaultdict
import os
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_system(directory,seed, component_number):
    np.random.seed(seed)

    # Maximum number of deployments
    max_deployments=4
    # Maximum number of partitions
    max_partitions=4
    # Number of drone
    Drone_number=1
    # Number of edge node
    Edge_number=5
    # Number of total VM types in the system
    VM_number=5
    # Number of layers in cloud
    cloud_layers_number=3
    # Number of layers in edge including one layer of drones
    edge_layers_number=3
    # Number of different function configuration in FaaS
    FaaS_number=5
    # Number of components which can run only on drone
    only_drone_components_number=2
     # Number of components which can run  both on edge and drone
    edge_drone_components_number=3
    # Number of components which can run only on edge
    only_edge_components_number=3
    # Number of components which can run on both edge and cloud
    edge_cloud_components_number=4
    # Number of components which can run only on cloud
    only_cloud_components_number=3

    min_memory=20
    max_memory=8192*4
    min_data_size=4
    max_data_size=8

    # Enter the components with the list of their compatible FaaS
    components_FaaS_list=[(12,[1,2]), (13,[2,3]), (14,[3,4]), (15,[4,5])]

    LC_number=5
    GC_number=5
    comp_threshold_range=(7, 10)
    path_threshold_range=(20, 25)
    # a list of network domain as thriple. The first item of thriple is a list of computational layer,
    # the second item is access delay and the third item is bandwidth
    network_technology=[([1, 2],0.001, 2000), ([3,4],0.001, 1500),([1,2,3,4,5,6,7],0.001,1000)]


    Dict={}
    Dict["DirectedAcyclicGraph"], DAG=Directed_Acyclic_Graph(component_number, seed)
    dic_DAG=Dict["DirectedAcyclicGraph"]
    Dict["Components"]=component_dic(component_number,max_deployments, max_partitions,dic_DAG, max_memory, min_data_size, max_data_size)
    Dict["EdgeResources"]=edge_res(Drone_number,Edge_number, edge_layers_number,max_memory,component_number)
    Dict["CloudResources"]= cloud_res(VM_number,cloud_layers_number , edge_layers_number,max_memory,component_number)
    Dict["FaaSResources"], comp_F_name_list=FaaS_res(Dict,FaaS_number, components_FaaS_list,cloud_layers_number,edge_layers_number,max_memory,component_number)

    Dict["CompatibilityMatrix"], Dict["Performance"]=compatibility_demand_matrix(Dict,only_drone_components_number,
                                                                                  edge_drone_components_number,
                                                                                  only_edge_components_number,
                                                                                  edge_cloud_components_number,
                                                                                 only_cloud_components_number,
                                                                                 comp_F_name_list,
                                                                                 Drone_number,Edge_number,
                                                                                 VM_number,FaaS_number,
                                                                                 edge_layers_number,
                                                                                 cloud_layers_number)

    Dict["Lambda"]=0.1
    Dict["LocalConstraints"]=Local_Constraints(DAG, LC_number, comp_threshold_range)
    Dict["GlobalConstraints"]=Global_Constraints(DAG,GC_number, path_threshold_range)
    Dict["NetworkTechnology"]=Network_technology(network_technology)

    Dict["Time"]=1
    #pdb.set_trace()
    output_json=directory+"/system_description.json"
    with open(output_json , 'w') as fp:
        json.dump(Dict, fp, cls=NpEncoder, indent=4)

    return output_json, Dict



def component_dic(comp_number, max_deployments, max_partitions, DAG, max_memory, min_data_size, max_data_size):

        comp={}
        for i in range(1,comp_number+1):
            dep={}
            dep_num=np.random.randint(1,max_deployments)
            partitions_number = 0
            parts={}
            part={}
            partitions_number+=1
            if i<comp_number:
                part["next"]=DAG["c"+str(i)]["next"]
                data_size_list=[]
                for rand in DAG["c"+str(i)]["next"]:
                     data_size_list.append(np.random.randint(min_data_size,max_data_size))
                part["data_size"]=data_size_list
            else:
                part["next"]=[]
                part["data_size"]=[np.random.randint(min_data_size,max_data_size)]

            part["early_exit_probability"]=0


            parts["h"+str(partitions_number)]=part
            dep["s1"]= parts
            part_num_list=[item for item in range(2, max_partitions+1)]
            np.random.shuffle(part_num_list)

            for j in range(2,dep_num+1):
                parts={}
                part_num = part_num_list.pop()

                for h in range(1,part_num+1):
                    partitions_number+=1
                    part={}

                    if h==part_num:
                        if i<comp_number:
                            part["next"]=DAG["c"+str(i)]["next"]

                        else:
                            part["next"]=[]
                    else:
                        part["next"]=["h"+str(partitions_number+1)]
                    if h<part_num:
                        part["early_exit_probability"]=np.random.uniform(0,1)
                    else:
                        part["early_exit_probability"]=0

                    if h==part_num and i<comp_number:
                         part["data_size"]=data_size_list
                    else:
                        part["data_size"]=[np.random.randint(min_data_size,max_data_size)]
                    parts["h"+str(partitions_number)]=part
                dep["s"+str(j)]= parts
            comp["c"+str(i)]=dep

        return comp

def edge_res(Dron_num,Edge_num,edge_layers_number,max_memory,comp_number):

        d={}
        d1={}

        # all drones are located in one computational layer
        for i in range(1,Dron_num+1):
            m={}
            m["number"]=1
            m["cost"]=np.random.uniform(4,5)
            m["memory"]=max_memory*comp_number
            m["n_cores"]=1
            d1["Drone"+str(i)]=m
        d["computationallayer1"]=d1
        idx=1
        for l in range(2,edge_layers_number+1):
            d2={}
            for i in range(1,Edge_num+1):
                m={}
                m["number"]=1
                m["cost"]=np.random.uniform(6,8)
                m["memory"]=max_memory*comp_number
                m["n_cores"]=1
                d2["EN"+str(idx)]=m
                idx+=1
            d["computationallayer"+str(l)]=d2
        return d

def cloud_res(VM_num, cloud_layers_num, edge_layers_num,max_memory,comp_number):


     idx=1
     d={}
     for l in range(1,cloud_layers_num+1):
        d1={}

        for i in range(1,VM_num+1):
                m={}

                m["number"]=np.random.randint(3,5)
                m["cost"]=np.random.random()*idx
                m["memory"]=max_memory*comp_number
                m["n_cores"]=1
                d1["VM"+str(idx)]=m
                idx+=1
        d["computationallayer"+str(l+edge_layers_num)]=d1


     return d

def FaaS_res(Dict,FaaS_num,components_FaaS_list,cloud_layers_num, edge_layers_num,max_memory,comp_number):
        #pdb.set_trace()
        d={}
        d1={}
        idx=1
        comp_F_name=[]
        components=Dict["Components"]
        for comp in components_FaaS_list:
            l=[]
            S=components["c"+str(comp[0])]
            for i in comp[1]:


               for s in S:
                 for h in S[s]:
                    m={}

                    m["cost"]=np.random.random()
                    m["memory"]=max_memory*comp_number
                    m["idle_time_before_kill"]=600
                    d1["F"+str(idx)]=m
                    l.append("F"+str(idx))
                    idx+=1
            comp_F_name.append((comp[0],l))
        d1["transition_cost"]= 0
        d["computationallayer"+str(cloud_layers_num+ edge_layers_num+1)]=d1



        return d, comp_F_name

def compatibility_demand_matrix(Dict,only_drone, drone_edge,only_edge,edge_cloud,
                                only_cloud,comp_F_name_list, Dron_num, Edge_num,
                                Cloud_num, FaaS_num,edge_layers_number, cloud_layers_number):

   drone_list=[]
   drone_memory_list=[]

   for i in range(1,Dron_num+1):
        Str="Drone"+str(i)
        drone_list.append(Str)


   edge_list=[]
   for i in range(1,Edge_num*(edge_layers_number-1)+1):
        Str="EN"+str(i)
        edge_list.append(Str)

   edge_only_list=copy.deepcopy(drone_list)
   edge_only_list.extend(edge_list)
   cloud_list=[]
   for i in range(1,Cloud_num*cloud_layers_number+1):
        Str="VM"+str(i)
        cloud_list.append(Str)


   drone_only_list=   copy.deepcopy(drone_list)
   drone_edge_list=copy.deepcopy(drone_list)
   drone_edge_list.extend(edge_list)
   edge_only_list=copy.deepcopy(edge_list)
   cloud_only_list=copy.deepcopy(cloud_list)
   #cloud_only_list.extend(FaaS_list)

   cloud_edge_list=copy.deepcopy(edge_list)
   cloud_edge_list.extend(cloud_only_list)


   compatibility_matrix={}
   components=Dict["Components"]


   demand_matrix={}
   drone_memory=[256, 512]
   edge_memory=[512, 1024, 2048]
   cloud_memory=[2048, 4096, 8192]
   FaaS_memory=[1024, 2048, 4096 ]
   for i in range(1, only_drone+1):
       S=components["c"+str(i)]
       d={}
       m={}
       s_idx=0

       for s in S:
         if s_idx==0:
             base_demand=[]
             for node in drone_list:
                base_demand.append(np.random.uniform(1,2))

         for h in S[s]:
             d1={}
             dic_list=[]
             for res in drone_only_list:
                 dd={}
                 dd["resource"]=res
                 dd["memory"]= np.random.choice(drone_memory)
                 dic_list.append(dd)
             m[h]=dic_list
             dron_idx=0
             for node in drone_list:
                 model={}
                 model["model"]="QTedge"
                 if s_idx == 0:
                     model["demand"]= base_demand[dron_idx]
                 else:
                     model["demand"]= base_demand[dron_idx]/len(S[s]) + np.random.uniform(0.05,0.1)
                 d1[node]=model
                 dron_idx += 1
             d[h]=d1

         s_idx += 1
       compatibility_matrix["c"+str(i)]=m
       demand_matrix["c"+str(i)]=d

   for i in range(only_drone+1, drone_edge+only_drone+1):
       S=components["c"+str(i)]
       d={}
       m={}
       s_idx=0

       for s in S:
         if s_idx==0:
             base_demand=[]
             for node in drone_list:
                base_demand.append(np.random.uniform(1,2))
             for node in edge_list:
                base_demand.append(np.random.uniform(0.6,1.1))
         for h in S[s]:
             d1={}
             dic_list=[]

             for res in drone_only_list:
                 dd={}
                 dd["resource"]=res
                 dd["memory"]= np.random.choice(drone_memory)
                 dic_list.append(dd)
             for res in edge_list:
                 dd={}
                 dd["resource"]=res
                 dd["memory"]= np.random.choice(edge_memory)
                 dic_list.append(dd)
             m[h]=dic_list
             edge_dron_idx=0
             for node in drone_edge_list:
                 model={}
                 model["model"]="QTedge"
                 if s_idx == 0:
                     model["demand"] = base_demand[edge_dron_idx]
                 else:
                     model["demand"] = base_demand[edge_dron_idx]/len(S[s]) + np.random.uniform(0.05,0.1)
                 d1[node]=model
                 edge_dron_idx += 1



             d[h]=d1

         s_idx += 1
       compatibility_matrix["c"+str(i)]=m
       demand_matrix["c"+str(i)]=d




   for i in range(drone_edge+only_drone+1, drone_edge+only_drone+only_edge+1):
       S=components["c"+str(i)]
       d={}
       m={}
       s_idx=0

       for s in S:
         if s_idx==0:
               base_demand=[]
               for node in edge_list:
                    base_demand.append(np.random.uniform(0.6,1.1))
         for h in S[s]:
             d1={}
             dic_list=[]
             for res in edge_list:
                 dd={}
                 dd["resource"]=res
                 dd["memory"]= np.random.choice(edge_memory)
                 dic_list.append(dd)
             m[h]=dic_list
             edge_idx=0
             for node in edge_list:
                 model={}
                 model["model"]="QTedge"
                 if s_idx == 0:
                     model["demand"]=base_demand[edge_idx]
                 else:
                     model["demand"]=base_demand[edge_idx]/len(S[s]) + np.random.uniform(0.05,0.1)
                 d1[node]=model
                 edge_idx += 1

             d[h]=d1
         s_idx += 1
       compatibility_matrix["c"+str(i)]=m
       demand_matrix["c"+str(i)]=d



   for i in range(drone_edge+only_drone+only_edge+1, drone_edge+only_drone+edge_cloud+only_edge+1):
       S=components["c"+str(i)]
       d={}
       m={}
       s_idx=0
       for s in S:
            if s_idx==0:
                 base_demand=[]
                 for node in edge_list:
                    base_demand.append(np.random.uniform(0.6,1.1))
                 for node in cloud_list:
                    base_demand.append(np.random.uniform(0.2,0.7))
            for h in S[s]:
                 d1={}
                 dic_list=[]
                 for res in edge_list:
                     dd={}
                     dd["resource"]=res
                     dd["memory"]= np.random.choice(edge_memory)
                     dic_list.append(dd)
                 for res in cloud_list:
                     dd={}
                     dd["resource"]=res
                     dd["memory"]= np.random.choice(cloud_memory)
                     dic_list.append(dd)
                 m[h]=dic_list
                 edge_cloud_idx=0
                 for node in cloud_edge_list:
                     model={}
                     model["model"]="QTedge"
                     if s_idx == 0:
                         model["demand"]=base_demand[edge_cloud_idx]
                     else:
                         model["demand"]=base_demand[edge_cloud_idx]/len(S[s]) + np.random.uniform(0.05,0.1)
                     d1[node]=model
                     edge_cloud_idx += 1


                 d[h]=d1
            s_idx += 1
       compatibility_matrix["c"+str(i)]=m
       demand_matrix["c"+str(i)]=d




   for i in range(drone_edge+only_drone+edge_cloud+only_edge+1, drone_edge+only_drone+only_cloud+edge_cloud+only_edge+1):
       S=components["c"+str(i)]
       d={}
       m={}
       s_idx = 0
       for s in S:
         if s_idx==0:
                 base_demand=[]
                 for node in cloud_list:
                    base_demand.append(np.random.uniform(0.2,0.7))
         for h in S[s]:
             d1={}
             dic_list=[]
             for res in cloud_list:
                 dd={}
                 dd["resource"]=res
                 dd["memory"]= np.random.choice(cloud_memory)
                 dic_list.append(dd)
             m[h]=dic_list
             cloud_idx = 0
             for node in cloud_list:
                 model={}
                 model["model"]="QTedge"
                 if s_idx == 0:
                     model["demand"]=base_demand[cloud_idx]
                 else:
                     model["demand"]=base_demand[cloud_idx]/len(S[s]) + np.random.uniform(0.05,0.1)
                 d1[node]=model
                 cloud_idx += 1
             d[h]=d1
         s_idx += 1
       compatibility_matrix["c"+str(i)]=m
       demand_matrix["c"+str(i)]=d



   for l in comp_F_name_list:
        d1={}
        idx=0
        Len=len(l[1])

        if "c"+str(l[0]) in compatibility_matrix.keys():
            F_number=int(Len/len(compatibility_matrix["c"+str(l[0])]))

            for h in  compatibility_matrix["c"+str(l[0])]:
                for i in range(F_number):
                       model={}
                       model["model"]="PACSLTK" #"MLLIBfaas"
                       #model["regressor_file"]= "MLmodels/faas/LRRidge.pickle"
                       x1=np.random.uniform(2,3)
                       model["demandWarm"]= x1
                       model["demandCold"]= x1+np.random.uniform(1,2)

                       dd={}
                       dd["resource"]=l[1][idx]
                       dd["memory"]= np.random.choice(FaaS_memory)
                       compatibility_matrix["c"+str(l[0])][h].append(dd)

                       demand_matrix["c"+str(l[0])][h][l[1][idx]]=model#[x1,x1+random.uniform(1,2)]
                       idx+=1
            #.append(d1)

   return compatibility_matrix,demand_matrix

def Local_Constraints( DAG, LC_number, comp_threshold_range):
    m={}
    if LC_number>0:
        local_constraints=[]
        comps=[]
        nodes=list(DAG.nodes)
        while len(local_constraints) < LC_number:
            node=np.random.choice(nodes)
            if node not in comps:
                comps.append(node)
                local_constraints.append((node,np.random.uniform(comp_threshold_range[0],comp_threshold_range[1])))

        for comp in local_constraints:
            d={}
            d["local_res_time"]=comp[1]
            m["c"+str(comp[0]+1)]=d

    return m

def Global_Constraints(DAG,GC_number,path_threshold_range):

    global_constraints=[]
    GCs=[]
    nodes=list(DAG.nodes)
    while len(global_constraints) < GC_number:
        source=np.random.choice(nodes)
        destination=np.random.choice(nodes)
        if not source == destination:
            paths=[path for path in nx.all_simple_paths(DAG, source, destination)]
            if len(paths)>0:
                if len(paths)>1:
                    idx=np.random.randint(0,len(paths)-1)
                else:
                    idx=0
                GC=paths[idx]
                if GC not in GCs:
                    GCs.append(GC)
                    global_constraints.append((GC,np.random.uniform(path_threshold_range[0],path_threshold_range[1])))
    p={}
    idx=1
    for path in global_constraints:
        m={}
        components=[]
        for comp in path[0]:

                c="c"+str(comp+1)
                components.append(c)
        m["components"]=components
        m["global_res_time"]=path[1]
        p["p"+str(idx)]=m
        idx+=1
    return p

def Network_technology(network_technology):

    n={}
    idx=1
    for ND in network_technology:
        m={}
        CL=[]
        for cl in ND[0]:

                c="computationallayer"+str(cl)
                CL.append(c)
        m["computationallayers"]=CL
        m["AccessDelay"]=ND[1]
        m["Bandwidth"]=ND[2]
        n["ND"+str(idx)]=m
        idx+=1
    return n

'''def Directed_Acyclic_Graph(component_number,Seed):
    is_DAG=False
    while not is_DAG:
        G = nx.gnp_random_graph(component_number,0.8,seed=Seed,directed=True)
        DAG = nx.DiGraph([(u,v,{'weight':np.random.random()}) for (u,v) in G.edges() if u<v])
        is_DAG = nx.is_directed_acyclic_graph(DAG)
        Seed += 1
    edge_list=list(DAG.in_edges())
    Edge_weight_list=[]
    for edge in edge_list:
        weight=DAG.get_edge_data(edge[0], edge[1])['weight']
        Edge_weight_list.append((edge[0], edge[1], weight))
    d = defaultdict(list)
    for k, *v in Edge_weight_list:
        d[k].append(v)

    LL=list(d.items())
    n={}
    Nodes=list(DAG.nodes)
    for node in Nodes:
        nexts=[edges[1] for edges in LL if edges[0]==node]
        m={}
        next_list=[]
        trans_list=[]
        if len(nexts)>0:
            rand_list=np.random.random(len(nexts[0]))
            Sum=sum(rand_list)
            idx=0
            for des in nexts[0]:
                DAG[node][des[0]]["weight"]=rand_list[idx]/Sum
                next_list.append("c"+str(des[0]+1))
                trans_list.append(DAG[node][des[0]]["weight"])
                idx+=1
        m["next"]=next_list
        m["transition_probability"]=trans_list


        n["c"+str(node+1)]=m
    return n, DAG'''
def Directed_Acyclic_Graph(component_number,Seed):
    is_DAG=False
    while not is_DAG:
        G = nx.gnp_random_graph(component_number-2,0.8,seed=Seed,directed=True)
        DAG = nx.DiGraph([(u,v,{'weight':np.random.random()}) for (u,v) in G.edges() if u<v])
        is_DAG = nx.is_directed_acyclic_graph(DAG)
        Seed += 1
    mapping={}
    for node in DAG:
        mapping[node]=int(node)+1
    #mapping = {0: "a", 1: "b", 2: "c"}
    DAG = nx.relabel_nodes(DAG, mapping)
    sources=[edge[0] for edge in DAG.in_degree if edge[1]==0]
    DAG.add_node(0)
    for node in sources:
        DAG.add_edge(0,node, weight=np.random.random())
    destinations=[edge[0] for edge in DAG.out_degree if edge[1]==0]
    des=len(DAG.nodes)
    DAG.add_node(des)
    for node in destinations:
        DAG.add_edge(node, des, weight=np.random.random())

    edge_list=list(DAG.in_edges())
    Edge_weight_list=[]
    for edge in edge_list:
        weight=DAG.get_edge_data(edge[0], edge[1])['weight']
        Edge_weight_list.append((edge[0], edge[1], weight))
    d = defaultdict(list)
    for k, *v in Edge_weight_list:
        d[k].append(v)

    LL=list(d.items())
    n={}
    Nodes=list(DAG.nodes)
    for node in Nodes:
        nexts=[edges[1] for edges in LL if edges[0]==node]
        m={}
        next_list=[]
        trans_list=[]
        if len(nexts)>0:
            rand_list=np.random.random(len(nexts[0]))
            Sum=sum(rand_list)
            idx=0
            for des in nexts[0]:
                DAG[node][des[0]]["weight"]=rand_list[idx]/Sum
                next_list.append("c"+str(des[0]+1))
                trans_list.append(DAG[node][des[0]]["weight"])
                idx+=1
        m["next"]=next_list
        m["transition_probability"]=trans_list


        n["c"+str(node+1)]=m
    return n, DAG
def main():
     '''comp_number=15
     dir="/Users/hamtasedghani/space4ai-d/Output_Files/large_scale/"
     #dir="/Users/hamtasedghani/Desktop/ServerlessAppPerfCostMdlOpt/Output_Files/large_scale/"
     for seed in range(1,11):
        directory = dir + str(comp_number)+"Components/Ins" + str(seed)
        if not os.path.exists(directory):
            os.makedirs(directory)
        generate_system(directory,seed,comp_number )'''
     dir="/Users/hamtasedghani/space4ai-d/Output_Files/large_scale_light_constraints/"
     for N in [5, 7, 10, 15 ]:
         for ins in range(10):
             system_file= dir + str(N) + "Components/Ins" + str(ins+1) + "/system_description.json"

             with open(system_file, "r") as a_file:
                 json_object = json.load(a_file)
             for comp in json_object["LocalConstraints"]:
                 json_object["LocalConstraints"][comp]["local_res_time"]=json_object["LocalConstraints"][comp]["local_res_time"]*10

             for path in json_object["GlobalConstraints"]:
                 json_object["GlobalConstraints"][path]["global_res_time"]=json_object["GlobalConstraints"][path]["global_res_time"]*10

             with open(system_file , 'w') as fp:
                json.dump(json_object, fp, cls=NpEncoder, indent=4)
if __name__ == '__main__':
    main()
