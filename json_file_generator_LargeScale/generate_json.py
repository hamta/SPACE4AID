
import pdb
import random
import numpy as np
import copy
import json

def main():
    for i in range(1,2):
        # Number of component in the system
        component_number=10
        # Number of drone 
        Drone_number=2
        # Number of edge node
        Edge_number=2
        # Number of total VM types in the system
        VM_number=4
        # Number of layers in cloud 
        cloud_layers_number=3
        # Number of layers in edge
        edge_layers_number=3
        # Number of different function configuration in FaaS
        FaaS_number=2
        # Number of components which can run only on drone 
        only_drone_components_number=1
         # Number of components which can run  both on edge and drone
        edge_drone_components_number=3
        # Number of components which can run only on edge 
        only_edge_components_number=2
        # Number of components which can run on both edge and cloud 
        edge_cloud_components_number=2
        # Number of components which can run only on cloud
        only_cloud_components_number=2
        
        # Enter the components with the list of their compatible FaaS
        components_FaaS_list=[(8,[1,2]),(9,[2]), (10,[1,2])]
        
        # a list of local constraints as tuples. The first item of tuple is an integer as the number of component and the second item is threshould 
        local_constraints=[(3,3), (4,4)]
        
        # a list of global constraints as tuples. The first item of tuple is a list of components and the second item is threshould 
        global_constraints=[([1, 2, 3],3), ([5,6],4)]
        
        # a list of network domain as thriple. The first item of thriple is a list of computational layer, 
        # the second item is access delay and the third item is bandwidth
        network_technology=[([1, 2, 3],0.001, 100), ([2,3],0.001, 200)]
        
        # a list of arcs of DAG. Each item of list has 4 items, the first and second are the node of arc, 
        # the third item is transition probability and the forth item is data size.
        DAG=[(1,2,0.9, 4),(2,3,1, 6)]
        
        Dict={}
        Dict["Components"]=component_dic(component_number)
        Dict["EdgeResources"]=edge_res(Drone_number,Edge_number, edge_layers_number)
        Dict["CloudResources"]= cloud_res(VM_number,cloud_layers_number , edge_layers_number)
        Dict["FaaSResources"], comp_F_name_list=FaaS_res(FaaS_number, components_FaaS_list,cloud_layers_number,edge_layers_number)
        
        Dict["CompatibilityMatrix"], Dict["DemandMatrix"]=compatibility_demand_matrix(only_drone_components_number,
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
        Dict["LocalConstraints"]=Local_Constraints(local_constraints)
        Dict["GlobalConstraints"]=Global_Constraints(global_constraints)
        Dict["NetworkTechnology"]=Network_technology(network_technology)
        Dict["DirectedAcyclicGraph"]=Directed_Acyclic_Graph(DAG)
        Dict["Time"]=1
        
        with open('system_description'+str(i)+'.json', 'w') as fp:
            json.dump(Dict, fp, indent=4)
        
     
    
    

def component_dic(comp_number):
       
        d={}
        for i in range(1,comp_number+1):
            m={}
            m["memory"]=0
            d["c"+str(i)]=m
        return d

def edge_res(Dron_num,Edge_num,edge_layers_number):
       
        d={}
        d1={}
        d2={}
        # all drones are located in one computational layer
        for i in range(1,Dron_num+1):
            m={}
            m["number"]=1
            m["cost"]=random.uniform(4,5)
            m["memory"]=100
            d1["Drone"+str(i)]=m
        d["computationallayer1"]=d1
        
        for l in range(2,edge_layers_number+1):
            for i in range(1,Edge_num+1):
                m={}
                m["number"]=1
                m["cost"]=random.uniform(6,8)
                m["memory"]=100
                d2["EN"+str(i)]=m
            d["computationallayer"+str(l)]=d2
        return d

def cloud_res(VM_num, cloud_layers_num, edge_layers_num):
   
    
     idx=1
     d={}
     for l in range(1,cloud_layers_num+1): 
        d1={}
       
        for i in range(1,VM_num+1):
                m={}
              
                m["max_number"]=random.randint(3,5)
                m["cost"]=random.random()*idx
                m["memory"]=100
                d1["VM"+str(idx)]=m
                idx+=1
        d["computationallayer"+str(l+edge_layers_num)]=d1
      
   
     return d 

def FaaS_res(FaaS_num,components_FaaS_list,cloud_layers_num, edge_layers_num):
        #pdb.set_trace()
        d={}
        d1={}
        idx=1
        comp_F_name=[]
        for comp in components_FaaS_list:
            l=[]
            for i in comp[1]:
                
                m={}
                
                m["cost"]=random.random()
                m["memory"]=100
                m["idle_time_before_kill"]=600
                d1["F"+str(idx)]=m
                l.append("F"+str(idx))
                idx+=1
            comp_F_name.append((comp[0],l))
        d["computationallayer"+str(cloud_layers_num+ edge_layers_num+1)]=d1
        d["transition_cost"]= 0
        
        
        return d, comp_F_name
    
def compatibility_demand_matrix(only_drone, drone_edge,only_edge,edge_cloud, 
                                only_cloud,comp_F_name_list, Dron_num, Edge_num,
                                Cloud_num, FaaS_num,edge_layers_number, cloud_layers_number):
   
   drone_list=[]
   for i in range(1,Dron_num+1):
        Str="Drone"+str(i)
        drone_list.append(Str)
   
   edge_list=[]
   for i in range(1,Edge_num*edge_layers_number+1):
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
   
   d={}
   m={}
   
   for i in range(1, only_drone+1):
        
         m["c"+str(i)]=copy.deepcopy(drone_only_list)
         d1={}
         for node in drone_list:
             d1[node]=random.uniform(1,2)
         
         
         d["c"+str(i)]=d1
         
   for i in range(only_drone+1, drone_edge+only_drone+1):
        
         m["c"+str(i)]=copy.deepcopy(drone_edge_list)
         d1={}
         for node in drone_list:
             d1[node]=random.uniform(1,2)
         
         for node in edge_list:
             d1[node]=random.uniform(1, 5)    
         d["c"+str(i)]=d1
   
         
   for i in range(drone_edge+only_drone+1, drone_edge+only_drone+only_edge+1):
        
         m["c"+str(i)]=copy.deepcopy(edge_only_list)
         d1={}
         
         for node in edge_list:
             d1[node]=random.uniform(1, 5)    
         d["c"+str(i)]=d1
   
   for i in range(drone_edge+only_drone+only_edge+1, drone_edge+only_drone+edge_cloud+only_edge+1):
        
         m["c"+str(i)]=copy.deepcopy(cloud_edge_list)
         d1={}
         for node in edge_list:
             d1[node]=random.uniform(1,5)
         for node in cloud_list:
             d1[node]=random.uniform(0.5,2)  
         # for node in FaaS_list:
         #     x=random.uniform(0,0.5)
         #     d1[node]=[x,x+random.uniform(0,0.5)]
         d["c"+str(i)]=d1
         
   for i in range(drone_edge+only_drone+edge_cloud+only_edge+1, drone_edge+only_drone+only_cloud+edge_cloud+only_edge+1):
        
         m["c"+str(i)]=copy.deepcopy(cloud_only_list)
         d1={}
         for node in cloud_list:
             d1[node]=random.uniform(0.5,2)  
         # for node in FaaS_list:
         #     x=random.uniform(0,0.5)
         #     d1[node]=[x,x+random.uniform(0,0.5)]
         d["c"+str(i)]=d1
   #pdb.set_trace() 
  
   for l in comp_F_name_list:
        d1={}
        if "c"+str(l[0]) in m.keys():
            for i in l[1]:
               m["c"+str(l[0])].append(i)
               x1=random.uniform(2,3)
               d["c"+str(l[0])][i]=[x1,x1+random.uniform(1,2)]
            #.append(d1)   
    
   return m,d    

def Local_Constraints(local_constraints):
    
    m={}
    for comp in local_constraints:
        d={}
        d["local_res_time"]=comp[1]
        m["c"+str(comp[0])]=d
            
    return m

def Global_Constraints(global_constraints):
    
    p={}
    idx=1
    for path in global_constraints:
        m={}
        components=[]
        for comp in path[0]:
        
                c="c"+str(comp)
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


def Directed_Acyclic_Graph(DAG):
    
    n={}
   
    for edge in DAG:
        m={}
        
        
        m["next"]="c"+str(edge[1])
        m["transition_probability"]=edge[2]
        m["data_size"]=edge[3]
        n["c"+str(edge[0])]=m
       
    return n
if __name__ == '__main__':
    main()