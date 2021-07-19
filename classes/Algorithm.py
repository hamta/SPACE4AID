from classes.Solution import Configuration
from classes.System import System
import sys
import numpy as np
import pdb    
import copy
import random
import time 
import multiprocessing as mpp 
from multiprocessing import Process, Pool
import itertools
## Algorithm
class Algorithm:
    
    def __init__(self, system,Y_hat=[]):
        self.system=system
        self.best_solution=None
        self.minimum_cost=None
         # creat random initial assignments
        
        self.conf=[]
        if Y_hat==[]:
            self.conf.append(Configuration(self.create_random_initial_solution(2,self.system)))
        else:
            self.conf.append(Configuration(Y_hat))
        
    
          
    ## Method to create the initial random solution
    #   @param self The object pointer
    #   @param seed Seed for random number generators
    def create_random_initial_solution(self, seed,S):
        
        np.seed = seed
       # done=False
       
       
        
        I=len(S.components)
        J=len(S.resources)
        y_hat=[]
        y=[]
        for i in range(I):
            H,J=S.compatibility_matrix[i].shape
            y_hat.append(np.full((H, J), 0, dtype=int))
            y.append(np.full((H, J), 0, dtype=int))
        candidate_nodes=[]
        for l in S.CLs:
            if l == list(S.CLs)[-1]:
                random_num=l.resources
                candidate_nodes.extend(random_num)
            else:
                random_num = random.choice(l.resources)
                candidate_nodes.append(random_num)
        
        flag=False
        prevoius_part=None
        for comp in S.components:
            random_dep=random.choice(comp.deployments)
            
            h=0
            
            for part in random_dep.partitions:
                
                
                i=S.dic_map_part_idx[comp.name][part.name][0]
                h_idx=S.dic_map_part_idx[comp.name][part.name][1]
                idx=np.nonzero(S.compatibility_matrix[i][h_idx,:])[0]
                #idx[idx>S.FaaS_start_index-1]=10000
                #idx=list(set(idx))
                index=list(set(candidate_nodes).intersection(idx))
                prob=1/len(index)
                step=0
                rn=np.random.random()
                
                for r in np.arange(0,1,prob):
                    if rn>r and rn<=r+prob:
                        j= index[step]
                       
                    else:
                        step+=1
                y[i][h_idx][j]=1
                y_hat[i][h_idx][j]=1
                
                if  S.graph.G.succ[comp.name]!={}:
                    if part.Next==list(S.graph.G.succ[comp.name].keys())[0]:
                    
                        S.graph.G[comp.name][part.Next]["data_size"]=part.data_size
                ############# computing Lambda ################
                # if S.graph.G.in_edges(comp.name):
                        
                #         if h==0:
                #             Sum = 0
                #             for n, c, data in S.graph.G.in_edges(comp.name, data=True):
                #                 Sum += float(data["transition_probability"])*S.components[S.dic_map_part_idx[n][prevoius_part.name][0]].output_Lambda
                #             part.Lambda=Sum
                            
                #         else:
                #             part.Lambda= prevoius_part.Lambda * (1-part.stop_probability)
                #         prevoius_part=copy.deepcopy(part)
                # else:
                #     if not flag:
                #         flag=True
                #         part.Lambda=S.Lambda * (1-part.stop_probability)
                #         prevoius_part=copy.deepcopy(part)
                # if part==list(random_dep.partitions)[-1]:
                #     S.components[S.dic_map_part_idx[comp.name][part.name][0]].output_Lambda= part.Lambda
                # h+=1  
                # ############# computing demand of FaaS ################
                # if j>=S.FaaS_start_index:
                    
                #     arrival_rate = part.Lambda
                #     res=next((k for k in S.dic_map_res_idx if S.dic_map_res_idx[k] == j), None)
                #     warm_service_time=S.demand_dict[comp.name][part.name][res][0]
                #     cold_service_time=S.demand_dict[comp.name][part.name][res][1]
                #     S.demand_matrix[i][h_idx][j] = S.resources[S.dic_map_res_idx[res]].get_avg_res_time(
                #         arrival_rate, warm_service_time, cold_service_time)

            
        for j in range(S.FaaS_start_index):
             
            number=np.random.randint(1,S.resources[j].number+1)
              
            for i in range(I):
                 H=S.compatibility_matrix[i].shape[0]
                 for h in range(H):
                    if y[i][h][j]>0:
                        y_hat[i][h][j] = y[i][h][j]*number
             
        
       
        return  y_hat
 
    
        
       
        
    ## Method to generate x which shows which resources are used
    # and y_bar which shows maximum number of each resource type that is used 
    #   @param self The object pointer
    #   @param y_hat assignment matrix
    def generate_x_y_bar(self,y_hat):
        
        I,J=y_hat.shape 
        
        x = np.full(J, 0, dtype=int)
        
        y_bar = np.full(J, 0, dtype=int)
        
        x[y_hat.sum(axis=0)>0]=1
        
        y_bar=y_hat.max(axis=0)
        
        return x, y_bar

    
   
   
    ## Method increases the number of allocated resources to a component
    #   @param self The object pointer
    #   @param comp_idx the index of component 
    #   @param solution The current feasible solution
    #   @param S an instance of System class
    #   @param return a boolean value
    def increase_number_of_resource(self,comp_idx, solution,S):
        resource_idx=np.nonzero(solution.Y_hat[comp_idx,:])[0] 
        if solution.Y_hat[comp_idx][resource_idx]<S.resource_number[resource_idx]:
            solution.Y_hat[comp_idx][resource_idx]+=1
            
            return True
        else:
            return False
    
    ## Method return all alternative resources of current allocated resource for the specified component 
    #   @param self The object pointer
    #   @param comp_idx the index of component 
    #   @param solution The current feasible solution 
    #   @param S an instance of System class
    def alternative_resources(self, comp_idx, solution, S): 
        
        l=np.greater(S.compatibility_matrix[comp_idx,:], solution.Y_hat[comp_idx,:])
        resource_idxs=np.where(l)[0]
        return resource_idxs
   
  
     ## Method reduce the number of VM in a cluster 
    #   @param self The object pointer
    #   @param comp_idx the index of component 
    #   @param solution The current feasible solution 
    #   @param S an instance of System class
    def reduce_cluster_size(self,resource_idx, solution, S):
        
      
        flag=True
        if resource_idx< S.FaaS_start_index:
            if S.resources[resource_idx].number>1:
               y_max=[] 
               for i in range(len(solution.Y_hat)):  
                  y_max.append(np.array(solution.Y_hat[i].max(axis=0), dtype=int))

               y_bar=[max(i) for i in itertools.zip_longest(*y_max, fillvalue = 0)]
             
               while flag and  y_bar[resource_idx].max()>1:
                 temp=copy.deepcopy(solution.Y_hat)
              
                 for i in range(len(solution.Y_hat)):
                     for h in range(len(solution.Y_hat[i])):
                         if temp[i][h,resource_idx]>1:
                             temp[i][h,resource_idx]-=1
                 new_solution=Configuration(temp)
                
                 flag, paths_performance, components_performance=new_solution.check_feasibility(S)
                 if flag:
                    solution= copy.deepcopy(new_solution)
                    y_max=[] 
                    for i in range(len(solution.Y_hat)):  
                          y_max.append(np.array(solution.Y_hat[i].max(axis=0), dtype=int))
        
                    y_bar=[max(i) for i in itertools.zip_longest(*y_max, fillvalue = 0)]
                   
        return solution
           
## IteratedLocalSearch  
#
# Specialization of Algorithm      
class IteratedLocalSearch(Algorithm):
    
    pass


## RandomGreedy
#
# Specialization of Algorithm
class RandomGreedy(Algorithm):
    
   def __init__(self, system, Y_hat=[]):
        super().__init__(system, Y_hat)
        
   
    ## Method generate random gready solution 
    #   @param self The object pointer
    #   @param S an instance of System class 
    #   @param N the number of random solution
   def random_greedy_single_processing(self,N):
       
    cost=np.full((1,2), tuple)
    costs=[]
    minimum_cost=float("inf")
    best_solution=None
    primary_best_solution=None
   
    for i in range(N):
        
        solution=Configuration(self.create_random_initial_solution(2,self.system))
       
        
        flag,  paths_performance, components_performance=solution.check_feasibility(self.system)
        #pdb.set_trace()
        if flag:
            
            primary_solution=copy.deepcopy(solution)
            cost[0][0]=solution.objective_function(0, self.system) 
            J=len(self.system.resources)
            for j in range(self.system.cloud_start_index,J):
                
                #if j>=self.system.cloud_start_index:
                    solution= self.reduce_cluster_size(j, solution, self.system)
           
            cost[0][1]=solution.objective_function(0, self.system)
            
            costs.append(copy.deepcopy(cost))
            
            
              
            if cost[0][1] < minimum_cost:
                    minimum_cost=cost[0][1]
                    best_solution=solution
                    primary_best_solution=primary_solution
        
    return minimum_cost, primary_best_solution, best_solution 
   
    
        

   def random_greedy(self):
        
        costs=[np.inf]
        primary_costs=[np.inf]
        solution=None
        primary_solution=None
        paths_performance=None
        components_performance=None
      
        #A=Algorithm(self.system)
        solution=self.conf[0]
       
        flag, primary_paths_performance, primary_components_performance =solution.check_feasibility(self.system)
       
        #pdb.set_trace()
        if flag:
           
           
            primary_costs=solution.objective_function(0, self.system) 
            J=len(self.system.resources)
            primary_solution=copy.deepcopy(solution)
            for j in range(J):
                solution= self.reduce_cluster_size(j, solution, self.system)
            
            flag, paths_performance,components_performance =solution.check_feasibility(self.system)
            costs=solution.objective_function(0, self.system)
        return costs,primary_costs, solution, primary_solution, flag, paths_performance, components_performance, primary_paths_performance, primary_components_performance

     
        










