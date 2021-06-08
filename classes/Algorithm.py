from classes.Solution import Configuration
from classes.System import System
import sys
import numpy as np
import pdb    
import copy
import random

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
        I,J=S.compatibility_matrix.shape
        # Initialize assignment matrix
        y_hat=np.full((I, J), 0, dtype=int)
        y=np.full((I, J), 0, dtype=int)
        
        candidate_nodes=[]
        for l in S.CL_resources:
            if l == list(S.CL_resources.keys())[-1]:
                random_num=S.CL_resources[l]
                candidate_nodes.extend(random_num)
            else:
                random_num = random.choice(S.CL_resources[l])
                candidate_nodes.append(random_num)
        
        for i in range(I):
            #self.get_compatable_layers(i, S)
            
            idx=np.nonzero(S.compatibility_matrix[i,:])[0]
            #idx[idx>S.FaaS_start_index-1]=10000
            #idx=list(set(idx))
            index=list(set(candidate_nodes).intersection(idx))
            prob=1/len(index)
            step=0
            rn=np.random.random()
            
            for r in np.arange(0,1,prob):
                if rn>r and rn<=r+prob:
                    j= index[step]
                    # if j==10000:
                    #    j = list(set(candidate_nodes).intersection(idx))
                       
                else:
                    step+=1
            y[i][j]=1
            y_hat[i][j]=1
        
        for j in range( J):
              if j< S.FaaS_start_index:
                  number=np.random.randint(1,S.resource_number[j]+1)
                    
                  for i in range(I):
                      if y[i][j]>0:
                          y_hat[i][j] = y[i][j]*number
              else:
                  y_hat[i][j] = y[i][j]
    
       
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
        
        I, J= solution.Y_hat.shape
        flag=True
        if resource_idx< len(S.resource_number):
            if S.resource_number[resource_idx]>1:
             
              # while flag and  solution.Y_hat[:,resource_idx].max()>1:
              #    Max = max(solution.Y_hat[:,resource_idx])
              #    comps_max=[i for i, j in enumerate(solution.Y_hat[:,resource_idx]) if j == Max]
                 
              #    temp=copy.deepcopy(solution.Y_hat)
              
              #    for i in comps_max:
              #        temp[i][resource_idx]-=1
              #    new_solution=Configuration(temp)
              #    flag=new_solution.check_feasibility(S)
              #    if flag:
              #       solution= new_solution
               while flag and  solution.Y_hat[:,resource_idx].max()>1:
                 temp=copy.deepcopy(solution.Y_hat)
              
                 for i in range(I):
                     if temp[i][resource_idx]>1:
                         temp[i][resource_idx]-=1
                 new_solution=Configuration(temp)
                 flag, Sum=new_solution.check_feasibility(S)
                 if flag:
                    solution= copy.deepcopy(new_solution)
                   
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
   def random_greedy(self,S,N):
    
    cost=np.full((1,2), tuple)
    costs=[]
    minimum_cost=float("inf")
    best_solution=None
    for i in range(N):
        self.conf[0]=Configuration(self.create_random_initial_solution(2,self.system))
        solution=self.conf[0]
      
        flag=solution.check_feasibility(S)
        #pdb.set_trace()
        if flag:
            cost[0][0]=solution.objective_function(0, S) 
            I,J=solution.Y_hat.shape
            for j in range(J):
                
                if j>=self.cloud_start_index:
                    solution= self.reduce_cluster_size(j, solution, S)
           
            cost[0][1]=solution.objective_function(0, S)
            
            costs.append(copy.deepcopy(cost))
            
            
            #pdb.set_trace()    
            if cost[0][1] < minimum_cost:
                    minimum_cost=cost[0][1]
                    best_solution=solution
    # set the best solution and best value properties                
    self.best_solution=best_solution
    self.minimum_cost=minimum_cost
    #self.costs=costs