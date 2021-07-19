from abc import ABC, abstractmethod
import numpy as np
import pdb
import sys
import copy
## PerformanceConstraint
#
# Class used to represent the performance constraint a configuration is 
# subject to
class PerformanceConstraint(ABC):

    ## @var threshold

    ## @var slack_value
    
    ## PerformanceConstraint class constructor
    #   @param self The object pointer
    #   @param max_res_time
    #   @param slack_value
    def __init__(self, max_res_time):
        self.max_res_time = max_res_time
        

    ## Method to check the feasibility of the constraint (abstract)
    #   @param self The object pointer
    @abstractmethod
    def check_feasibility(self):
        pass
    
    ## Method to evaluate performance of component_idx
    #   @param self The object pointer
    #   @param S an instance of System class
    #   @param y_hat a solution
    #   @param component_idx The index of current component
    def get_perf_evaluation(self,S,y_hat,component_idx):
        
       
        perf_evaluation=0
        # get the resource of current component and evaluat the performance of the component
        j=np.nonzero(y_hat[component_idx])
        for h in range(len(j[0])):
            
            if j[1][h]<S.FaaS_start_index:
                 
                 perf_evaluation += S.resources[j[1][h]].performance_evaluator.evaluate_partition(
                                         component_idx,j[0][h],j[1][h],y_hat, S)
            else:
                
                 perf_evaluation += S.resources[j[1][h]].performance_evaluator.evaluate_partition(
                                     component_idx,j[0][h],j[1][h], S)
        if len(j[0])>1:
            network_delay=0
            
            
            for h in range(len(j[0])):
                if h<len(j[0])-1:
                    for comp in  S.dic_map_part_idx:
                        for part in  S.dic_map_part_idx[comp]:
                            if S.dic_map_part_idx[comp][part]==(component_idx,j[0][h]):
                                comp_name=copy.deepcopy(comp)
                                part_name=copy.deepcopy(part)
                    
                    for dep in S.components[component_idx].deployments:
                        for partition in dep.partitions:
                            if partition.name==part_name:
                                data_size=partition.data_size
                    network_delay +=self.get_network_delay(j[1][h],j[1][h+1],S, data_size)
            perf_evaluation +=network_delay
        return perf_evaluation


     ## Method to return the common network domain between the resource of first component and the second one
    #   @param self The object pointer
    #   @param cpm1_resource resource of first component
    #   @param cpm1_resource resource of second component
    #   @param S instance of System class
    #   @param return common network domain
    def get_network_delay(self,cpm1_resource,cpm2_resource,S, data_size):
       
        
        # CL1=list(filter(lambda resource: (cpm1_resource in resource), S.resources_CL))[0][1]
        # CL2=list(filter(lambda resource: (cpm2_resource in resource), S.resources_CL))[0][1]
        CL1=next((cl.name for cl in S.CLs if cpm1_resource in cl.resources ), None)
        CL2=next((cl.name for cl in S.CLs if cpm2_resource in cl.resources ), None)
        
        # x1=list(filter(lambda cl: (CL1 in cl), S.CL_NDs))[0][1]
        # x2=list(filter(lambda cl: (CL2 in cl), S.CL_NDs))[0][1]
        ND1=[l.ND_name for l in list(filter(lambda NT: (CL1 in NT.computationallayers), S.network_technologies))]
        ND2=[l.ND_name for l in list(filter(lambda NT: (CL2 in NT.computationallayers), S.network_technologies))]
        ND=list(set(ND1).intersection(ND2))
      
        if len(ND)==0:
            print("ERROR: no network domain available between two resources "
              + str(cpm1_resource)+" and "+str(cpm2_resource)) 
            sys.exit(1)
        
        elif len(ND)==1:
            l=list(filter(lambda network_technology: 
                              (ND[0] in network_technology.ND_name), S.network_technologies))
            # network_delay=l[0].performance_evaluator.evaluate(
            #                    l[0].access_delay,l[0].bandwidth, S.data_sizes[comp_index][self.path.index(comp_index)+1])
            
           
            network_delay=l[0].performance_evaluator.evaluate(
                               l[0].access_delay,l[0].bandwidth,data_size)
        
        else :
            #comp2_key = [key for key, value in S.dic_map_com_idx.items() if value == self.path.index(comp_index)+1][0]
            network_delay=float("inf")
            for nd in ND:
                l=list(filter(lambda network_technology: 
                              (nd in network_technology.ND_name), S.network_technologies))
                new_network_delay=l[0].performance_evaluator.evaluate(
                               l[0].access_delay,l[0].bandwidth, data_size)
                if new_network_delay<network_delay:
                   network_delay=new_network_delay 
        return network_delay
## LocalConstraint
#
# Specialization of PerformanceConstraint, related to a single Component
class LocalConstraint(PerformanceConstraint):
    
   
    ## @var component
    # Component the constraint is related to 

    ## LocalConstraint class constructor
    #   @param self The object pointer
    #   @param component
    #   @param max_res_time
    #   @param slack_value
   
    
    def __init__(self,  component_idx,max_res_time):
        super().__init__(max_res_time)
        self.component_idx=component_idx
        
    
    
    
    
    ## Method to check the feasibility of the constraints
    #   @param self The object pointer
    #   @param S an instance of system class
    #   @param solution a solution of the problem
    def check_feasibility(self,S,solution):
            flag=False
           
            # evaluate the performance of component
           
            perf_evaluation=self.get_perf_evaluation(S,solution.Y_hat,self.component_idx)
           
            # check if the denumerator is equal to zero
            if not np.isnan(perf_evaluation):
                solution.local_slack_value[self.component_idx]= self.max_res_time - perf_evaluation
                if perf_evaluation<=self.max_res_time:
                    flag=True
            else:
                 solution.local_slack_value[self.component_idx]=float('Inf')  
                 
                 
            return flag,perf_evaluation   
            


## GlobalConstraint
#
# Specialization of PerformanceConstraint, related to a list of Component 
# objects
class GlobalConstraint(PerformanceConstraint):
    
    ## @var path
    # List of Component objects the constraint is related to 
    
    ## GlobalConstraint class constructor
    #   @param self The object pointer
    #   @param path a list of component index which are in the Path
    #   @param max_res_time
    def __init__(self, path,max_res_time):
        super().__init__(max_res_time)
        self.path = path
        
   
     
        
    
    
    ## Method to check the feasibility of the global constraints
    #   @param self The object pointer
    #   @param S instance of System class
    #   @param solution a solution of the problem
    #   of the components in the path
    #   return a boolean value
    def check_feasibility(self,S,solution):
        flag=False
     
        performance_of_components=[]
        perf_evaluation=0
        # check if the performance evaluation of components are not avalable, evaluate them
       
        for comp_index in self.path:
            try:
                perf_evaluation=self.get_perf_evaluation(S,solution.Y_hat,comp_index)
            except:
                pdb.set_trace()
            
            if not perf_evaluation==float("inf") and \
                not np.isnan(perf_evaluation) and perf_evaluation>0:
           
                   performance_of_components.append(perf_evaluation)
            else:
                   solution.global_slack_value=float('Inf')
                   return flag, float('Inf')
                
        # sum the performance of all components in the current path
        Sum=sum(performance_of_components)
        
        # compute network delay
        for comp_index in self.path:
           #pdb.set_trace()
           if self.path.index(comp_index)+1<len(self.path):
               
               j=np.nonzero(solution.Y_hat[comp_index])
               part1_resource=j[1][-1]
               j1=np.nonzero(solution.Y_hat[self.path[self.path.index(comp_index)+1]])
               part2_resource=j1[1][0]
               # cpm1_resource=np.nonzero(solution.Y_hat[comp_index][comp_index,:])[0][0]
               # cpm2_resource=np.nonzero(solution.Y_hat[self.path[self.path.index(comp_index)+1],:])[0][0]
               if not part1_resource==part2_resource:
                   comp1_key = [key for key, value in S.dic_map_com_idx.items() if value == comp_index][0]
                   comp2_key = [key for key, value in S.dic_map_com_idx.items() if value == self.path.index(comp_index)+1][0]
                   data_size=S.graph.G.get_edge_data(comp1_key, comp2_key)["data_size"]
                   network_delay=self.get_network_delay(part1_resource,part2_resource,S,data_size)
                   #if len(ND)>0:
                       # l=list(filter(lambda network_technology: 
                       #               (ND[0] in network_technology.ND_name), S.network_technologies))
                           
                       # network_delay=l[0].performance_evaluator.evaluate(
                       #         l[0].access_delay,l[0].bandwidth, S.data_sizes[comp_index][self.path.index(comp_index)+1])
                      
                   Sum=Sum + network_delay
                   # else:
                   #     pdb.set_trace()
                   #     ND=self.get_ND(cpm1_resource,cpm2_resource,S)
                   #     print("ERROR: no network domain available between two resources "
                   #           + str(cpm1_resource)+" and "+str(cpm2_resource))
                           
           
        # compute the slack value of the current solution
        solution.global_slack_value=self.max_res_time - Sum

        if Sum <= self.max_res_time:
           flag=True 
       
        return flag, Sum
