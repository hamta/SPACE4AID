from abc import ABC, abstractmethod
import numpy as np
import pdb
import sys
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
        
        key=""
        # get the resource of current component and evaluat the performance of the component
        j=np.nonzero(y_hat[component_idx])[0][0]
       
        if j<S.cloud_start_index:
             key="EdgeResources"
             perf_evaluation = S.resources[key].performance_evaluator.evaluate_component(
                                     component_idx,j,y_hat, S.demand_matrix, S.Lambdas)
        elif j<S.FaaS_start_index:
             key="CloudResources"
             perf_evaluation = S.resources[key].performance_evaluator.evaluate_component(
                                     component_idx,j,y_hat, S.demand_matrix, S.Lambdas)
        else:
             key="FaaSResources"
             perf_evaluation = S.resources[key].performance_evaluator.evaluate_component(
                                     component_idx,j, S.demand_matrix)
        
        return perf_evaluation

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
        
    ## Method to return the common network domain between the resource of first component and the second one
    #   @param self The object pointer
    #   @param cpm1_resource resource of first component
    #   @param cpm1_resource resource of second component
    #   @param S instance of System class
    #   @param return common network domain
    def get_network_delay(self,comp_index,cpm1_resource,cpm2_resource,S):
       
       
        CL1=list(filter(lambda resource: (cpm1_resource in resource), S.resources_CL))[0][1]
        CL2=list(filter(lambda resource: (cpm2_resource in resource), S.resources_CL))[0][1]
        x1=list(filter(lambda cl: (CL1 in cl), S.CL_NDs))[0][1]
        x2=list(filter(lambda cl: (CL2 in cl), S.CL_NDs))[0][1]
        ND=list(set(x1).intersection(x2))
      
        if len(ND)==0:
            print("ERROR: no network domain available between two resources "
              + str(cpm1_resource)+" and "+str(cpm2_resource)) 
            sys.exit(1)
        
        elif len(ND)==1:
            l=list(filter(lambda network_technology: 
                              (ND[0] in network_technology.ND_name), S.network_technologies))
            network_delay=l[0].performance_evaluator.evaluate(
                               l[0].access_delay,l[0].bandwidth, S.data_sizes[comp_index][self.path.index(comp_index)+1])
        
        else :
            network_delay=float("inf")
            for nd in ND:
                l=list(filter(lambda network_technology: 
                              (nd in network_technology.ND_name), S.network_technologies))
                new_network_delay=l[0].performance_evaluator.evaluate(
                               l[0].access_delay,l[0].bandwidth, S.data_sizes[comp_index][self.path.index(comp_index)+1])       
                if new_network_delay<network_delay:
                   network_delay=new_network_delay 
        return network_delay
     
        
    
    
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
               cpm1_resource=np.nonzero(solution.Y_hat[comp_index,:])[0][0]
               cpm2_resource=np.nonzero(solution.Y_hat[self.path[self.path.index(comp_index)+1],:])[0][0]
               if not cpm1_resource==cpm2_resource:
                   network_delay=self.get_network_delay(comp_index,cpm1_resource,cpm2_resource,S)
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
