import numpy as np
from classes.Constraints import LocalConstraint, GlobalConstraint
import pdb
## Configuration
class Configuration:
    
    ## @var n_components
    # Total number of existing components
    
    ## @var n_resources
    # Dictionary with the total number of existing resources indexed by type
    
    ## @var Y_hat
    # 2D numpy array storing the number of Resources assigned to each Component
   

    ## Configuration class constructor
    #   @param self The object pointer
    #   @param Y_hat 2D numpy array storing the number of Resources assigned 
    #          to each Component
    def __init__(self, Y_hat):
        self.Y_hat = Y_hat
        self.n_components, self.n_resources = Y_hat.shape
        self.local_slack_value=np.full(self.n_components,np.inf,dtype=float)
        self.global_slack_value=None
    
   
    ## Method to check the preliminary constraints of the current configuration
    #   @param self The object pointer
    #   @param compatibility_matrix 
    #   @param resource_number an array to show the number of each resource
    #   @return a flag to represent if the preliminary constraints are satisfied
    def preliminary_constraints_check_assignments(self, compatibility_matrix, 
                                                  resource_number):
        #pdb.set_trace()
        flag = False
        #check if each components is assigned to only one resource
        if all(np.count_nonzero(row)==1 for row in self.Y_hat):
            # convert y_hat to y (binary)
            y = np.array(self.Y_hat > 0, dtype=int)
            # check if y is equal or greater than compatability matrix
            if np.all(np.less_equal(y, compatibility_matrix)):
                # check if y_hat is equal or less than n_j
                if all(self.Y_hat.max(axis=0)[0:resource_number.shape[0]]<=resource_number):
                    flag = True             
        return flag       
        
    
    ## Method to check the memory constraint
    #   @param self The object pointer
    #   @param S an instance of system class
    def memory_constraints_check(self, S):
        
        # creat y from y_hat
        y = np.array(self.Y_hat > 0, dtype=int)
        I,J=self.Y_hat.shape
        
        # for each column check is the sum of memory requrement of the components which is assigned to 
        # current resource is greater than the maximum capacity of the current resource
        for j in range(J):
            
            if (y[:,j]*np.array(list(c.memory for c in 
                                     list(S.components.values())))).sum(axis=0) > S.resource_memory[j]:
                return False
        
        return True    
    
    ## Method to check the feasibility of the current configuration
    #   @param self The object pointer
    #   @param S an instance of system class
    #   @param path a list of components included in a path
    def check_feasibility(self, S):
        I= len(S.LC)
        flag = False
        Sum=0
        # creat a list to show each component satisfies its local constraint, 
        # if so, the coresponding element is True, and False otherwise.
        check_components = np.full((I,2), tuple)
        
        # check if the preliminary constraints are satisfied
        #flag = self.preliminary_constraints_check_assignments(S.compatibility_matrix,S.resource_number)
        
        # check if the memory constraint is satisfied
        flag=self.memory_constraints_check(S)
        if flag:
             # check local constraints for all components
           # pdb.set_trace()
            for i in range(I):
                lc = LocalConstraint(int(S.LC[i][0]),S.LC[i][1])
                check_components[i][0]=S.LC[i][0]
                check_components[i][1]=lc.check_feasibility(S,self)
                
            
        else:
            return flag
        
        flag = all([ comp[1][0] for comp in check_components ] )        
      
            
         # check global constraint is the local constraints of all components are satisfied
        for path_idx in S.GC:
            if flag:
                
                gc=GlobalConstraint(path_idx[0],path_idx[1])
                #performance_of_components=[comp[1][1] for comp in check_components]
                try:
                    flag, Sum=gc.check_feasibility(S,self)
                except:
                    pdb.set_trace()
        
        return flag, Sum

     ## Method to compute the cost of a feasible solution
    #   @param self The object pointer
    #   @param solution_idx The index of current feasible solution
    #   @param S an instance of System class
    #   @param SCAR a boolean valuse shows if SCAR is used
    def objective_function(self, solution_idx, S):
       
       I,J=self.Y_hat.shape 
        
       x = np.full(J, 0, dtype=int) 
       #y = np.array(self.conf[solution_idx].Y_hat > 0, dtype=int) 
       x[self.Y_hat.sum(axis=0)>0]=1
       y_bar=np.array(self.Y_hat.max(axis=0), dtype=int) 
       
       cost=0
       # compute cost of edge
       for j in range(S.cloud_start_index):
           cost+=S.resource_cost[j] * x[j]
       
        # compute cost of VMs  
       for j in range(S.cloud_start_index,S.FaaS_start_index):
           cost+=S.resource_cost[j] * y_bar[j]
          
       # compute the cost of FaaS and transition cost if not using SCAR   
       for j in range(S.FaaS_start_index,J):
           
           if y_bar[j]>0:
               comp_indexes=np.nonzero(self.Y_hat[:,j])[0]
               for comp_idx in comp_indexes:
                   cost+= S.resource_cost[j] * S.demand_matrix[comp_idx][j] * S.Lambdas[comp_idx] * S.T 
                  # + int(not(SCAR)) * S.Lambdas[comp_idx] * S.T
       
       return cost