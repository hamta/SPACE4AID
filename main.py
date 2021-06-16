from classes.System import System
#from classes.Solution import Configuration
from classes.Constraints import LocalConstraint, GlobalConstraint
from classes.Algorithm import Algorithm, RandomGreedy
from classes.Performance import ServerFarmPE, EdgePE
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pdb
import random
import copy
import json
import matplotlib.pyplot as plt
import time 
import multiprocessing as mpp 
from multiprocessing import Process, Pool

 ## Method to create the pure json file from system description by removing the comments lines
    #   @param main_json_file system description file address
def create_pure_json(main_json_file):
    data = []
    with open(main_json_file,"r") as fp:
        Lines = fp.readlines()
    for line in Lines:
        
        
        idx=line.find('#')
        if ( idx != -1):
            
            last=line[-1]
            line=line[0:idx]
            if last == "\n":
                line=line+last
        data.append(line)
    
    pure_json='pure_json.json'
    filehandle = open(pure_json, "w")
    filehandle.writelines(data)
    filehandle.close()
    return pure_json    

## Method to generate the output json file as the optimal placement for a specified Lambda
    #   @param Lambda incoming workload rate
    #   @param result the optimal result which is a list including all information
    #   @param S an instance of system class including system description
def generate_output_json_file(Lambda, result,S):
   
    best_solution=result[2]
    output={}
    L={}
    I,J=best_solution.Y_hat.shape
    output["Lambda"]=round(float(Lambda), 5)
    
    for i in range(I):
       comp_key = [key for key, value in S.dic_map_com_idx.items() if value == i][0]
       res_idx=int(np.nonzero(best_solution.Y_hat[i,:])[0][0])
      
       res_key = [key for key, value in S.dic_map_res_idx.items() if value == res_idx][0]
       res={}
       res_des={}
       res_des["description"]=S.description[res_key]
       res_des["cost"]=result[0][res_idx]
       res_des["memory"]=S.resources[res_idx].memory
       if res_idx< S.FaaS_start_index:    
            res_des["number"]=int(best_solution.Y_hat[i][res_idx])
       else:
          
            res_des["idle_time_before_kill"]=S.resources[res_idx].idle_time_before_kill
            res_des["transition_cost"]=S.resources[res_idx].transition_cost
            
       res_des["response_time"]=result[6][i][1][1]
       List=[item for item in S.LC if i in item]
       if len(List)>0:
          
           res_des["response_time_threshold"]=List[0][1]
       else:
           res_des["response_time_threshold"]=str(float("inf"))
       res[res_key]=res_des
       CL=S.resources[res_idx].CLname
       comp_layer={}
       comp_layer[CL]=res
       L[comp_key]=comp_layer
    output["components"]=L
    global_constraints={}
    idx=0
    for path in S.GC:
        p={}
        P="path " + str(idx+1)
        comp_list=[]
        for comp in path[0]:
            comp_key = [key for key, value in S.dic_map_com_idx.items() if value == comp][0]
            comp_list.append(comp_key)
        p["components"]=comp_list
        p["path_response_time"]=result[5][idx][1]
        p["path_response_time_threshold"]=path[1]
    global_constraints[P]=p
    output["global_constraints"]=global_constraints   
        
    output["total_cost"]=sum(result[0])
    output_json="OutputFiles/Lambda_"+str(round(float(Lambda), 5)) + '_output_json.json'
    a_file = open(output_json, "w")
    json.dump(output, a_file, indent=4)
    a_file.close()
    
    
    return output

## Method to create an instance of Algorithm class and run the random gready method used in multiprocessing manner
    #   @param m parameter 
    #   @param S an instance of system class including system description
def fun_gready(m,S):
    GA=RandomGreedy(S)
    return GA.random_greedy()



def main(system_file, iteration, start_lambda,end_lambda, step):
    
     system_file=create_pure_json(system_file)
     
     for Lambda in np.arange(start_lambda, end_lambda,step ):
     
            a_file = open(system_file, "r")
            
            json_object = json.load(a_file)
            a_file.close()
            json_object["Lambda"] = Lambda
            
           
            a_file = open(system_file, "w")
            json.dump(json_object, a_file, indent=4)
            
            
            a_file.close()
            S = System(system_file)
            
            
            start=time.time()
         
            
  ################## Multiprocessing ###################
            cpuCore=int(mpp.cpu_count())
            if __name__ == '__main__':         
                   # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"       
                    start = time.time()
                    
                   
                    with Pool(processes=cpuCore) as pool:
                          import functools
                         
                          partial_gp = functools.partial(fun_gready, S=S)
                          result = pool.map(partial_gp, range(int(iteration)))
                          
                    end = time.time()
                   
                    SortedResult=sorted(result,key=lambda l:sum(l[0]))
                    
                     
                    tm1=end-start

            
            if SortedResult[0][4]:
                generate_output_json_file(Lambda, SortedResult[0], S)

    
if __name__ == '__main__':
    system_file = sys.argv[1]
    iteration = sys.argv[2]
    start_Lambda= sys.argv[3]
    end_Lambda= sys.argv[4]
    step_Lambda= sys.argv[5]
    #system_file ="/Users/hamtasedghani/Desktop/large_scale/10_components/10-instance/data"   
    #system_file ="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/system_description11.json"    
    
    
    main(system_file, iteration,start_Lambda, end_Lambda, step_Lambda)