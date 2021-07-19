
from classes.System import System
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
import pickle
from openpyxl import Workbook
import xlsxwriter

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
    
    pure_json=config_folder+'/RandomGreedy.json'
    filehandle = open(pure_json, "w")
    filehandle.writelines(data)
    filehandle.close()
   
    
def generate_json_files (Lambdas, config_folder,temp_folder, bandwidth_scenarios, descriptions):
  
      #json_files_name= folder_address + "/RandomGreedy_"
      for file_name in descriptions:
         
          for bandwidth_scenario in bandwidth_scenarios:
              for Lambda in Lambdas:
                     json_file=config_folder+"/"+file_name+".json"
                     with open(json_file) as f:
                       state_data= json.load(f)
                    
                     state_data["Lambda"] = Lambda
                     state_data["NetworkTechnology"]["ND2"]["Bandwidth"]=bandwidth_scenario[1]
                     # for state in state_data['states']:
                     #   del state['area_codes']
                      
                     
                     new_file_name=temp_folder+"/"+file_name + bandwidth_scenario[0]+str(round(Lambda,3)) + ".json"
                     with open(new_file_name, 'w') as f:
                       json.dump(state_data, f, indent=2)


def compute_save_result(bandwidth_scenarios, iteration_number, 
                        temp_folder,output_folder, Lambda_list, descriptions ):
    
    for file_name in descriptions:   
        for bandwidth_scenario in bandwidth_scenarios: 
        
              Lambdas=[]
              costs_table=[]
             #pdb.set_trace()
              best_solutions=[]
              loca_constraints=[]
              global_constraints=[]
              edge_utilizations=[]
              cloud_utilizations=[]
              best_costs=[]
              #data = np.loadtxt(Lambda_bound, delimiter=',')
          
              for Lambda in Lambda_list:
                  
                    system_file=temp_folder + "/"+file_name + bandwidth_scenario[0]+str(round(Lambda,3)) + ".json"
                    a_file = open(system_file, "r")
                    
                    json_object = json.load(a_file)
                    a_file.close()
                    json_object["Lambda"] = Lambda
                    
                   
                    a_file = open(system_file, "w")
                    json.dump(json_object, a_file, indent=4)
                    
                    
                    a_file.close()
                    S = System(system_file)
                 
                  
                    minimum_cost=float("inf")
                    best_solution=None
                    primary_best_solution=None
                    start=time.time()
                   
                 
                    GA=RandomGreedy(S)
                    minimum_cost, primary_best_solution, best_solution =GA.random_greedy_single_processing(iteration_number)
                    
                    
                   
                    end=time.time()
                    
                   
                   
                   
                    if primary_best_solution != None:
                        print("Execution time for ", iteration_number, " number random initial solution is:", end-start)
                        print("\n Lambda: \n", Lambda)
                        print(" \n primary_best_solution: \n", primary_best_solution.Y_hat)
                        I= len(primary_best_solution.Y_hat)
                        
                        print("\n Performance evaluation of components: \n")
                        for i in range(I):
                                #if i >3:
                                    #pdb.set_trace()
                                    lc1 = LocalConstraint(i,1000)
                                    print("component : " , i,lc1.check_feasibility(S,primary_best_solution))
                        path_idx=0            
                        for GC in S.GC:
                            #pdb.set_trace()
                            gc1=GlobalConstraint(GC[0], GC[1])
                            #performance_of_components=[comp[1][1] for comp in check_components]
                            flag, Sum=gc1.check_feasibility(S,primary_best_solution)
                            print("Path" , path_idx, ": ", GC[0], " \n Performance: ",Sum)
                            path_idx +=1
                            
                           
                        print("\n Utilization of Edge resources: \n")
                        for j in range(S.cloud_start_index):
                                #if i >3:
                                    #pdb.set_trace()
                                    res = ServerFarmPE()
                                    print("resource : " , j,res.compute_utilization(j, primary_best_solution.Y_hat, S))
                                    
                        print("\n Utilization of Cloud resources: \n")
                        for j in range(S.cloud_start_index, S.FaaS_start_index, 1):
                                #if i >3:
                                    #pdb.set_trace()
                                    res = ServerFarmPE()
                                    print("resource : " , j,res.compute_utilization( j, primary_best_solution.Y_hat, S))
                    else:
                        print("There is not any feasible solution")
                    
                    
                    if best_solution != None:
                        best_solutions.append(best_solution.Y_hat)
                        print("\n best_solution: \n", best_solution.Y_hat)
                        print("\n Performance evaluation of components: \n")
                        LocalConstraints=[]
                        for i in range(len(S.components)):
                                    lc1 = LocalConstraint(i,1000)
                                    xx=lc1.check_feasibility(S,best_solution)
                                    print("component : " , i,xx)
                                    LocalConstraints.append(xx)
                           
                        loca_constraints.append(LocalConstraints)
                        path_idx=0
                        GlobalConstraints=[]
                        for GC in S.GC:
                            #pdb.set_trace()
                            gc1=GlobalConstraint(GC[0], GC[1])
                            #performance_of_components=[comp[1][1] for comp in check_components]
                            flag, Sum=gc1.check_feasibility(S,best_solution)
                            print("Path" , path_idx, ": ", GC[0], " \n Performance: ",Sum)
                            path_idx +=1
                            GlobalConstraints.append(Sum)
                    
                        global_constraints.append(GlobalConstraints)
                        print("\n Utilization of Edge resources: \n")
                        EdgeUtilizations=[]
                        for j in range(S.cloud_start_index):
                                #if i >3:
                                    #pdb.set_trace()
                                    res = ServerFarmPE()
                                    xy=res.compute_utilization(j, best_solution.Y_hat, S)
                                    print("resource : " , j,xy)
                                    EdgeUtilizations.append(xy)
                                    
                        print("\n Utilization of Cloud resources: \n")
                        edge_utilizations.append(EdgeUtilizations)
                        CloudUtilization=[]
                        for j in range(S.cloud_start_index, S.FaaS_start_index, 1):
                                #if i >3:
                                    #pdb.set_trace()
                                    res = ServerFarmPE()
                                    yy=res.compute_utilization(j, best_solution.Y_hat, S)
                                    print("resource : " , j,yy)
                                    CloudUtilization.append(yy)
                        cloud_utilizations.append(CloudUtilization)
                        best_costs.append(minimum_cost)
                        print("Best Cost: " + str(minimum_cost))
                    else:
                      
                        print("There is not any feasible solution")
                        global_constraints.append([np.nan])
           
           
                    
                  
                   
                    if best_solution !=None:
                        #cost=best_solution.objective_function(0, S)
                        costs_table.append(minimum_cost)
                        Lambdas.append(Lambda)
                    else:
                            costs_table.append(np.nan)
                            Lambdas.append(Lambda)

             
              np.save(output_folder+ "/best_solutions_"+file_name+ bandwidth_scenario[0], best_solutions, allow_pickle=True)
              np.save(output_folder+ "/loca_constraints_"+ file_name+ bandwidth_scenario[0], loca_constraints)
              np.save(output_folder + "/global_constraints_" + file_name+ bandwidth_scenario[0], global_constraints)
              np.save(output_folder +"/edge_utilizations_"+ file_name+ bandwidth_scenario[0], edge_utilizations)
              np.save(output_folder + "/cloud_utilizations_"+ file_name+ bandwidth_scenario[0], cloud_utilizations)
              np.save(output_folder + "/best_costs_"+ file_name + bandwidth_scenario[0], costs_table)
              np.save(output_folder + "/best_costs"+file_name, best_costs)
              np.save(output_folder + "/execution_time"+file_name, end-start)
              np.save(output_folder + "/Lambda_"+ file_name+ bandwidth_scenario[0], Lambdas)
              np.save(output_folder + "/thereshould", S.GC[0][1])
          
          
                
              
        	 
     
def draw_plots(output_folder,bandwidth_scenarios, descriptions):
  
      #bandwidth_scenario="5G_4gb"
      bandwidth_scenario="4G"
     
      for file_name in descriptions:
          Lambdas=np.load(output_folder + "/Lambda_"+file_name+bandwidth_scenario  +".npy")
          best_costs=np.load(output_folder + "/best_costs_"+file_name+bandwidth_scenario+".npy")
          plt.plot(Lambdas,best_costs,label=file_name )
          plt.xlabel("$\lambda$ (req/s)", fontsize=10)
          plt.ylabel("Hourly Cost ($) ", fontsize=10)
      
          #plt.plot(Lambdas, allcost)
      plt.legend(loc=1, prop={'size': 7}) 
      plt.savefig(output_folder+"/cost.pdf", dpi=200)
      plt.show()  
     
   
      for file_name in descriptions:  
         for bandwidth_scenario in bandwidth_scenarios:
           Lambdas=np.load(output_folder + "/Lambda_" +file_name+bandwidth_scenario[0]+".npy")
           global_constraints=np.load(output_folder + "/global_constraints_" +file_name+ bandwidth_scenario[0]+".npy",allow_pickle=True)
          
           plt.plot(Lambdas,global_constraints, label =file_name +" global constraint $P_1$ under "+ bandwidth_scenario[0] )
         
           plt.xlabel("$\lambda$ (req/s)", fontsize=10)
           plt.ylabel("$\widehatR_{P_1}$ (second)", fontsize=10)
      
      th=np.load(output_folder + "/thereshould" + ".npy")
      thereshould=[th] * len(Lambdas)
      plt.plot(Lambdas, thereshould, label ='Threshold') 
      plt.legend(loc=4, prop={'size': 7}) 
      plt.savefig(output_folder+"/GlobalConstraints.pdf", dpi=200)
      plt.show()   
   
    

def main( temp_folder,config_folder,output_folder):     
     
   
    input_file=config_folder+"/Input_file.json"
    with open(input_file) as f:
            data = json.load(f)
    if "LambdaBound" in data.keys(): 
        Lambda_bound_start=data["LambdaBound"]["start"]
        Lambda_bound_end=data["LambdaBound"]["end"]
        Lambda_bound_step=data["LambdaBound"]["step"]
    else:
        print("ERROR: no LambdaBound in input file")
        sys.exit(1)
     
    bandwidth_scenarios=[]
    if "BandwidthScenario" in data.keys():
        for key in data["BandwidthScenario"]:
            bandwidth_scenarios.append((key,data["BandwidthScenario"][key]["bandwidth"]))
    else:
        print("ERROR: no BandwidthScenario in input file")
        sys.exit(1)
    
    if "IterationNumber" in data.keys():
       iteration_number=data["IterationNumber"] 
    else:
       print("ERROR: no IterationNumber in input file")
       sys.exit(1)
   
    
    Lambdas=[]
    for Lambda in np.arange(Lambda_bound_start, Lambda_bound_end, Lambda_bound_step):
        Lambdas.append(Lambda)
        
    random_greedy_description=config_folder+"/Random_Greedy.json"
    create_pure_json(random_greedy_description)
    
    
    #descriptions=["RandomGreedy", "OnlyEdge", "OnlyCloud"]  
    descriptions=["RandomGreedy"]
    generate_json_files (Lambdas,config_folder, temp_folder, bandwidth_scenarios, descriptions) 
    compute_save_result( bandwidth_scenarios, iteration_number, temp_folder,output_folder, Lambdas , descriptions)
    
    
       
      
    draw_plots(output_folder,bandwidth_scenarios, descriptions)
      
if __name__ == '__main__':
    
    
    temp_folder=sys.argv[2]   # address of temp files' folder
    config_folder=sys.argv[3]  # address of config folder
    output_folder=sys.argv[4]   # address of output folder
  
    # temp_folder="/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/Temp_files"
    # config_folder="/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/ConfigFiles"
    # output_folder="/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/Output_files"
   
    
    main( temp_folder,config_folder,output_folder)
    