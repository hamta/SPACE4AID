
from classes.System import System
from classes.Algorithm import  RandomGreedy, HyperOpt
import matplotlib.pyplot as plt
import numpy as np
import sys
import pdb
import random
import copy
import json
import matplotlib.pyplot as plt
import time 
import pickle
from openpyxl import Workbook
import xlsxwriter
from hyperopt import hp, Trials
from datetime import datetime
import pyspark

## Method to create the pure json file from system description by removing 
# the comments lines
#   @param main_json_file Name of the file with the system description
#   @param config_folder The address of config_folder
def create_pure_json(main_json_file, config_folder):
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
   
## Method to generate json files for a list of specified Lambdas
#   @param Lambdas A list of incoming workload rate
#   @param config_folder The address of config_folder
#   @param temp_folder The address of temp_folder
#   @param bandwidth_scenarios A list of bandwidth scenario 
#   @param descriptions A list of methods
def generate_json_files (Lambdas, config_folder,temp_folder, bandwidth_scenarios, descriptions):
  
      
      for file_name in descriptions:
         
          for bandwidth_scenario in bandwidth_scenarios:
              for Lambda in Lambdas:
                     json_file=config_folder+"/"+file_name+".json"
                     with open(json_file) as f:
                       state_data= json.load(f)
                    
                     state_data["Lambda"] = Lambda
                     state_data["NetworkTechnology"]["ND2"]["Bandwidth"]=bandwidth_scenario[1]
                     
                     new_file_name=temp_folder+"/"+file_name + bandwidth_scenario[0]+str(round(Lambda,3)) + ".json"
                     with open(new_file_name, 'w') as f:
                       json.dump(state_data, f, indent=2)


            
        	 
## Method to read the result and draw the plots 
#   @param output_folder_list A list of out put files address
#   @param bandwidth_scenarios A list of bandwidth scenario 
#   @param descriptions A list of methods     
def draw_plots(output_folder_list,bandwidth_scenarios, descriptions):
  

        #output_folder_list=["/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/Output_files"]
        bandwidth_scenario="5G_4gb"
        plt.rc('ytick', labelsize=9)
        plt.rc('xtick', labelsize=9)
        for idx,output_folder in enumerate(output_folder_list):
            Lambdas=np.load(output_folder + "/Lambda_"+bandwidth_scenario  +".npy")
            RandomGreedy_best_costs=np.load(output_folder + "/RandomGreedy_best_costs_"+bandwidth_scenario+".npy")
            HyperOpt_best_costs=np.load(output_folder + "/HyperOpt_best_costs_"+bandwidth_scenario+".npy")
            Mixed_HyperOpt_best_costs=np.load(output_folder + "/mixed_best_costs_"+bandwidth_scenario+".npy")
            zip_lists = zip(Mixed_HyperOpt_best_costs, RandomGreedy_best_costs, Lambdas)
            difference=[]
            LambdaList=[]
            
            for mixed, Random_Greedy, Lambda in zip_lists:
                
                if not np.isnan(mixed) and not np.isnan(Random_Greedy):
                    difference.append((mixed-Random_Greedy)*100/Random_Greedy)
                    LambdaList.append(Lambda)
                
            if idx==0:        
                plt.plot(LambdaList,difference,label="1000 Iterations" )
            else:
                plt.plot(LambdaList,difference,label="5000 Iterations" )
            plt.xlabel("$\lambda$ (req/s)", fontsize=10)
            plt.ylabel("Cost ratio (%)", fontsize=10)
            
        #plt.bar(difference, LambdaList)
        #plt.xticks(difference,LambdaList)
        #plt.savefig(output_folder+"/cost.pdf", dpi=200)
        
        plt.legend()
        #fig = plt.figure(figsize=(4,3))
        plt.savefig(output_folder+"/cost_ratio_mixed.pdf", dpi=200)
        plt.show() 
        plt.rc('ytick', labelsize=10)
        plt.rc('xtick', labelsize=10)
        for idx, output_folder in enumerate(output_folder_list):
            Lambdas=np.load(output_folder + "/Lambda_"+bandwidth_scenario  +".npy")
            RandomGreedy_best_costs=np.load(output_folder + "/RandomGreedy_best_costs_"+bandwidth_scenario+".npy")
            HyperOpt_best_costs=np.load(output_folder + "/HyperOpt_best_costs_"+bandwidth_scenario+".npy")
            Mixed_HyperOpt_best_costs=np.load(output_folder + "/mixed_best_costs_"+bandwidth_scenario+".npy")
                
            zip_lists = zip(HyperOpt_best_costs, RandomGreedy_best_costs, Lambdas)
            difference=[]
            LambdaList=[]
           
            for Hyperopt, Random_Greedy, Lambda in zip_lists:
                if not np.isnan(Hyperopt) and not np.isnan(Random_Greedy):
                    difference.append((Hyperopt-Random_Greedy)*100/Random_Greedy)
                    LambdaList.append(Lambda)
            if idx==0:        
                plt.plot(LambdaList,difference,label="1000 Iterations" )
            else:
                plt.plot(LambdaList,difference,label="5000 Iterations" )
            
            #plt.rc('ytick', labelsize=6)
            plt.xlabel("$\lambda$ (req/s)", fontsize=10)
            plt.ylabel("Cost ratio (%)", fontsize=10)
           
        plt.legend()
        plt.savefig(output_folder+"/cost_ratio_Hyperopt.pdf", dpi=200)
        plt.show()  
        
       
        n=3
        r = np.arange(n)
        width = 0.2
        for idx, output_folder in enumerate(output_folder_list):
            
                mixed_execution_time_list=np.load(output_folder + "/mixed_execution_time_" + bandwidth_scenario +".npy")
                mixed_execution_time=sum(mixed_execution_time_list)/len(Lambdas)
                RandomGreedy_execution_time_list=np.load(output_folder + "/RandomGreedy_execution_time_" + bandwidth_scenario+".npy")
                RandomGreedy_execution_time=sum(RandomGreedy_execution_time_list)/len(Lambdas)
                HyperOpt_execution_time_list=np.load(output_folder + "/HyperOpt_execution_time_" + bandwidth_scenario+".npy")
                HyperOpt_execution_time=sum(HyperOpt_execution_time_list)/len(Lambdas)
                #x2=[RandomGreedy_execution_time,HyperOpt_execution_time, mixed_execution_time]
                if idx==0:
                    plt.bar(r,[RandomGreedy_execution_time,HyperOpt_execution_time,mixed_execution_time], label="1000 Iterations" ,width=0.2)
                else:
                    plt.bar(r+width,[RandomGreedy_execution_time,HyperOpt_execution_time,mixed_execution_time], label="5000 Iterations" ,width=0.2)
                  
        plt.xticks(r + width/2,["RandomGreedy", "HyperOpt", "Mixed"])
        plt.xlabel('Methods', fontsize=10)
        plt.ylabel('Average running time (s)', fontsize=10)
        plt.yscale('log')
        plt.legend()
        plt.savefig(output_folder+"/execution_time.pdf", dpi=200)
        plt.show()
       
       

     
## Method to run random greedy, HyperOpt and hybrid method and save the result
#   @param bandwidth_scenarios A list of bandwidth scenario 
#   @param iteration_number The number of iterations
#   @param temp_folder The address of temp_folder
#   @param output_folder The address of output_folder
#   @param Lambdas A list of incoming workload rate
#   @param descriptions A list of methods     
def mixed_RandomGreedy_HyperOpt(bandwidth_scenarios, iteration_number, 
                        temp_folder,output_folder, Lambda_list, descriptions):
    
    for file_name in descriptions:   
        for bandwidth_scenario in bandwidth_scenarios: 
              
              Lambdas=[]
              RandomGreedy_best_costs=[]
              mixed_best_costs=[]
              HyperOpt_best_costs=[]
            
              RandomGreedy_best_solutions=[]
              mixed_best_solutions=[]
              HyperOpt_best_solutions=[]
              
              
              RandomGreedy_execution_time=[]
              mixed_execution_time=[]
              HyperOpt_execution_time=[]
             
              round_num=3
              for Lambda in Lambda_list:
                    print("\n Lambda="+str(Lambda)+"\n")
                   
                  
                       
                    system_file=temp_folder + "/"+file_name + bandwidth_scenario[0]+str(round(Lambda,round_num)) + ".json"
                    a_file = open(system_file, "r")
                    
                    json_object = json.load(a_file)
                    a_file.close()
                    json_object["Lambda"] = Lambda
                    
                   
                    a_file = open(system_file, "w")
                    json.dump(json_object, a_file, indent=4)
                    
                    
                    a_file.close()
                    S = System(system_file)
                 
                    start=time.time()
                    Hyp=HyperOpt(S)
                  
                    new_HyperOpt_minimum_cost, new_HyperOpt_best_solution =Hyp.random_hyperopt(iteration_number)
                    HyperOpt_execution_time.append(time.time()-start)
                   
                    
                    start=time.time()
                    GA=RandomGreedy(S)
                    # if seed=None, it generate random numbers by np.random 
                    seed=int(Lambda*(100**round_num))
                    random_greedy_result=GA.random_greedy(seed, MaxIt=iteration_number)
                    new_RandomGreedy_minimum_cost=random_greedy_result[2][1]
                    new_RandomGreedy_best_solution=random_greedy_result[2][0]
                    primary_best_solution=random_greedy_result[1][0]
                   
                    RandomGreedy_solutions=random_greedy_result[0]
                    res_parts_random_list, VM_numbers_random_list, CL_res_random_list =random_greedy_result[3]
                    RandomGreedy_execution_time.append(time.time()-start)
                    pdb.set_trace()
                    print("\n RandomGreedy_minimum_cost="+str(new_RandomGreedy_minimum_cost)+"\n")
                    pdb.set_trace()
                    vals_list=Hyp.creat_trials_by_RandomGreedy(RandomGreedy_solutions, res_parts_random_list, VM_numbers_random_list, CL_res_random_list)
                    start=time.time()
                    new_mixed_minimum_cost, new_mixed_best_solution =Hyp.random_hyperopt(iteration_number, vals_list)
                    mixed_execution_time.append(time.time()-start)
                   
                    Lambdas.append(Lambda)
                    if new_mixed_minimum_cost<float("inf") and new_mixed_best_solution!= None:
                        
                        mixed_best_solutions.append(new_mixed_best_solution)
                        mixed_best_costs.append(new_mixed_minimum_cost)
                    else:
                        mixed_best_solutions.append(np.nan)
                        mixed_best_costs.append(np.nan)
                    
                    
                    if new_RandomGreedy_minimum_cost<float("inf") and new_RandomGreedy_best_solution!= None:
                       
                        RandomGreedy_best_solutions.append(new_RandomGreedy_best_solution)
                        RandomGreedy_best_costs.append(new_RandomGreedy_minimum_cost)
                    else:
                        RandomGreedy_best_solutions.append(np.nan)
                        RandomGreedy_best_costs.append(np.nan)
                        
                    if new_HyperOpt_minimum_cost<float("inf") and new_HyperOpt_best_solution!= None:
                        HyperOpt_best_solutions.append(new_HyperOpt_best_solution)
                        HyperOpt_best_costs.append(new_HyperOpt_minimum_cost)
                    else:
                        HyperOpt_best_solutions.append(np.nan)
                        HyperOpt_best_costs.append(np.nan)
                        
                   
                   
                    
           
              np.save(output_folder+ "/RandomGreedy_best_solutions_"+ bandwidth_scenario[0], RandomGreedy_best_solutions, allow_pickle=True)
              np.save(output_folder+ "/mixed_best_solutions_"+ bandwidth_scenario[0], mixed_best_solutions, allow_pickle=True)
              np.save(output_folder+ "/HyperOpt_best_solutions_"+ bandwidth_scenario[0], HyperOpt_best_solutions, allow_pickle=True)
              
              
              np.save(output_folder + "/mixed_best_costs_" + bandwidth_scenario[0], mixed_best_costs)
              np.save(output_folder + "/RandomGreedy_best_costs_" + bandwidth_scenario[0], RandomGreedy_best_costs)
              np.save(output_folder + "/HyperOpt_best_costs_" + bandwidth_scenario[0], HyperOpt_best_costs)
              
              np.save(output_folder + "/mixed_execution_time_" + bandwidth_scenario[0], mixed_execution_time)
              np.save(output_folder + "/RandomGreedy_execution_time_" + bandwidth_scenario[0], RandomGreedy_execution_time)
              np.save(output_folder + "/HyperOpt_execution_time_" + bandwidth_scenario[0], HyperOpt_execution_time)
              
              np.save(output_folder + "/Lambda_"+ bandwidth_scenario[0], Lambdas)
                     









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
    create_pure_json(random_greedy_description, config_folder)
    
    
    #descriptions=["RandomGreedy", "OnlyEdge", "OnlyCloud"]  
    descriptions=["RandomGreedy"]
    generate_json_files (Lambdas,config_folder, temp_folder, bandwidth_scenarios, descriptions) 
    
    mixed_RandomGreedy_HyperOpt( bandwidth_scenarios, iteration_number, temp_folder,output_folder, Lambdas , descriptions)
      
    #draw_plots(output_folder,bandwidth_scenarios, descriptions)
      
if __name__ == '__main__':
    
    
    temp_folder=sys.argv[1]   # address of temp files' folder
    config_folder=sys.argv[2]  # address of config folder
    output_folder=sys.argv[3]   # address of output folder
  
    # temp_folder="/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/Temp_files"
    # config_folder="/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/ConfigFiles"
    # # # output_folder2="/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/Output_files/Output_Files-5000Iterations"
    # #output_folder="/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/Output_files"
    # # output_folder1="/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/Output_files/Output_Files"
    # output_folder2="/Users/hamtasedghani/Desktop/USB/Output_Files-5000Iterations/Newoutput-without createValTime"
    # output_folder1="/Users/hamtasedghani/Desktop/USB/Output_Files-1000Iterations/Newoutput-without createValTime"
    # output_folder=[output_folder1,output_folder2]
    
    main( temp_folder,config_folder,output_folder)
    