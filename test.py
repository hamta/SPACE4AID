from classes.System import System
from classes.Algorithm import  RandomGreedy, TabuSearchHeurispy, TabuSearchSolid
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
import multiprocessing as mpp

## Method to create the pure json file from system description by removing 
# the comments lines
#   @param main_json_file Name of the file with the system description
#   @param config_folder The address of config_folder
def create_pure_json(main_json_file, config_folder,json_file):
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
    
    pure_json=config_folder+"/"+json_file+".json"
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
                     if "ND2" in state_data["NetworkTechnology"]:
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
#   @param seed Seed for random number generation  
def mixed_RandomGreedy_HyperOpt(bandwidth_scenarios, iteration_number, 
                        temp_folder,output_folder, Lambda_list, descriptions, seed):
    
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
              Lambda_list=[0.1]
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
                    proc = mpp.current_process()
                    pid = proc.pid
                    seed=seed*pid
                    iteration_number_RG=50
                    max_iterations=10
                    GA=RandomGreedy(S)
                    random_greedy_result=GA.random_greedy(seed, MaxIt=iteration_number_RG)
                    initial_solution=random_greedy_result[1].elite_results[0].solution
                    initial_cost=random_greedy_result[1].elite_results[0].solution.objective_function(S)
                   
                    # 
                    #x=GA.change_component_placement(initial_solution)
                    # y=GA.get_partitions_with_j(initial_solution,x[0])
                    
                    
                    method="random"
                    tabu_memory=500
                    #pdb.set_trace()
                    TS_Solid= TabuSearchSolid(seed,iteration_number_RG,tabu_memory, max_iterations, min_score=None,system=S)
                    best_solution_Solid, best_cost_solid,current_cost_list,best_cost_list=TS_Solid.run(method=method)
                    
                  
                    
                    # TS_Heurispy=TabuSearchHeurispy(S,seed,iteration_number_RG)
                    # memory_space=[50]
                    # max_search_without_improvement=[10]
                    # repetitions=1
                    # best_result_Heurispy,current_cost_list,best_cost_list=TS_Heurispy.main_tabu_search(max_iterations,memory_space,max_search_without_improvement,repetitions)
                    # best_solution_tabu=best_result_Heurispy["mejor_solucion"].Y_hat
                    # best_cost_tabu=best_result_Heurispy["f_mejor_solucion"]
                    
                    
                    
                    it_list=[]
                    for i in range(len(current_cost_list)):
                        it_list.append(i)
                    
                    #best_cost_list.append(best_cost_tabu)
                    #it_list.append(it+1)
                    
                    plt.plot(it_list,current_cost_list,label="Current cost" )
                    plt.plot(it_list,best_cost_list,label="Best cost" )
                    plt.xlabel("Max iterations", fontsize=10)
                    plt.ylabel("Cost", fontsize=10)
                    plt.legend()
                    plt.title("RG Iter= "+ str(iteration_number_RG)+ ", Neighboring: " +method )
                    plt.show()
                    print("best cost: ", best_cost_list[-1])
                    print("Running time Tabu search with worst initial solution:",time.time()-start)
                    # GA=RandomGreedy(S)
                    #pdb.set_trace()
                    # random_greedy_result=GA.random_greedy(seed, MaxIt=iteration_number)
                    # TS=TabuSearch(S)
                    # x=TS.change_FaaS(random_greedy_result[1].elite_results[0].solution)
                    # new_RandomGreedy_minimum_cost=random_greedy_result[2][1]
                    # new_RandomGreedy_best_solution=random_greedy_result[2][0]
                    # primary_best_solution=random_greedy_result[1][0]
                   
                    # RandomGreedy_solutions=random_greedy_result[0]
                    # res_parts_random_list, VM_numbers_random_list, CL_res_random_list =random_greedy_result[3]
                    # RandomGreedy_execution_time.append(time.time()-start)
                   
                    # print("\n RandomGreedy_minimum_cost="+str(new_RandomGreedy_minimum_cost)+"\n")
                   
                    # #vals_list=Hyp.creat_trials_by_RandomGreedy(RandomGreedy_solutions, res_parts_random_list, VM_numbers_random_list, CL_res_random_list)
                    # start=time.time()
                    # #new_mixed_minimum_cost, new_mixed_best_solution =Hyp.random_hyperopt(seed,iteration_number, vals_list)
                    # mixed_execution_time.append(time.time()-start)
                   
                    # Lambdas.append(Lambda)
                    








def main( temp_folder,config_folder,output_folder,seed,json_file):     
     
   
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
        
    random_greedy_description=config_folder+"/"+json_file+".json"
    create_pure_json(random_greedy_description, config_folder,json_file)
    
    
    #descriptions=["RandomGreedy", "OnlyEdge", "OnlyCloud"]  
    descriptions=[json_file]
    generate_json_files (Lambdas,config_folder, temp_folder, bandwidth_scenarios, descriptions) 
    
    mixed_RandomGreedy_HyperOpt( bandwidth_scenarios, iteration_number, temp_folder,output_folder, Lambdas , descriptions,seed)
      
    #draw_plots(output_folder,bandwidth_scenarios, descriptions)
      
if __name__ == '__main__':
    
    
    # temp_folder=sys.argv[1]   # address of temp files' folder
    # config_folder=sys.argv[2]  # address of config folder
    # output_folder=sys.argv[3]   # address of output folder
    # seed=sys.argv[4]
    
    
    temp_folder="/Users/hamtasedghani/space4ai-d/Temp_files"
    config_folder="/Users/hamtasedghani/space4ai-d/ConfigFiles"
    # # output_folder2="/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/Output_files/Output_Files-5000Iterations"
    #output_folder="/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/Output_files"
    # output_folder1="/Users/hamtasedghani/Desktop/untitled folder/space4ai-d/Output_files/Output_Files"
    output_folder2="/Users/hamtasedghani/Desktop/USB/Output_Files-5000Iterations/Newoutput-without createValTime"
    output_folder1="/Users/hamtasedghani/Desktop/USB/Output_Files-1000Iterations/Newoutput-without createValTime"
    output_folder=[output_folder1,output_folder2]
    seed=2
    json_file="system_description_for_tabu"
    main( temp_folder,config_folder,output_folder,seed,json_file)
    
