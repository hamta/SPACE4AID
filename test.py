from classes.System import System
from classes.Algorithm import  RandomGreedy, Tabu_Search, Simulated_Annealing
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
def TabuSearch_run(bandwidth_scenarios, iteration_number, 
                        temp_folder,output_folder, Lambda_list, descriptions, seed):
    
    for file_name in descriptions:   
        for bandwidth_scenario in bandwidth_scenarios: 
          
              round_num=3
              method_list=["random", "best"]
              Lambda=0.1
              for method in method_list:
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
                    seed=2
                    iteration_number_RG=10
                    max_iterations=100
                   # GA=RandomGreedy(S,2)
                    #random_greedy_result=GA.random_greedy( MaxIt=iteration_number_RG)
                    # initial_solution=random_greedy_result[1].elite_results[0].solution
                    # initial_cost=random_greedy_result[1].elite_results[0].solution.objective_function(S)
                    pdb.set_trace()
                    # 
                    #x=GA.change_component_placement(initial_solution)
                    # y=GA.get_partitions_with_j(initial_solution,x[0])
                    
                    
                    
                    tabu_memory=500
                    #pdb.set_trace()
                    start=time.time()
                    TS_Solid= Tabu_Search(iteration_number_RG,seed,tabu_memory, max_iterations, min_score=None,system=S)
                    random_gready_time=time.time()-start
                    best_solution_Solid, best_cost_solid,current_cost_list,best_cost_list, time_list=TS_Solid.run(method=method)
                    time_list = [x - start for x in time_list]
                    time_list.insert(0, 0)
                    best_cost_list.insert(0,best_cost_list[0])
                    current_cost_list.insert(0,current_cost_list[0])
                    
                        
                  
                    
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
                    np.save(output_folder + "/Tabu_"+ method +'_iterations.npy',it_list) 
                    np.save(output_folder + "/Tabu_"+ method +'_current_cost.npy',current_cost_list) 
                    np.save(output_folder + "/Tabu_"+ method +'_best_cost.npy',best_cost_list) 
                    np.save(output_folder + "/Tabu_"+ method +'_time.npy',time_list) 
                    
                    plt.plot(it_list[:12],current_cost_list[:12],label="Current cost" )
                    plt.plot(it_list[:12],best_cost_list[:12],label="Best cost" )
                    plt.xlabel("Max iterations", fontsize=10)
                    plt.ylabel("Cost", fontsize=10)
                    plt.legend()
                    plt.title("RG Iter= "+ str(iteration_number_RG)+ ", Neighboring: " +method )
                    plt.show()
                    
                    
                    plt.plot(time_list[:12],current_cost_list[:12],label="Current cost" )
                    plt.plot(time_list[:12],best_cost_list[:12],label="Best cost" )
                    plt.xlabel("Time (s)", fontsize=10)
                    plt.ylabel("Cost ($)", fontsize=10)
                    
                    plt.title("RG Iter= "+ str(iteration_number_RG)+ ", Neighboring: " +method )
                    #plt.plot(time_list[0],label="Random Gready finished" )
                    plt.axvline(x=random_gready_time,color='k', linestyle='--', label="Random Gready finished")
                    plt.legend()
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
                    
def SimulatedAnealing_run(bandwidth_scenarios, iteration_number, 
                        temp_folder,output_folder, Lambda_list, descriptions, seed):
    
    for file_name in descriptions:   
        for bandwidth_scenario in bandwidth_scenarios: 
          
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
                    seed=2
                    iteration_number_RG=10
                    max_iterations=500
                   # GA=RandomGreedy(S,2)
                    #random_greedy_result=GA.random_greedy( MaxIt=iteration_number_RG)
                    # initial_solution=random_greedy_result[1].elite_results[0].solution
                    # initial_cost=random_greedy_result[1].elite_results[0].solution.objective_function(S)
                  
                    # 
                    #x=GA.change_component_placement(initial_solution)
                    # y=GA.get_partitions_with_j(initial_solution,x[0])
                    
                    
                   
                    start=time.time()
                    #TS_Solid= Tabu_Search(iteration_number_RG,seed,tabu_memory, max_iterations, min_score=None,system=S)
                    SA=Simulated_Annealing(iteration_number_RG,seed, 5, .99, max_iterations, min_energy=None, schedule='exponential', system=S)
                    random_gready_time=time.time()-start
                    best_solution_Solid, best_cost_solid,current_cost_list,best_cost_list, time_list=SA.run()
                    
                    time_list = [x - start for x in time_list]
                    time_list.insert(0, 0)
                    best_cost_list.insert(0,best_cost_list[0])
                    current_cost_list.insert(0,current_cost_list[0])
                    
                        
                  
                    
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
                    
                    
                    np.save(output_folder + "/SA_iterations.npy",it_list) 
                    np.save(output_folder + "/SA_current_cost.npy",current_cost_list) 
                    np.save(output_folder + "/SA_best_cost.npy",best_cost_list) 
                    np.save(output_folder + "/SA_time.npy",time_list) 
                    np.save(output_folder + "/random_gready_time.npy",random_gready_time) 
                    
                    plt.plot(it_list,current_cost_list,label="Current cost" )
                    plt.plot(it_list,best_cost_list,label="Best cost" )
                    plt.xlabel("Max iterations", fontsize=10)
                    plt.ylabel("Cost", fontsize=10)
                    plt.legend()
                    plt.title("RG Iter= "+ str(iteration_number_RG))
                    plt.show()
                    
                    
                    plt.plot(time_list[:12],current_cost_list[:12],label="Current cost" )
                    plt.plot(time_list[:12],best_cost_list[:12],label="Best cost" )
                    plt.xlabel("Time (s)", fontsize=10)
                    plt.ylabel("Cost ($)", fontsize=10)
                    
                    plt.title("RG Iter= "+ str(iteration_number_RG) )
                    #plt.plot(time_list[0],label="Random Gready finished" )
                    plt.axvline(x=random_gready_time,color='k', linestyle='--', label="Random Gready finished")
                    plt.legend()
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
                    

def run_Tabu_SA(bandwidth_scenarios, iteration_number, 
                        temp_folder,output_folder, Lambda_list, descriptions, seed):
    
    
    method_list=[ "random", "best" ]#["random", "best" ]
    Lambda=0.1
    iteration_number_RG=10          
    for file_name in descriptions:   
        for bandwidth_scenario in bandwidth_scenarios: 
          
              round_num=3
              
              for method in method_list:
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
                    seed=2
                    
                    max_iterations_Tabu=50
                   # GA=RandomGreedy(S,2)
                    #random_greedy_result=GA.random_greedy( MaxIt=iteration_number_RG)
                    # initial_solution=random_greedy_result[1].elite_results[0].solution
                    # initial_cost=random_greedy_result[1].elite_results[0].solution.objective_function(S)
                  
                    # 
                    #x=GA.change_component_placement(initial_solution)
                    # y=GA.get_partitions_with_j(initial_solution,x[0])
                    
                    
                    
                    tabu_memory=500
                    #pdb.set_trace()
                    start=time.time()
                    TS_Solid= Tabu_Search(iteration_number_RG,seed,tabu_memory, max_iterations_Tabu, min_score=None,system=S)
                    random_gready_time=time.time()-start
                    best_solution_Tabu, best_cost_Tabu,current_cost_list_Tabu,best_cost_list_Tabu, time_list_Tabu=TS_Solid.run(method=method)
                    time_list_Tabu = [x - start for x in time_list_Tabu]
                    time_list_Tabu.insert(0, 0)
                    best_cost_list_Tabu.insert(0,best_cost_list_Tabu[0])
                    current_cost_list_Tabu.insert(0,current_cost_list_Tabu[0])
                    
                    it_list_Tabu=[]
                    for i in range(len(current_cost_list_Tabu)):
                        it_list_Tabu.append(i)    
                    
                    np.save(output_folder + "/Tabu_"+ method +'_iterations.npy',it_list_Tabu) 
                    np.save(output_folder + "/Tabu_"+ method +'_current_cost.npy',current_cost_list_Tabu) 
                    np.save(output_folder + "/Tabu_"+ method +'_best_cost.npy',best_cost_list_Tabu) 
                    np.save(output_folder + "/Tabu_"+ method +'_time.npy',time_list_Tabu) 
                    pdb.set_trace()
                    
                    
                    
            ################################### SA ###################################
             
        max_iterations_SA= 50 #len(it_list_Tabu)-1
        start=time.time()
        SA=Simulated_Annealing(iteration_number_RG,seed, 5, .99, max_iterations_SA, min_energy=None, schedule='exponential', system=S)
        #random_gready_time=time.time()-start
        best_solution_SA, best_cost_SA,current_cost_list_SA,best_cost_list_SA, time_list_SA=SA.run()
      
        time_list_SA = [x - start for x in time_list_SA]
        time_list_SA.insert(0, 0)
        best_cost_list_SA.insert(0,best_cost_list_SA[0])
        current_cost_list_SA.insert(0,current_cost_list_SA[0])
        
        np.save(output_folder + "/SA_iterations.npy",it_list_Tabu) 
        np.save(output_folder + "/SA_current_cost.npy",current_cost_list_SA) 
        np.save(output_folder + "/SA_best_cost.npy",best_cost_list_SA) 
        np.save(output_folder + "/SA_time.npy",time_list_SA) 
        np.save(output_folder + "/random_gready_time.npy",random_gready_time) 
        pdb.set_trace()
      
def draw_Tabu_SA(output_folder):
     method_list=["random","best"] 
     iteration_number_RG=10
     for method in method_list:
         it_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_iterations.npy')
         current_cost_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_current_cost.npy')
         best_cost_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_best_cost.npy')
         time_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_time.npy')
         
         #plt.plot(it_list_Tabu,current_cost_list_Tabu,label="Current cost " + method )
         if method=="best":
             plt.plot(time_list_Tabu[:35],best_cost_list_Tabu[:35],label="Local search best result" )
         else:
             plt.plot(time_list_Tabu[:47],best_cost_list_Tabu[:47],label="Tabu search best result " )
         plt.xlabel("Max iterations", fontsize=10)
         plt.ylabel("Cost", fontsize=10)
         
     random_gready_time= np.load(output_folder + "/random_gready_time.npy") 
    # it_list_SA=np.load(output_folder +'/SA_iterations.npy',allow_pickle=True)
    # current_cost_list_SA=np.load(output_folder + "/SA_current_cost.npy")
     best_cost_list_SA=np.load(output_folder +"/SA_best_cost.npy", allow_pickle=True)
     time_list_SA=np.load(output_folder + "/SA_time.npy", allow_pickle=True)           
     plt.title("RG Iter= "+ str(iteration_number_RG), fontsize=8)
      #plt.plot(it_list_Tabu,current_cost_list_SA,label="Current cost" )
     plt.plot(time_list_SA[:70],best_cost_list_SA[:70],label="Simulated annealing best result" )
     plt.axvline(x=random_gready_time,color='k', linestyle='--', label="Random Gready finished")
     plt.xlabel("Time", fontsize=10)
     plt.ylabel("Cost", fontsize=10)
     plt.legend()
    
     plt.show()
     
     for method in method_list:
         it_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_iterations.npy')
         current_cost_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_current_cost.npy')
         best_cost_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_best_cost.npy')
         time_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_time.npy')
         
         #plt.plot(it_list_Tabu,current_cost_list_Tabu,label="Current cost " + method )
         if method=="best":
             plt.plot(time_list_Tabu[:40],best_cost_list_Tabu[:40],label="Local search best cost" )
         else:
             plt.plot(time_list_Tabu[:50],best_cost_list_Tabu[:50],label="Tabu search best cost " )
             plt.plot(time_list_Tabu[:50],current_cost_list_Tabu[:50],label="Tabu search current cost " )
             
         plt.xlabel("Time", fontsize=10)
         plt.ylabel("Cost", fontsize=10)
     plt.legend()
    
     plt.show()

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
    
    TabuSearch_run( bandwidth_scenarios, iteration_number, temp_folder,output_folder, Lambdas , descriptions,seed)
    SimulatedAnealing_run(bandwidth_scenarios, iteration_number, temp_folder,output_folder, Lambdas , descriptions,seed) 
    #run_Tabu_SA(bandwidth_scenarios, iteration_number, temp_folder,output_folder, Lambdas , descriptions,seed) 
    #draw_plots(output_folder,bandwidth_scenarios, descriptions)
    draw_Tabu_SA(output_folder) 
   
              
    
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
    output_folder="/Users/hamtasedghani/space4ai-d/OutputFiles_Tabu_SA"
    seed=2
    json_file="system_description_for_tabu"
    #json_file="RandomGreedy"
    #json_file="system_description1"
    main( temp_folder,config_folder,output_folder,seed,json_file)
    
