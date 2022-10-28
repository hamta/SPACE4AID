from classes.System import System
from classes.Algorithm import RandomGreedy, Tabu_Search, Simulated_Annealing, Genetic_algorithm
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import random
import copy
import json
import matplotlib.pyplot as plt
import time
#from json_file_generator_LargeScale.generate_json_partition_version import generate_system
from multiprocessing import Pool
import functools
import warnings
from os import path

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
def TabuSearch_run(S,iteration_number_RG, max_iterations,
                        output_folder, seed,Max_time_RG, Max_time, Lambda, K=1, besties_RG=None):



              method_list= [ "best"] #["random", "best"]

              for method in method_list:

                  #  start=time.time()
                   # proc = mpp.current_process()
                   # pid = proc.pid
                   # seed=seed*pid

                   # GA=RandomGreedy(S,2)
                    #random_greedy_result=GA.random_greedy( MaxIt=iteration_number_RG)
                    # initial_solution=random_greedy_result[1].elite_results[0].solution
                    # initial_cost=random_greedy_result[1].elite_results[0].solution.objective_function(S)

                    #
                    #x=GA.change_component_placement(initial_solution)
                    # y=GA.get_partitions_with_j(initial_solution,x[0])



                    tabu_memory=50
                    #pdb.set_trace()
                    start=time.time()

                    TS_Solid= Tabu_Search(iteration_number_RG,seed,system=S,Max_time_RG=Max_time_RG,K=K, besties_RG=besties_RG)
                    result,best_sol_info, Starting_points_info=TS_Solid.run_TS (method, tabu_memory, min_score=None, max_steps=max_iterations,Max_time=Max_time)

                    best_best_cost, best_best_solution, best_current_cost_list, best_best_cost_list, best_time_list = best_sol_info
                    Starting_point_solutions, Starting_point_best_costs, Starting_point_current_cost_list, Starting_point_best_cost_list, Starting_point_time_list = Starting_points_info

                    TS_time=time.time()-start
                   # time_list = [x - start for x in time_list]
                    #time_list.insert(0, 0)
                    #best_cost_list.insert(0,best_cost_list[0])
                    #current_cost_list.insert(0,current_cost_list[0])




                    # TS_Heurispy=TabuSearchHeurispy(S,seed,iteration_number_RG)
                    # memory_space=[50]
                    # max_search_without_improvement=[10]
                    # repetitions=1
                    # best_result_Heurispy,current_cost_list,best_cost_list=TS_Heurispy.main_tabu_search(max_iterations,memory_space,max_search_without_improvement,repetitions)
                    # best_solution_tabu=best_result_Heurispy["mejor_solucion"].Y_hat
                    # best_cost_tabu=best_result_Heurispy["f_mejor_solucion"]



                    #it_list=[]
                    #for i in range(len(current_cost_list)):
                     #   it_list.append(i)

                    #best_cost_list.append(best_cost_tabu)
                    #it_list.append(it+1)
                    result.print_result(S, solution_file=output_folder + "/"+method+"_Tabu_" + str(round(float(Lambda), 5))+".json")
                    np.save(output_folder + "/"+method+"_Tabu_starting_points_best_costs_" +str(round(float(Lambda), 5))+".npy",Starting_point_best_costs)
                    np.save(output_folder + "/"+method+"_Tabu_starting_points_best_cost_list_" +str(round(float(Lambda), 5))+".npy",Starting_point_best_cost_list)
                    np.save(output_folder + "/"+method+"_Tabu_starting_points_solutions_" + str(round(float(Lambda), 5))+".npy",Starting_point_solutions ,allow_pickle=True)
                    np.save(output_folder + "/"+method+"_Tabu_starting_points_current_cost_" + str(round(float(Lambda), 5))+".npy",Starting_point_current_cost_list )
                    np.save(output_folder + "/"+method+"_Tabu_starting_points_time_list_" +str(round(float(Lambda), 5))+".npy",Starting_point_time_list)

                    np.save(output_folder + "/"+method+"_Tabu_best_best_cost_" + str(round(float(Lambda), 5))+".npy",best_best_cost )
                    np.save(output_folder + "/"+method+"_Tabu_best_best_solution_" + str(round(float(Lambda), 5))+".npy",best_best_solution.Y_hat ,allow_pickle=True)
                    np.save(output_folder + "/"+method+"_Tabu_best_current_cost_list_" + str(round(float(Lambda), 5))+".npy",best_current_cost_list )
                    np.save(output_folder + "/"+method+"_Tabu_best_best_cost_list_" + str(round(float(Lambda), 5))+".npy",best_best_cost_list )
                    np.save(output_folder + "/"+method+"_Tabu_best_time_list_" + str(round(float(Lambda), 5))+".npy",best_time_list )
                    np.save(output_folder + "/"+method+"_Tabu_time_" + str(round(float(Lambda), 5))+".npy",TS_time )
                    np.save(output_folder + "/"+method+"_Tabu_"+str(round(float(Lambda), 5))+"_counter_obj_evaluation.npy",TS_Solid.counter_obj_evaluation)
                    return result


                    #np.save(output_folder + "/Tabu_"+ method +"_"+str(round(float(Lambda), 5))+'_iterations.npy',it_list)
                    #np.save(output_folder + "/Tabu_"+ method +"_"+str(round(float(Lambda), 5))+'_time.npy',time_list)
                    #plt.figure()
                    #plt.plot(it_list,current_cost_list,label="Current cost" )
                    #plt.plot(it_list,best_cost_list,label="Best cost" )
                    #plt.xlabel("Iterations", fontsize=10)
                    #plt.ylabel("Cost", fontsize=10)
                    #plt.legend()
                    #plt.title("RG Iter= "+ str(iteration_number_RG)+ ", Neighboring: " +method )
                    #plt.savefig("Tabu_It_cost_"+ method+".pdf")
                    #plt.show()

                    #plt.figure()
                    #plt.plot(time_list,current_cost_list,label="Current cost" )
                    #plt.plot(time_list,best_cost_list,label="Best cost" )
                    #plt.xlabel("Time (s)", fontsize=10)
                    #plt.ylabel("Cost ($)", fontsize=10)

                    #plt.title("RG Iter= "+ str(iteration_number_RG)+ ", Neighboring: " +method )
                    #plt.plot(time_list[0],label="Random Gready finished" )
                    #plt.axvline(x=random_gready_time,color='k', linestyle='--', label="Random Gready finished")
                    #plt.legend()
                    #plt.savefig("Tabu_time_cost_"+ method+".pdf")
                    #plt.show()

def SimulatedAnealing_run( S,iteration_number_RG, max_iterations,
                        output_folder, seed,Max_time_RG, Max_time,Lambda,K=1, besties_RG=None):





                    #proc = mpp.current_process()
                    #pid = proc.pid
                    #seed=seed*pid

                   # GA=RandomGreedy(S,2)
                    #random_greedy_result=GA.random_greedy( MaxIt=iteration_number_RG)
                    # initial_solution=random_greedy_result[1].elite_results[0].solution
                    # initial_cost=random_greedy_result[1].elite_results[0].solution.objective_function(S)

                    #
                    #x=GA.change_component_placement(initial_solution)
                    # y=GA.get_partitions_with_j(initial_solution,x[0])

                    temp_begin=5
                    schedule_constant=0.99

                    start=time.time()
                    #TS_Solid= Tabu_Search(iteration_number_RG,seed,tabu_memory, max_iterations, min_score=None,system=S)
                    SA_Solid= Simulated_Annealing(iteration_number_RG,seed,system=S,Max_time_RG=Max_time_RG,K=K, besties_RG=besties_RG)
                    result,best_sol_info, Starting_points_info=SA_Solid.run_SA( temp_begin, schedule_constant, max_iterations,  min_energy=None, schedule='exponential',Max_time=Max_time)


                    best_best_cost, best_best_solution, best_current_cost_list, best_best_cost_list, best_time_list = best_sol_info
                    Starting_point_solutions, Starting_point_best_costs, Starting_point_current_cost_list, Starting_point_best_cost_list, Starting_point_time_list = Starting_points_info

                    SA_time=time.time()-start


                    np.save(output_folder + "/SA_starting_points_best_costs_" +str(round(float(Lambda), 5))+".npy",Starting_point_best_costs)
                    np.save(output_folder + "/SA_starting_points_best_cost_list_" +str(round(float(Lambda), 5))+".npy",Starting_point_best_cost_list)
                    np.save(output_folder + "/SA_starting_points_solutions_" + str(round(float(Lambda), 5))+".npy",Starting_point_solutions ,allow_pickle=True)
                    np.save(output_folder + "/SA_starting_points_current_cost_" + str(round(float(Lambda), 5))+".npy",Starting_point_current_cost_list )
                    np.save(output_folder + "/SA_starting_points_time_list_" +str(round(float(Lambda), 5))+".npy",Starting_point_time_list)

                    np.save(output_folder + "/SA_best_best_cost_" + str(round(float(Lambda), 5))+".npy",best_best_cost )
                    np.save(output_folder + "/SA_best_best_solution_" + str(round(float(Lambda), 5))+".npy",best_best_solution.Y_hat, allow_pickle=True )
                    np.save(output_folder + "/SA_best_current_cost_list_" + str(round(float(Lambda), 5))+".npy",best_current_cost_list )
                    np.save(output_folder + "/SA_best_best_cost_list_" + str(round(float(Lambda), 5))+".npy",best_best_cost_list )
                    np.save(output_folder + "/SA_best_time_list_" + str(round(float(Lambda), 5))+".npy",best_time_list )
                    np.save(output_folder + "/SA_time_" + str(round(float(Lambda), 5))+".npy",SA_time )
                    np.save(output_folder + "/SA_"+str(round(float(Lambda), 5))+"_counter_obj_evaluation.npy",SA_Solid.counter_obj_evaluation)


                    '''time_list = [x - start for x in time_list]
                    time_list.insert(0, 0)
                    best_cost_list.insert(0,best_cost_list[0])
                    current_cost_list.insert(0,current_cost_list[0])


                    it_list=[]
                    for i in range(len(current_cost_list)):
                        it_list.append(i)'''

                    #best_cost_list.append(best_cost_tabu)
                    #it_list.append(it+1)


                    #plt.figure()
                    #plt.plot(time_list,current_cost_list,label="Current cost" )
                    #plt.plot(time_list,best_cost_list,label="Best cost" )
                    #plt.xlabel("Time (s)", fontsize=10)
                    #plt.ylabel("Cost ($)", fontsize=10)

                    #plt.title("RG Iter= "+ str(iteration_number_RG) )
                    #plt.plot(time_list[0],label="Random Gready finished" )
                    #plt.axvline(x=random_gready_time,color='k', linestyle='--', label="Random Gready finished")
                    #plt.legend()
                    #plt.savefig("SA_time_cost.pdf")
                    #plt.show()

                    ## GA=RandomGreedy(S)


def GeneticAlgorithm_run(S,iteration_number_RG, max_iteration_number,output_folder,seed,
                         K_init_population,Max_time_RG, Max_time, Lambda, besties_RG=None):

    mutation_rate = 0.7
    crossover_rate = 0.5
    start=time.time()
    GA = Genetic_algorithm(iteration_number_RG,seed, crossover_rate, mutation_rate, max_iteration_number, S,Max_time_RG=Max_time_RG,Max_time=Max_time, besties_RG=besties_RG)

    result, best_sol_cost_list_GA, time_list_GA, population=GA.run_GA(K_init_population)
    GA_time=time.time()-start
    best_sol=result.solution
    best_cost=result.objective_function(S)
    it_list=[]
    time_list_GA = [x - start for x in time_list_GA]
    time_list_GA.insert(0, 0)
    best_sol_cost_list_GA.insert(0,best_sol_cost_list_GA[0])
    for i in range(len(best_sol_cost_list_GA)):
        it_list.append(i)
    np.save(output_folder + "/GA_"+str(round(float(Lambda), 5))+"_iterations.npy",it_list)
    np.save(output_folder + "/GA_"+str(round(float(Lambda), 5))+"_time.npy",GA_time)
    np.save(output_folder + "/GA_"+str(round(float(Lambda), 5))+"_best_cost_list.npy",best_sol_cost_list_GA)
    np.save(output_folder + "/GA_"+str(round(float(Lambda), 5))+"_best_solution.npy",best_sol, allow_pickle=True)
    np.save(output_folder + "/GA_"+str(round(float(Lambda), 5))+"_best_cost.npy",best_cost)
    np.save(output_folder + "/GA_"+str(round(float(Lambda), 5))+"_time.npy",time_list_GA)
    np.save(output_folder + "/GA_"+str(round(float(Lambda), 5))+"_counter_obj_evaluation.npy",GA.counter_obj_evaluation)
    np.save(output_folder + "/RG_GA_"+str(round(float(Lambda), 5))+"_time.npy",time_list_GA[1])
    #plt.figure()
    #plt.plot(it_list,best_sol_cost_list_GA,label="Best cost" )
    #plt.xlabel("Iterations", fontsize=10)
    #plt.ylabel("Cost", fontsize=10)
    #plt.legend()
    #plt.title("RG Iter= "+ str(iteration_number_RG))
    #plt.savefig("GA_it_cost.pdf")

    #plt.figure()
    #plt.plot(time_list_GA,best_sol_cost_list_GA,label="Best cost" )
    #plt.xlabel("Time (s)", fontsize=10)
    #plt.ylabel("Cost ($)", fontsize=10)
    #plt.legend()
    #plt.title("RG Iter= "+ str(iteration_number_RG))
    #plt.savefig("GA_time_cost.pdf")
    #np.save(output_folder + "/random_gready_time.npy",random_gready_time)

def Random_Greedy_run(S,iteration_number_RG,output_folder,seed,Max_time_RG,Lambda,description,K=1):


    RG=RandomGreedy(S,seed)
    best_result_no_update, elite, random_params=RG.random_greedy(K=K,MaxIt = iteration_number_RG, MaxTime= Max_time_RG)

    RG_cost=elite.elite_results[0].solution.objective_function(S)
    RG_solution=elite.elite_results[0].solution
    elite.elite_results[0].print_result(S, solution_file=output_folder + "/"+ description + "_random_greedy_" + str(round(float(Lambda), 5))+".json")
    np.save(output_folder + "/"+ description + "_random_greedy_" + str(round(float(Lambda), 5))+".npy",Max_time_RG)
    np.save(output_folder + "/"+ description + "_random_greedy_cost_" +str(round(float(Lambda), 5))+".npy",RG_cost)
    np.save(output_folder + "/"+ description + "_random_greedy_solution_" + str(round(float(Lambda), 5))+".npy",RG_solution.Y_hat ,allow_pickle=True)
    elite_sol=[]
    if len(elite.elite_results)<K:
        K=len(elite.elite_results)
    for i in range(K):
        elite_sol.append(elite.elite_results[i].solution)
    np.save(output_folder + "/"+ description + "_random_greedy_K_best_solution_" + str(round(float(Lambda), 5))+".npy",elite_sol ,allow_pickle=True)
    print("\n RG_"+ description+ "_cost: " + str(RG_cost))
    return elite_sol

def draw_Tabu_SA_GA(system_file,output_folder,iteration_number_RG):
     method_list=["random","best"]
     plt.figure()
     for method in method_list:
         #plt.plot(it_list_Tabu,current_cost_list_Tabu,label="Current cost " + method )
         if method=="best":
             it_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_iterations.npy')
             current_cost_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_current_cost_list.npy')
             best_cost_list_Tabu_best=np.load(output_folder + "/Tabu_"+ method +'_best_cost_list.npy')
             time_list_Tabu_best=np.load(output_folder + "/Tabu_"+ method +'_time.npy')
             plt.plot(time_list_Tabu_best,best_cost_list_Tabu_best,label="Local search best result" )
         else:
             it_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_iterations.npy')
             current_cost_list_Tabu=np.load(output_folder + "/Tabu_"+ method +'_current_cost_list.npy')
             best_cost_list_Tabu_random=np.load(output_folder + "/Tabu_"+ method +'_best_cost_list.npy')
             time_list_Tabu_random=np.load(output_folder + "/Tabu_"+ method +'_time.npy')
             plt.plot(time_list_Tabu_random,best_cost_list_Tabu_random,label="Tabu search best result " )
         #plt.xlabel("Max iterations", fontsize=10)
         #plt.ylabel("Cost", fontsize=10)

     random_gready_time= np.load(output_folder + "/random_gready_time.npy")
    # it_list_SA=np.load(output_folder +'/SA_iterations.npy',allow_pickle=True)
    # current_cost_list_SA=np.load(output_folder + "/SA_current_cost.npy")
     best_cost_list_SA=np.load(output_folder +"/SA_best_cost_list.npy", allow_pickle=True)
     time_list_SA=np.load(output_folder + "/SA_time.npy", allow_pickle=True)
     best_cost_list_GA=np.load(output_folder +"/GA_best_cost_list.npy", allow_pickle=True)
     best_sol_GA=np.load(output_folder +"/GA_best_solution.npy", allow_pickle=True)
     from classes.Solution import Result, Configuration
     with open(system_file, "r") as a_file:
         json_object = json.load(a_file)

     S = System(system_json=json_object)
     sol = Configuration(best_sol_GA)
     result = Result()
     result.solution=sol
     feasible = result.check_feasibility(S)

     time_list_GA=np.load(output_folder + "/GA_time.npy", allow_pickle=True)
     total_RG_time=np.load(output_folder + "/total_random_greedy_time.npy")
     total_RG_cost=np.load(output_folder + "/total_random_greedy_cost.npy")
      #plt.plot(it_list_Tabu,current_cost_list_SA,label="Current cost" )
     plt.plot(time_list_SA,best_cost_list_SA,label="Simulated annealing best result" )
     plt.plot(time_list_GA,best_cost_list_GA,label="Genetic Algorithm best result" )
     plt.axvline(x=random_gready_time,color='k', linestyle='--', label="Random Gready finished")
     plt.axhline(y=total_RG_cost,color='y', linestyle='--', label="Random Gready totally")
     plt.title("RG Iter= "+ str(iteration_number_RG), fontsize=8)
     plt.xlabel("Time", fontsize=10)
     plt.ylabel("Cost", fontsize=10)
     plt.legend()
     plt.savefig("All1.pdf")
     #plt.show()

def draw_ratio(output):
    N_list=[7,10, 15]
    for N in N_list:
        plt.figure()
        output_folder = output + str(N) + "Components"
       # output_folder_mixed=output+"/large_scale_5000ItRG_1hHeuristices/"+str(N)+"components"
        method_list=["GA", "SA", "TS", "LS", "Total_RG"]
        for idx,method in enumerate(method_list):
            Lambdas=[]
            RandomGreedy_best_costs=[]
            GA_best_costs=[]
            Mixed_GA_best_costs=[]
            method_idx=method_list.index(method)
            if method == "TS":
                method = "random_Tabu"
            if method == "LS":
                method = "best_Tabu"
            for Lambda in np.arange(0.01,0.15,0.02):
                Lambdas.append(round(Lambda,5))
                RG_cost=0
                mixed_cost=0
                valid_RG=0
                valid_mixed=0
                for ins in range(10):
                   #if ins>6:
                   #    RG= np.load(output_folder + "/instance"+str(ins+1)+ "/random_greedy_cost_60s.npy")
                   #    RG_cost+=RG
                   #else:
                    try:

                        RG= np.load(output_folder + "/Ins"+str(ins+1)+ "/half_exec_time_random_greedy_cost_" + str(round(float(Lambda), 5))+".npy")
                        RG_cost+=RG
                        valid_RG+=1
                    except:
                        print("RG: there is not Lambda= " + str(Lambda) + " in instance " + str(ins))
                    try:
                        if method=="GA":
                             mixed=np.load(output_folder + "/Ins"+str(ins+1)+ "/"+ method+"_"+str(round(float(Lambda), 5))+"_best_cost.npy")
                             mixed_cost+= mixed
                             valid_mixed+=1
                        elif  method=="Total_RG":
                            mixed= np.load(output_folder + "/Ins"+str(ins+1)+ "/total_exec_time_random_greedy_cost_"+ str(round(float(Lambda), 5))+".npy")
                            mixed_cost+= mixed
                            valid_mixed+=1
                        else:
                            mixed= np.load(output_folder + "/Ins"+str(ins+1)+ "/" + method +"_best_best_cost_"+ str(round(float(Lambda), 5))+".npy")
                            mixed_cost+= mixed
                            valid_mixed+=1

                    except:
                          print(method+ " there is not Lambda= " + str(Lambda) + " in instance " + str(ins))

                RandomGreedy_best_costs.append(RG_cost/valid_RG)
                Mixed_GA_best_costs.append(mixed_cost/valid_mixed)
                zip_lists = zip(Mixed_GA_best_costs, RandomGreedy_best_costs, Lambdas)
            difference=[]
            LambdaList=[]

            for mixed, Random_Greedy, Lambda in zip_lists:

                if not np.isnan(mixed) and not np.isnan(Random_Greedy):
                    difference.append((mixed-Random_Greedy)*100/Random_Greedy)
                    LambdaList.append(Lambda)

            #plt.figure()
            plt.plot(LambdaList,difference,label=method_list[method_idx] )

            plt.xlabel("$\lambda$ (req/s)", fontsize=10)
            plt.ylabel("Cost ratio (%)", fontsize=10)
            # (Mixed, RandomGreedy)

        #plt.bar(difference, LambdaList)
        #plt.xticks(difference,LambdaList)
        #plt.savefig(output_folder+"/cost.pdf", dpi=200)


            #fig = plt.figure(figsize=(4,3))
            plt.rc('ytick', labelsize=10)
            plt.rc('xtick', labelsize=10)
            plt.legend()
            #plt.savefig(output_folder+"/cost_ratio_mixed_"+method+".pdf", dpi=200)
            #plt.show()

        plt.savefig(output_folder+"/all.pdf", dpi=200)

def draw_winner_methods_comparision(output):
    plt.figure()
    N_list=[7, 10, 15]
    width = 0.2
    N_total=np.arange(start=0, stop=len(N_list)*2, step=2)
    diff=0
    GA_count_list=[]
    SA_count_list=[]
    TS_count_list=[]
    LS_count_list=[]
    RG_count_list=[]
    for N in N_list:

        output_folder=output+str(N)+"Components"

        counter_list=np.full((5), 0, dtype=int)
        for ins in range(10):

            for Lambda in np.arange(0.01,0.15,0.02):
               #if ins>6:
               #    RG= np.load(output_folder + "/instance"+str(ins+1)+ "/random_greedy_cost_60s.npy")
               #    RG_cost+=RG
               #else:

                RG= np.load(output_folder + "/Ins"+str(ins+1)+ "/half_exec_time_random_greedy_cost_" +str(round(float(Lambda), 5))+".npy")
                RG_total= np.load(output_folder + "/Ins"+str(ins+1)+ "/total_exec_time_random_greedy_cost_" +str(round(float(Lambda), 5))+".npy")
                GA=np.load(output_folder + "/Ins"+str(ins+1)+ "/GA_"+str(round(float(Lambda), 5))+"_best_cost.npy")
                SA=np.load(output_folder + "/Ins"+str(ins+1)+ "/SA_best_best_cost_"+str(round(float(Lambda), 5))+".npy")
                TS=np.load(output_folder + "/Ins"+str(ins+1)+ "/random_Tabu_best_best_cost_"+str(round(float(Lambda), 5))+".npy")
                LS=np.load(output_folder + "/Ins"+str(ins+1)+ "/best_Tabu_best_best_cost_"+str(round(float(Lambda), 5))+".npy")
                cost_list=[float(GA), float(SA), float(TS), float(LS), float(RG_total)]
                min_value = min(cost_list)
                min_val_idx = [i for i in range(len(cost_list)) if cost_list[i]==min_value]
                if len(min_val_idx)>1:
                    if len(min_val_idx)<=4 and cost_list[min_val_idx[0]]<RG:
                        for i in range(len(min_val_idx)):
                            counter_list[min_val_idx[i]]+=1
                else:
                    counter_list[min_val_idx[0]]+=1

            #plt.figure()

        #ax = plt.subplot(111)
        GA_count_list.append(counter_list[0])
        SA_count_list.append(counter_list[1])
        TS_count_list.append(counter_list[2])
        LS_count_list.append(counter_list[3])
        RG_count_list.append(counter_list[4])


    plt.bar(N_total+diff-0.4, GA_count_list, width)
    plt.bar(N_total+diff-0.2, SA_count_list, width)
    plt.bar(N_total+diff,  TS_count_list, width)
    plt.bar(N_total+diff+0.2, LS_count_list, width)
    plt.bar(N_total+diff+0.4, RG_count_list, width)

    plt.xticks(N_total, [str(x) for x in N_list] )
    plt.xlabel("Number of components")
    plt.ylabel("Number of wins")
    plt.legend(["Genetic Algorithm", "Simulated Annealing", "Tabu Search","Local Search", "Total_RG"],fontsize=7)
    plt.savefig(output+'Number_of_win.pdf')
    plt.show()

def draw_counter_obj_methods_comparision(output):
    plt.figure()
    N_list=[7, 10, 15]
    width = 0.2
    N_total=np.arange(start=0, stop=len(N_list)*2, step=2)
    diff=0
    GA_count_list=[]
    SA_count_list=[]
    TS_count_list=[]
    LS_count_list=[]
    for N in N_list:

        output_folder=output+str(N)+"Components"

        counter_list=np.full((4), 0, dtype=int)
        for ins in range(10):

            for Lambda in np.arange(0.01,0.15,0.02) :
               #if ins>6:
               #    RG= np.load(output_folder + "/instance"+str(ins+1)+ "/random_greedy_cost_60s.npy")
               #    RG_cost+=RG
               #else:

                #RG= np.load(output_folder + "/instance"+str(ins+1)+ "/random_greedy_cost_" +str(round(float(Lambda), 5))+".npy")
                GA=np.load(output_folder + "/Ins"+str(ins+1)+ "/GA_"+str(round(float(Lambda), 5))+"_counter_obj_evaluation.npy")
                SA=np.load(output_folder + "/Ins"+str(ins+1)+ "/SA_"+str(round(float(Lambda), 5))+"_counter_obj_evaluation.npy")
                TS=np.load(output_folder + "/Ins"+str(ins+1)+ "/random_Tabu_"+str(round(float(Lambda), 5))+"_counter_obj_evaluation.npy")
                LS=np.load(output_folder + "/Ins"+str(ins+1)+ "/best_Tabu_"+str(round(float(Lambda), 5))+"_counter_obj_evaluation.npy")
                counter_list=[float(GA), float(SA), float(TS), float(LS)]

        GA_count_list.append(counter_list[0])
        SA_count_list.append(counter_list[1])
        TS_count_list.append(counter_list[2])
        LS_count_list.append(counter_list[3])

    plt.bar(N_total+diff-0.4, GA_count_list, width)
    plt.bar(N_total+diff-0.2, SA_count_list, width)
    plt.bar(N_total+diff,  TS_count_list, width)
    plt.bar(N_total+diff+0.2, LS_count_list, width)

    plt.xticks(N_total, [str(x) for x in N_list] )
    plt.xlabel("Number of components")
    plt.ylabel("Counter of objective function evaluation")
    plt.yscale("log")
    plt.legend(["Genetic Algorithm", "Simulated Annealing", "Tabu Search","Local Search"],fontsize=7)
    plt.savefig(output+'Counter_obj.pdf')
    plt.show()



def main(system_file,iteration_number_RG, max_iteration_number,
                        temp_folder,output_folder,K_init_population,large_scale=False):

    Max_time_RG=6
    Max_time=60
    K_best=10

    if large_scale:
        comps=[10]
        for N in comps:
            for Lambda in np.arange(0.05,0.1,0.05):
                for ins in range(9,10,1):
                    Path=output_folder+ "/large_scale/"+str(N)+"components/instance"+str(ins+1)
                    if not os.path.exists(Path):
                        os.makedirs(Path)
                    seed = ins+1
                    #system_file = generate_system(Path, seed)
                    system_file = output_folder+ "/large_scale/"+str(N)+"components/instance"+str(ins+1)+"/system_description.json"
                    #draw_Tabu_SA_GA(system_file,Path,iteration_number_RG)
                    starting_points=Random_Greedy_run(system_file,iteration_number_RG,Path,seed,Max_time_RG,Lambda,K_best)
                    #TabuSearch_run(system_file,iteration_number_RG, max_iteration_number,Path, seed,Max_time_RG, Max_time,Lambda, besties_RG=starting_points)
                    SimulatedAnealing_run(system_file,iteration_number_RG, max_iteration_number,Path,seed,Max_time_RG, Max_time, Lambda,besties_RG=starting_points)
                    GeneticAlgorithm_run(system_file,iteration_number_RG, max_iteration_number,Path,seed,K_init_population,Max_time_RG, Max_time, Lambda, besties_RG=starting_points)
    else:
        seed=11
        TabuSearch_run(system_file,iteration_number_RG, max_iteration_number, temp_folder,output_folder, seed,Max_time_RG, Max_time)
        SimulatedAnealing_run(system_file,iteration_number_RG, max_iteration_number, temp_folder,output_folder,seed,Max_time_RG, Max_time)
        GeneticAlgorithm_run(system_file,iteration_number_RG, max_iteration_number, temp_folder,output_folder,seed,K_init_population,Max_time_RG, Max_time)

        draw_Tabu_SA_GA(output_folder,iteration_number_RG)

def parallel_Lambda(Ins_Lambda,iteration_number_RG, max_iteration_number,output_folder,K_init_population,N):

    Max_time_RG=120
    Max_time=60
    K_best=10
    ins=Ins_Lambda[1]
    Lambda=Ins_Lambda[0]


    Path=output_folder+str(N)+"Components/Ins"+str(ins+1)
    if not os.path.exists(Path):
        os.makedirs(Path)
    seed = ins+1
    print('\n Lambda='+str(Lambda)+ ' instance=' + str(ins+1))
    #system_file = generate_system(Path, seed)
    system_file = Path+"/system_description.json"
    with open(system_file, "r") as a_file:
         json_object = json.load(a_file)

    json_object["Lambda"] = Lambda
    S=System(system_json=json_object)
    #draw_Tabu_SA_GA(system_file,Path,iteration_number_RG)
   # starting_points=Random_Greedy_run(S,iteration_number_RG,Path,seed,Max_time_RG,Lambda,"total_exec_time",K_best)
    Max_time_RG=Max_time
    starting_points=Random_Greedy_run(S,iteration_number_RG,Path,seed,Max_time_RG,Lambda,"half_exec_time",K_best)
    result=TabuSearch_run(S,iteration_number_RG, max_iteration_number,Path, seed,Max_time_RG, Max_time,Lambda, besties_RG=starting_points)
    x=1
    #SimulatedAnealing_run(S,iteration_number_RG, max_iteration_number,Path,seed,Max_time_RG, Max_time, Lambda,besties_RG=starting_points)
    #GeneticAlgorithm_run(S,iteration_number_RG, max_iteration_number,Path,seed,K_init_population,Max_time_RG, Max_time, Lambda, besties_RG=starting_points)

def return_Ins_Lambda(Ins, start_Lambda, end_Lambda,step ):
    Instances=np.arange(0,Ins,1)
    Lambdas= np.arange(start_Lambda, end_Lambda, step)
    all_combinations = []

    list1_permutations = itertools.permutations(Lambdas, len(Instances))

    for r in itertools.product(Lambdas, Instances):
        all_combinations.append(r)
    return all_combinations




def main_parallel(iteration_number_RG, max_iteration_number,output_folder,K_init_population):
    cpuCore = int(mpp.cpu_count())
    Ins_Lambda=return_Ins_Lambda(10, 0.1, 1, 0.05)
    # Ins_Lambda=[(0, 0.45),(1,0.2), (2,0.25), (1,0.25),(1,0.4),(1,0.45),(2,0.35),(2,0.4),(2,0.45),(3,0.2),(3,0.45),(4,0.45),(5,0.2),(5,0.3),(5,0.35),(5,0.4),(5,0.45),(6,0.4),(6,0.45),(7,0.45),(8,0.45),(9,0.35),(9,0.45)]
    for N in [5]: #[7,10, 15]:


        if __name__ == '__main__':

            start = time.time()

            with Pool(processes=cpuCore) as pool:

                partial_gp = functools.partial(parallel_Lambda,iteration_number_RG=iteration_number_RG, max_iteration_number=max_iteration_number,
                                               output_folder=output_folder,K_init_population=K_init_population,N=N )

                full_result = pool.map(partial_gp, Ins_Lambda)

            end = time.time()





if __name__ == '__main__':

    output_folder= "/Users/hamtasedghani/space4ai-d/Output_Files/large_scale/" #"/Users/hamtasedghani/space4ai-d/Output_Files_without_branch_new_version_1min/large_scale/"
    system_file=output_folder+"/15Components/Ins1/system_description.json"
    iteration_number_RG=1
    Path=output_folder+str(7)+"Components/Ins"+str(1)
    seed=1
    Max_time_RG=60
    Lambda=0.08
    K_best=10
    max_iteration_number=1
    Max_time=6
    K_init_population=4*K_best
    parallel_Lambda([0.5, 3],iteration_number_RG, max_iteration_number,output_folder,K_init_population,10)
    main_parallel(iteration_number_RG, max_iteration_number,output_folder,K_init_population)
    '''with open(system_file, "r") as a_file:
         json_object = json.load(a_file)

    json_object["Lambda"] = Lambda
    S=System(system_json=json_object)
    starting_points=Random_Greedy_run(S,iteration_number_RG,Path,seed,Max_time_RG,Lambda,"_",K_best)
    TabuSearch_run(S,iteration_number_RG, max_iteration_number,Path, seed,Max_time_RG, Max_time,Lambda, besties_RG=starting_points)
    SimulatedAnealing_run(S,iteration_number_RG, max_iteration_number,Path,seed,Max_time_RG, Max_time, Lambda,besties_RG=starting_points)
    GeneticAlgorithm_run(S,iteration_number_RG, max_iteration_number,Path,seed,K_init_population,Max_time_RG, Max_time, Lambda, besties_RG=starting_points)


    # temp_folder=sys.argv[1]   # address of temp files' folder
    # config_folder=sys.argv[2]  # address of config folder
    # output_folder=sys.argv[3]   # address of output folder
    # seed=sys.argv[4]
    system_file = "ConfigFiles/Random_Greedy.json" #"ConfigFiles/large_scale.json" # "ConfigFiles/RG-MaskDetection.json"
    temp_folder="Temp_files"
    iteration_number_RG=5000
    max_iteration_number=1
    K_init_population=40

    output_folder="OutputFiles_Tabu_SA"
    large_scale=True'''
    #main( system_file,iteration_number_RG, max_iteration_number,temp_folder,output_folder,K_init_population,large_scale)
    #warnings.formatwarning = my_formatwarning
    #main_parallel(iteration_number_RG, max_iteration_number,output_folder,K_init_population)
    #draw_ratio(output_folder)
    #draw_winner_methods_comparision(output_folder)
    #draw_counter_obj_methods_comparision(output_folder)
