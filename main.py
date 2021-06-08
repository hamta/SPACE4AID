

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
import pickle

def create_pure_json(main_json_file):
    data = []
    with open(main_json_file,"r") as fp:
        Lines = fp.readlines()
    for line in Lines:
        
        
        idx=line.find('S4AI')
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
    
def generate_json_files (pure_json_file, Lambdas, folder_address, bandwidth_scenarios):
    
     
     # main_file=sys.argv[1]
      # Lambda_bound=sys.argv[2]
      # folder_address=sys.argv[3]
      # bandwidth_scenario=sys.argv[4]
      #main_file='/Users/hamtasedghani/Desktop/SPACE4AI/ConfigFiles/system_description11.json'
      # Lambda_bound="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/Lambda_bound.txt"
      # folder_address="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/Lambda_files"
      # bandwidth_scenario=4000
      #pdb.set_trace()
      
      json_files_name= folder_address + "/system_description"
      for bandwidth_scenario in bandwidth_scenarios:
          for Lambda in Lambdas:
                
                 with open(pure_json_file) as f:
                   state_data= json.load(f)
                
                 state_data["Lambda"] = Lambda
                 state_data["NetworkTechnology"]["ND2"]["Bandwidth"]=bandwidth_scenario[1]
                 # for state in state_data['states']:
                 #   del state['area_codes']
                  
                 
                 new_file_name=json_files_name + bandwidth_scenario[0]+str(round(Lambda,3)) + ".json"
                 with open(new_file_name, 'w') as f:
                   json.dump(state_data, f, indent=2)
      return json_files_name

def compute_save_result(graph_file, bandwidth_scenarios, iteration_number, 
                        folder_address,json_files_name, Lambda_list ):
    
        
      # config_files_address=sys.argv[1]
      # Lambda_bound=sys.argv[2]
      # graph_file=sys.argv[3]
      # iteration_number=int(sys.argv[4])
      # bandwidth_scenario=sys.argv[5]
      
       # main_file='/Users/hamtasedghani/Desktop/SPACE4AI/ConfigFiles/system_description11.json'
      # Lambda_bound="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/Lambda_bound.txt"
      # config_files_address="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/Lambda_files"
      # graph_file="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/DAG_Conf.txt"
      
      #iteration_number=1000
      
     
     # bandwidth_scenario_name="5G_4gb"
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
              
                system_file=json_files_name + bandwidth_scenario[0] + str(round(Lambda,3)) + ".json"
                a_file = open(system_file, "r")
                
                json_object = json.load(a_file)
                a_file.close()
                json_object["Lambda"] = Lambda
                
               
                a_file = open(system_file, "w")
                json.dump(json_object, a_file, indent=4)
                
                
                a_file.close()
                S = System(system_file, graph_file)
             
                cost=np.full((1,2), tuple)
                costs=[]
                minimum_cost=float("inf")
                best_solution=None
                primary_best_solution=None
                start=time.time()
                N=iteration_number
                primary_solutions=[]
                solutions=[]
                for i in range(N):
                    A=Algorithm(S)
                    solution=A.conf[0]
                    
                    flag, Sum=solution.check_feasibility(S)
                   
                    #pdb.set_trace()
                    if flag:
                        primary_solutions.append(copy.deepcopy(solution.Y_hat))
                        cost[0][0]=solution.objective_function(0, S) 
                        I,J=solution.Y_hat.shape
                        primary_solution=copy.deepcopy(solution)
                        for j in range(J):
                            solution= A.reduce_cluster_size(j, solution, S)
                        
                        cost[0][1]=solution.objective_function(0, S)
                        
                        costs.append(copy.deepcopy(cost))
                        solutions.append(copy.deepcopy(solution.Y_hat))
                        
                           
                        if cost[0][1] < minimum_cost:
                                minimum_cost=cost[0][1]
                                best_solution=solution
                                primary_best_solution=primary_solution
                                flag1, Sum1=best_solution.check_feasibility(S)
                end=time.time()
                
               
               
                # S = System(system_file, graph_file)
                # A=RandomGreedy(S)
                # A.random_greedy(S,1000)
               
                if primary_best_solution != None:
                    print("Execution time for ", N, " number random initial solution is:", end-start)
                    print("\n Lambda: \n", Lambda)
                    print(" \n primary_best_solution: \n", primary_best_solution.Y_hat)
                    I,J= primary_best_solution.Y_hat.shape
                    
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
                                print("resource : " , j,res.compute_utilization(I, j, primary_best_solution.Y_hat, S.demand_matrix, S.Lambdas))
                                
                    print("\n Utilization of Cloud resources: \n")
                    for j in range(S.cloud_start_index, S.FaaS_start_index, 1):
                            #if i >3:
                                #pdb.set_trace()
                                res = ServerFarmPE()
                                print("resource : " , j,res.compute_utilization(I, j, primary_best_solution.Y_hat, S.demand_matrix, S.Lambdas))
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
                    path_idx=0    
                    loca_constraints.append(LocalConstraints)
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
                                xy=res.compute_utilization(I, j, best_solution.Y_hat, S.demand_matrix, S.Lambdas)
                                print("resource : " , j,xy)
                                EdgeUtilizations.append(xy)
                                
                    print("\n Utilization of Cloud resources: \n")
                    edge_utilizations.append(EdgeUtilizations)
                    CloudUtilization=[]
                    for j in range(S.cloud_start_index, S.FaaS_start_index, 1):
                            #if i >3:
                                #pdb.set_trace()
                                res = ServerFarmPE()
                                yy=res.compute_utilization(I, j, best_solution.Y_hat, S.demand_matrix, S.Lambdas)
                                print("resource : " , j,yy)
                                CloudUtilization.append(yy)
                    cloud_utilizations.append(CloudUtilization)
                    best_costs.append(minimum_cost)
                    print("Best Cost: " + str(minimum_cost))
                else:
                    # for GC in S.GC:
                    #     pdb.set_trace()
                    #     gc1=GlobalConstraint(GC[0], GC[1])
                    #     #performance_of_components=[comp[1][1] for comp in check_components]
                    #     flag, Sum=gc1.check_feasibility(S,best_solution)
                    #     print("Path" , path_idx, ": ", GC[0], " \n Performance: ",Sum)
                    #     path_idx +=1
                    #     GlobalConstraints.append(Sum)
                
                    # global_constraints.append(GlobalConstraints)
                    print("There is not any feasible solution")
        
       
       
                
              
                #S = System(system_file, graph_file)
                
                 # best_Y_hat=A.best_solution.Y_hat
                 # A1=RandomGreedy(S,best_Y_hat)
                #pdb.set_trace()
                if flag:
                    cost=best_solution.objective_function(0, S)
                    costs_table.append(cost)
                    Lambdas.append(Lambda)
                
         
         
          np.save(folder_address+ "/best_solutions_"+ bandwidth_scenario[0], best_solutions, allow_pickle=True)
          np.save(folder_address+ "/loca_constraints_"+  bandwidth_scenario[0], loca_constraints)
          np.save(folder_address + "/global_constraints_" +  bandwidth_scenario[0], global_constraints)
          np.save(folder_address +"/edge_utilizations_"+  bandwidth_scenario[0], edge_utilizations)
          np.save(folder_address + "/cloud_utilizations_"+  bandwidth_scenario[0], cloud_utilizations)
          np.save(folder_address + "/best_costs_"+ bandwidth_scenario[0], costs_table)
          np.save(folder_address + "/best_costs", best_costs)
          np.save(folder_address + "/execution_time", end-start)
          np.save(folder_address + "/Lambda_" + bandwidth_scenario[0], Lambdas)
          np.save(folder_address + "/thereshould", S.GC[0][1])
  
     
def draw_plots(folder_address,bandwidth_scenarios):
     #plt.title('Cost')
      #allcost=np.array(len(Lambdas),minimum_cost)
      #pdb.set_trace()
      Lambdas=np.load(folder_address + "/Lambda_" + bandwidth_scenarios[0][0] +".npy")
      best_costs=np.load(folder_address + "/best_costs_"+ bandwidth_scenarios[0][0]+".npy")
      plt.plot(Lambdas,best_costs)
      plt.xlabel("Lambda (req/s)")
      plt.ylabel("Hourly Cost ($)")
      plt.savefig("cost.pdf", dpi=200)
      #plt.plot(Lambdas, allcost)
      plt.show()        
     
     
      for bandwidth_scenario in bandwidth_scenarios:
          Lambdas=np.load(folder_address + "/Lambda_" + bandwidth_scenario[0] +".npy")
          global_constraints=np.load(folder_address + "/global_constraints_" +  bandwidth_scenario[0]+".npy")
          
          plt.plot(Lambdas,global_constraints, label ="Global constraint $P_1$ under "+ bandwidth_scenario[0] )
         
          plt.xlabel("$\lambda$ (req/s)", fontsize=8)
          plt.ylabel("$\widehatR_{P_1}$ (second)", fontsize=8)
      th=np.load(folder_address + "/thereshould" + ".npy")
      thereshould=[th] * len(Lambdas)
      plt.plot(Lambdas, thereshould, label ='Threshold') 
      plt.legend(loc=4, prop={'size': 6.5})  
      plt.show()   
    #   #pdb.set_trace()  
    #   config_files_address="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/Lambda_files"
    # #config_files_address=sys.argv[1]
    # # Lambdas=sys.argv[2]
    #   pdb.set_trace()
    # # fourG_scenario=np.load(config_files_address +  "/global_constraints" + sys.argv[3] + ".npy")
    # # fiveG_slow_scenario=config_files_address +  "/global_constraints" + sys.argv[4] + ".npy"
    # # fiveG_fast_scenario=config_files_address +  "/global_constraints" + sys.argv[5] + ".npy"
    #   #Lambda_bound="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/Lambda_bound.txt"
    #   fourG_scenario=np.load(config_files_address +  "/global_constraints_4G.npy")
    #   fiveG_slow_scenario=np.load(config_files_address +  "/global_constraints_5G_2gb.npy")
    #   fiveG_fast_scenario=np.load(config_files_address +  "/global_constraints_5G_4gb.npy")
      
    #   fourG_scenario_Lambdas=np.load(config_files_address +  "/Lambda_4G.npy")
    #   fiveG_slow_scenario_Lambdas=np.load(config_files_address +  "/Lambda_5G_2gb.npy")
    #   fiveG_fast_scenario_Lambdas=np.load(config_files_address +  "/Lambda_5G_4gb.npy")
      
    #   plt.plot(fourG_scenario_Lambdas, fourG_scenario, label ='Numbers')
    #   plt.plot(fiveG_slow_scenario_Lambdas, fiveG_slow_scenario)
    #   plt.plot(fiveG_fast_scenario_Lambdas, fiveG_fast_scenario)
    #    # Len=max(len(fourG_scenario), len(fiveG_slow_scenario), len(fiveG_fast_scenario))
    #    # data = np.loadtxt(Lambda_bound, delimiter=',')
    #    # Lambdas=np.array(size(Len),)
    #    # for Lambda in np.arange(data[0], data[1], data[2]):
    

def main(main_json_file, graph_file, input_file,folder_address):     
     
    
    #system_file="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/Input_file.json"
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
    #pdb.set_trace()
    
    Lambdas=[]
    for Lambda in np.arange(Lambda_bound_start, Lambda_bound_end, Lambda_bound_step):
        Lambdas.append(Lambda)
        
    pure_json_file=create_pure_json(main_json_file)
    
    json_files_name=generate_json_files (pure_json_file, Lambdas, folder_address, bandwidth_scenarios) 
       
    compute_save_result(graph_file, bandwidth_scenarios, iteration_number, 
                        folder_address,json_files_name, Lambdas )
       
      
    draw_plots(folder_address,bandwidth_scenarios)
      
if __name__ == '__main__':
    
    main_json_file=sys.argv[1]
    graph_file=sys.argv[2]
    input_file=sys.argv[3]
    folder_address=sys.argv[4]
    # main_json_file="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/system_description11.json"
    # graph_file="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/DAG_Conf.txt"
    # input_file="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/Input_file.json"
    # folder_address="/Users/hamtasedghani/Desktop/AISPrintFirstPaper/Lambda_files"
    
    main(main_json_file, graph_file, input_file,folder_address)