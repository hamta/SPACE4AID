import pdb
from classes.Solution import Configuration, Result, EliteResults
import output_yaml_generator
import system_file_json_generator
import Input_json_generator
from classes.Logger import Logger
from classes.System import System
from classes.Algorithm import Algorithm, RandomGreedy, Tabu_Search, Simulated_Annealing, Genetic_algorithm
import time
import sys
import os
import json
import numpy as np
import multiprocessing as mpp
from multiprocessing import Pool
import functools
import argparse


## Function to create a directory given its name (if the directory already
# exists, nothing is done)
#   @param directory Name of the directory to be created
def createFolder (directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ("Error: Creating directory. " +  directory)


## Function to create the pure json file from system description by removing
# the comments lines
#   @param main_json_file Name of the file with the system description
#   @return Name of the pure json file with the system description
def create_pure_json(main_json_file):

    data = []

    # read file
    with open(main_json_file,"r") as fp:
        Lines = fp.readlines()

    # loop over lines and drop portions preceded by '#'
    for line in Lines:
        idx = line.find('#')
        if (idx != -1):
            last = line[-1]
            line = line[0:idx]
            if last == "\n":
                line = line+last
        data.append(line)

    # write data on the new json file
    pure_json = 'pure_json.json'
    filehandle = open(pure_json, "w")
    filehandle.writelines(data)
    filehandle.close()

    return pure_json


## Function to generate the output json file as the optimal placement for a
# specified Lambda
#   @param Lambda incoming workload rate
#   @param result The optimal Solution.Result returned by fun_greedy
#   @param S An instance of System.System class including system description
#   @param onFile True if the result should be printed on file (default: True)
def generate_output_json(Lambda, result, S, onFile = True):

    # generate name of output file (if required)
    if onFile:
        output_json = "Output_Files/Lambda_" + \
                        str(round(float(Lambda), 10)) + \
                            '_output_json.json'
    else:
        output_json = ""

    result.print_result(S, solution_file=output_json)


## Function to create an instance of Algorithm class and run the random greedy
# method
#   @param core_params Tuple of parameters required by the current core:
#                      (number of iterations, seed, logger)
#   @param S An instance of System.System class including system description
#   @param verbose Verbosity level
#   @return The results returned by Algorithm.RandomGreedy.random_greedy
def fun_greedy(core_params, S, verbose, K=1):
    core_logger = Logger(verbose=verbose)
    if core_params[3] != "":
        log_file = open(core_params[3], "a")
        core_logger.stream = log_file
    GA = RandomGreedy(S, seed=core_params[2], log=core_logger)
    result = GA.random_greedy( MaxIt=core_params[0], K=K, MaxTime=core_params[1])
    if core_params[3] != "":
        log_file.close()
    return result


## Function to get a list whose n-th element stores a tuple with the number
# of iterations to be performed by the n-th cpu core and the seed it should
# use for random numbers generation
#   @param iteration Total number of iterations to be performed
#   @param seed Seed for random number generation
#   @param cpuCore Total number of cpu cores
#   @param logger Current logger
#   @return The list of parameters required by each core (number of
#           iterations, seed and logger)
def get_core_params(iteration, Max_time, seed, cpuCore, logger):
    core_params = []
    local_itr = int(iteration / cpuCore)
    remainder_itr = iteration % cpuCore
    local_Max_time = int(Max_time / cpuCore)
    remainder_Max_time = iteration % cpuCore
    for r in range(cpuCore):
        if logger.stream != sys.stdout:
            if cpuCore > 1 and logger.verbose > 0:
                log_file = ".".join(logger.stream.name.split(".")[:-1])
                log_file += "_" + str(r) + ".log"
            else:
                log_file = logger.stream.name
        else:
            log_file = ""
        r_seed = r * r * cpuCore * cpuCore * seed
        current_local_itr = local_itr
        current_local_Max_time = local_Max_time
        if r < remainder_itr:
            current_local_itr = local_itr + 1
        if r < remainder_Max_time:
            current_local_Max_time = local_Max_time + 1

        core_params.append((current_local_itr, current_local_Max_time, r_seed, log_file))
    return core_params

def Random_Greedy_run(S,iteration_number_RG,seed,Max_time_RG,logger, startingPointsNumber):

    cpuCore = int(mpp.cpu_count())
    core_params = get_core_params(iteration_number_RG, Max_time_RG, seed, cpuCore, logger)
    '''GA = RandomGreedy(S, seed, log=logger)
    result = GA.random_greedy( MaxIt=iteration_number_RG, K=startingPointsNumber, MaxTime=Max_time_RG)
    for res in result[1].elite_results:
        if res.solution is not None:
            print(res.performance)
            print(res.solution.Y_hat)
            print(res.cost)'''

    solutions=[]
    feasible_found = False
    elite_sol = []
    if __name__ == '__main__':
            start=time.time()
            with Pool(processes=cpuCore) as pool:

                partial_gp = functools.partial(fun_greedy, S=S,
                                               verbose=logger.verbose, K=startingPointsNumber)

                full_result = pool.map(partial_gp, core_params)

            end = time.time()
            exec=end -start
            first_unfeasible = False
            # get final list combining the results of all threads
            for tid in range(cpuCore):
                if feasible_found:
                    if full_result[tid][1].elite_results[0].performance[0]:
                        elite_sol.merge(full_result[tid][1], True)
                else:
                    if full_result[tid][1].elite_results[0].performance[0]:
                        feasible_found = True
                        elite_sol = full_result[tid][1]
                    else:
                        if not first_unfeasible:
                            elite_sol = full_result[tid][1]
                            first_unfeasible = True
                        else:
                            elite_sol.merge(full_result[tid][1], False)
                #if len(elite_sol.elite_results)<K:
                #    K=len(elite_sol.elite_results)

            if feasible_found:
                for sol in elite_sol.elite_results:
                    if sol.cost < np.inf:
                        solutions.append(sol.solution)
            else:
                for sol in elite_sol.elite_results:
                    if sol.violation_rate < np.inf:
                        solutions.append(sol.solution)


    '''for res in elite_sol.elite_results:
        if res.solution is not None:
            print(res.performance)
            print(res.solution.Y_hat)
            print(res.cost)'''
    print("RG cost: " + str(elite_sol.elite_results[0].cost))
    return feasible_found, solutions, elite_sol.elite_results[0]

def TabuSearch_run(S,iteration_number_RG, max_iterations,
                 seed,Max_time_RG, Max_time, method,tabu_memory,  K=1, besties_RG=None):

            TS_Solid= Tabu_Search(iteration_number_RG,seed,system=S,Max_time_RG=Max_time_RG,K=K, besties_RG=besties_RG)
            result, best_sol_info, Starting_points_info=TS_Solid.run_TS (method, tabu_memory, min_score=None, max_steps=max_iterations,Max_time=Max_time)

            return result

def SimulatedAnealing_run( S,iteration_number_RG, max_iterations,
                         seed,Max_time_RG, Max_time,temp_begin,schedule_constant,K=1, besties_RG=None):
                #temp_begin=5
                #schedule_constant=0.99

                SA_Solid= Simulated_Annealing(iteration_number_RG,seed,system=S,Max_time_RG=Max_time_RG,K=K, besties_RG=besties_RG)
                result, best_sol_info, Starting_points_info=SA_Solid.run_SA( temp_begin, schedule_constant, max_iterations,  min_energy=None, schedule='exponential',Max_time=Max_time)
                return result

def GeneticAlgorithm_run(S,iteration_number_RG, max_iteration_number,seed,
                         K_init_population, Max_time_RG, Max_time, mutation_rate,crossover_rate, besties_RG=None):

    #mutation_rate = 0.7
    #crossover_rate = 0.5
    GA = Genetic_algorithm(iteration_number_RG,seed, crossover_rate, mutation_rate, max_iteration_number, S, Max_time_RG=Max_time_RG,Max_time=Max_time, besties_RG=besties_RG)

    result, best_sol_cost_list_GA, time_list_GA, population=GA.run_GA(K_init_population)

    return result

def main(application_dir):

    input_json_dir=Input_json_generator.make_input_json(application_dir)
    with open(input_json_dir, "r") as a_file:
        input_json = json.load(a_file)
    if "VerboseLevel" in input_json.keys():
        logger = Logger(stream = sys.stderr, verbose=input_json["VerboseLevel"])
    else:
        error.log("{} does not exist.".format("VerboseLevel"))
        sys.exit(1)
    if "Methods" in input_json.keys():
        Methods=input_json["Methods"]
        if "method1" in Methods.keys():
            name=Methods["method1"]["name"]
            if "random" in name.lower():
                iteration_number_RG=Methods["method1"]["iterations"]
                Max_time_RG=Methods["method1"]["duration"]
            else:
                error.log("{} should be random greedy.".format("method1"))
                sys.exit(1)
        else:
            error.log("{} does not exist".format("Random greedy parameters"))
            sys.exit(1)
    else:
        error.log("{} does not exist".format("Methods"))
        sys.exit(1)
    startingPointNumber = 1
    if "method2" in Methods.keys():
        try:
            method_name=Methods["method2"]["name"]
            startingPointNumber=Methods["method2"]["startingPointNumber"]
            max_iteration_number=Methods["method2"]["iterations"]
            Max_time=Methods["method2"]["duration"]
        except Exception as e:
           error.log(" parameter {} does not exist".format(e))
           sys.exit(1)

    if "Seed" in input_json.keys():
        seed=input_json["Seed"]
    else:
        logger.log("{} does not exist".format("Seed"))
        sys.exit(1)
    print("\nStart parsing YAML files... ")
    system_file=system_file_json_generator.make_system_file(application_dir)
    #system_file = application_dir+ "/space4ai-d/system_description.json"#/SystemFile-Demo.json"#"ConfigFiles/RG-MaskDetection.json" # "ConfigFiles/Random_Greedy.json"

    system_file = create_pure_json(system_file)
    with open(system_file, "r") as a_file:
        json_object = json.load(a_file)
    #json_object["Lambda"] = Lambda

    print("\n Start parsing config files... ")
    S = System(system_json=json_object, log=logger)
    print("\n Start searching by  Random Greedy ... ")
    feasibility, starting_points, result = Random_Greedy_run(S,iteration_number_RG,seed,Max_time_RG, logger, startingPointNumber)
    if not feasibility:
        error.log("No feasible solution is found by RG")
    else:
        if "method2" in Methods.keys():
            print("\n Start searching by heuristic method ... ")
            if "tabu" in method_name.lower():
                if "specialParameters" in Methods["method2"].keys():
                    if "tabuSize" in Methods["method2"]["specialParameters"].keys():
                        tabu_size= Methods["method2"]["specialParameters"]["tabuSize"]
                    else:
                        error.log("{} does not exist".format("Tabu memory"))
                        sys.exit(1)
                else:
                    error.log("{} does not exist".format("specialParameters"))
                    sys.exit(1)

                result=TabuSearch_run(S,iteration_number_RG, max_iteration_number, seed,Max_time_RG, Max_time,"random",tabu_size,K=1, besties_RG=starting_points)
            elif "local" in method_name.lower():
                result=TabuSearch_run(S,iteration_number_RG, max_iteration_number, seed,Max_time_RG, Max_time,"best",1, K=1, besties_RG=starting_points)
            elif "simulated" in method_name.lower():
                if "specialParameters" in Methods["method2"].keys():
                    if "temperature" in Methods["method2"]["specialParameters"].keys():
                        temp_begin= Methods["method2"]["specialParameters"]["temperature"]
                    else:
                        error.log("{} does not exist".format("initial temperature"))
                        sys.exit(1)
                    if "scheduleConstant" in Methods["method2"]["specialParameters"].keys():
                        schedule_constant= Methods["method2"]["specialParameters"]["scheduleConstant"]
                        result = SimulatedAnealing_run(S,iteration_number_RG, max_iteration_number, seed,Max_time_RG, Max_time,temp_begin,schedule_constant,K=1, besties_RG=starting_points)
                    else:
                        error.log("{} does not exist".format("temperature"))
                        sys.exit(1)
                else:
                    error.log("{} does not exist".format("specialParameters"))
                    sys.exit(1)
            elif "genetic" in method_name.lower():
                if "specialParameters" in Methods["method2"].keys():
                    if "mutationRate" in Methods["method2"]["specialParameters"].keys():
                        mutation_rate= Methods["method2"]["specialParameters"]["mutationRate"]
                    else:
                        error.log("{} does not exist".format("mutation rate"))
                        sys.exit(1)
                    if "crossoverRate" in Methods["method2"]["specialParameters"].keys():
                        crossover_rate = Methods["method2"]["specialParameters"]["crossoverRate"]
                        result = GeneticAlgorithm_run(S,iteration_number_RG, max_iteration_number, seed, startingPointNumber,Max_time_RG, Max_time,mutation_rate,crossover_rate, besties_RG=starting_points)
                    else:
                        error.log("{} does not exist".format("crossover rate"))
                        sys.exit(1)

                else:
                    error.log("{} does not exist".format("specialParameters"))
                    sys.exit(1)
        else:
            result = Result()
            result.solution = starting_points[0]
            result.objective_function(S)


    output_json=application_dir+"/space4ai-d/Output.json"

    result.print_result(S,output_json)
    output_yaml_generator.main(application_dir+"/")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SPACE4AI-D")

    parser.add_argument("-C", "--application_dir",
                        help="Application directory")
    #parser.add_argument('-O', "--output_file",
   #                     help="Output folder")


    args = parser.parse_args()

    # initialize error stream
    error = Logger(stream = sys.stderr, verbose=1, error=True)
    dic={}
    # check if the system configuration file exists
    if not os.path.exists(args.application_dir):
        error.log("{} does not exist".format(args.application_dir))
        sys.exit(1)
    else:
        application_dir=args.application_dir

   # if not os.path.exists(args.output_file):
     #   error.log("{} does not exist".format(args.output_file))
     #   sys.exit(1)
  #  else:
     #   output_dir=args.output_file
    #output_dir="OutputFiles_demo"

    #application_dir="ConfigFiles_demo"
    #error = Logger(stream = sys.stderr, verbose=1, error=True)
    main(application_dir)
