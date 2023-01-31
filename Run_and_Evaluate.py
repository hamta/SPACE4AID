from classes.Logger import Logger
from classes.System import System
from classes.Algorithm import Algorithm, RandomGreedy, Tabu_Search, Simulated_Annealing, Genetic_algorithm
from classes.Solution import Configuration, Result, EliteResults
import time
import sys
import os
import json
import numpy as np
import multiprocessing as mpp
from multiprocessing import Pool
import functools
import argparse
from difflib import SequenceMatcher
import pathlib

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
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
    GA = RandomGreedy(S,seed=core_params[2], log=core_logger)
    result = GA.random_greedy(MaxIt=core_params[0], K=K, MaxTime=core_params[1])
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
                        elite_sol = EliteResults(full_result[tid][1].K)
                        elite_sol.elite_results.add(Result())
                        elite_sol.add(full_result[tid][1].elite_results[0])
                    else:
                        if not first_unfeasible:
                            elite_sol = EliteResults(full_result[tid][1].K)
                            elite_sol.elite_results.add(Result())
                            elite_sol.add(full_result[tid][1].elite_results[0], feasible_found)
                            first_unfeasible = True
                        else:
                            elite_sol.merge(full_result[tid][1], feasible_found)
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


## Main function
#   @param system_file Name of the file storing the system description
#   @param config Dictionary of configuration parameters
#   @param log_directory Directory for logging
def main_diff_Lambda(dic, log_directory):
    # initialize logger
    logger = Logger(verbose=1)

    methods_list=["Random_Greedy", "Local_Search", "Tabu_Search", "Simulated_Annealing", "Genetic_Algorithm"]


    if log_directory != "":
        log_file = open(os.path.join(log_directory, "LOG.log"), "a")
        logger.stream = log_file
    system_file=dic["system_file"]
    # load system description
    system_file = create_pure_json(system_file)
    with open(system_file, "r") as a_file:
        json_object = json.load(a_file)

    separate_loggers = (logger.verbose > 0 and log_directory != "")
    if not "config_file" in dic.keys():
        solution_file=dic["solution_file"]
        Lambda=dic["Lambda"]
        json_object["Lambda"] = Lambda
        print("\n"+str(Lambda))
        print("\n"+solution_file)
        if separate_loggers:
            log_file_lambda = "LOG_" + str(Lambda) + ".log"
            log_file_lambda = os.path.join(log_directory, log_file_lambda)
            log_file_lambda = open(log_file_lambda, "a")
            logger_lambda = Logger(stream=log_file_lambda,
                                   verbose=logger.verbose)
        else:
            logger_lambda = logger
        S = System(system_json=json_object, log=logger_lambda)
        A = Algorithm(S)
        result=A.create_solution_by_file(solution_file)
        generate_output_json(Lambda,result, S)
    else:
        # configuration parameters
        config = {}
        with open(dic["config_file"], "r") as config_file:
            config = json.load(config_file)
        if "Methods" in config.keys():
            Methods=config["Methods"]
            if "method1" in Methods.keys():
                name=Methods["method1"]["name"]
                similarity=[similar(name, method) for method in methods_list]
                idx=similarity.index(max(similarity))
                method1=methods_list[idx]
                iteration_number_RG=Methods["method1"]["iterations"]
                Max_time_RG=Methods["method1"]["duration"]
            else:
                error.log("{} does not exist".format("method1"))
                sys.exit(1)
            if "method2" in Methods.keys():
                name = Methods["method2"]["name"]
                similarity = [similar(name, method) for method in methods_list]
                idx = similarity.index(max(similarity))
                method2 = methods_list[idx]
                startingPointsNumber = Methods["method2"]["startingPointsNumber"]
                max_iteration_number = Methods["method2"]["iterations"]
                Max_time = Methods["method2"]["duration"]
            else:
                method2 = ""
        else:
            error.log("{} does not exist".format("Methods"))
            sys.exit(1)

        seed = config["Seed"]
        if not "Lambda" in dic.keys():

            start_lambda = config["LambdaBound"]["start"]
            end_lambda = config["LambdaBound"]["end"]
            step = config["LambdaBound"]["step"]
        else:
            start_lambda = dic["Lambda"]
            end_lambda = start_lambda + 1
            step = 1
        # loop over Lambdas
        for Lambda in np.arange(start_lambda, end_lambda, step):

            # initialize logger for current lambda
            if separate_loggers:
                log_file_lambda = "LOG_" + str(Lambda) + ".log"
                log_file_lambda = os.path.join(log_directory, log_file_lambda)
                log_file_lambda = open(log_file_lambda, "a")
                logger_lambda = Logger(stream=log_file_lambda,
                                       verbose=logger.verbose)
            else:
                logger_lambda = logger

            # set the current Lambda in the system description
            json_object["Lambda"] = Lambda

            # initialize system
            S = System(system_json=json_object, log=logger_lambda)

            ################## Multiprocessing ###################
            cpuCore = int(mpp.cpu_count())
            core_params = get_core_params(iteration_number_RG, Max_time_RG, seed, cpuCore, logger_lambda)

            if __name__ == '__main__':

                start = time.time()

                with Pool(processes=cpuCore) as pool:

                    partial_gp = functools.partial(fun_greedy, S=S,
                                                   verbose=logger_lambda.verbose, K=startingPointsNumber)

                    full_result = pool.map(partial_gp, core_params)

                end = time.time()

                # get final list combining the results of all threads
                elite = full_result[0][1]
                for tid in range(1, cpuCore):
                    elite.merge(full_result[tid][1])

                # compute elapsed time
                tm1 = end-start
            import pdb
            #pdb.set_trace()
            # print result
            logger_lambda.log("Printing final result", 1)
            elite.elite_results[0].solution.logger = logger_lambda

            if method2 != "":
                starting_points_sol=[]
                startingPointsNumber=min(startingPointsNumber, len(elite.elite_results) )
                for i in range(startingPointsNumber):
                    starting_points_sol.append(elite.elite_results[i].solution)
                if method2==methods_list[1]:
                    TS_Solid= Tabu_Search(iteration_number_RG,seed,system=S,Max_time_RG=Max_time_RG,K=startingPointsNumber, besties_RG=starting_points_sol)
                    result, best_sol_info, Starting_points_info =TS_Solid.run_TS ("best", 50, min_score=None, max_steps=max_iteration_number,Max_time=Max_time)
                elif method2==methods_list[2]:
                    TS_Solid= Tabu_Search(iteration_number_RG,seed,system=S,Max_time_RG=Max_time_RG,K=startingPointsNumber, besties_RG=starting_points_sol)
                    result, best_sol_info, Starting_points_info =TS_Solid.run_TS ("random", 50, min_score=None, max_steps=max_iteration_number,Max_time=Max_time)
                elif method2==methods_list[3]:
                    temp_begin=5
                    schedule_constant=0.99
                    SA_Solid= Simulated_Annealing(iteration_number_RG,seed,system=S,Max_time_RG=Max_time_RG,K=startingPointsNumber, besties_RG=starting_points_sol)
                    result, best_sol_info, Starting_points_info =SA_Solid.run_SA (temp_begin, schedule_constant, max_steps=max_iteration_number,min_energy=None, schedule='exponential',Max_time=Max_time)
                else:
                    mutation_rate = 0.7
                    crossover_rate = 0.5
                    GA_Solid= Genetic_algorithm(iteration_number_RG,seed, crossover_rate, mutation_rate , max_steps=max_iteration_number, system=S, Max_time_RG=Max_time_RG, Max_time=Max_time, besties_RG=starting_points_sol)
                    result, best_sol_cost_list_GA, time_list_GA, population =GA_Solid.run_GA (K=startingPointsNumber)
                generate_output_json(Lambda, result, S)
            else:
                generate_output_json(Lambda, elite.elite_results[0], S)

            tm2 = time.time()-start
            logger.log("Lambda: {} --> elapsed_time: {}".format(Lambda, tm1+tm2))

            if separate_loggers:
                log_file_lambda.close()

    if log_directory != "":
        log_file.close()

def parse_config_file(S, input_json_dir, system_file, logger):

    with open(input_json_dir, "r") as a_file:
        input_json = json.load(a_file)
    '''if "VerboseLevel" in input_json.keys():
        logger = Logger(stream = sys.stderr, verbose=input_json["VerboseLevel"])
    else:
        error.log("{} does not exist.".format("VerboseLevel"))
        sys.exit(1)'''
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
    system_file = create_pure_json(system_file)
    with open(system_file, "r") as a_file:
        json_object = json.load(a_file)
    #json_object["Lambda"] = Lambda

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

    return result


## Main function
#   @param system_file Name of the file storing the system description
#   @param config Dictionary of configuration parameters
#   @param log_directory Directory for logging
def main(dic, log_directory):
    logger = Logger()#verbose=args.verbose)

    if log_directory != "":
        log_file = open(os.path.join(log_directory, "LOG.log"), "a")
        logger.stream = log_file
    system_file=dic["system_file"]
    # load system description
   # system_file = create_pure_json(system_file)
    with open(system_file, "r") as a_file:
        json_object = json.load(a_file)

    separate_loggers = (logger.verbose > 0 and log_directory != "")
    if not "config_file" in dic.keys():
        solution_file=dic["solution_file"]
        Lambda=dic["Lambda"]
        json_object["Lambda"] = Lambda
        print("\n"+str(Lambda))
        print("\n"+solution_file)
        if separate_loggers:
            log_file_lambda = "LOG_" + str(Lambda) + ".log"
            log_file_lambda = os.path.join(log_directory, log_file_lambda)
            log_file_lambda = open(log_file_lambda, "a")
            logger_lambda = Logger(stream=log_file_lambda,
                                   verbose=logger.verbose)
        else:
            logger_lambda = logger
        S = System(system_json=json_object, log=logger_lambda)
        A = Algorithm(S)
        result=A.create_solution_by_file(solution_file)
       # generate_output_json(Lambda,result, S)
    else:
        if "Lambda" in dic.keys():
            Lambda=dic["Lambda"]
            #print("Lambda: " + str(Lambda))
            json_object["Lambda"] = Lambda
        S = System(system_json=json_object, log=logger)
        config_file = dic["config_file"]
        result = parse_config_file(S, config_file, system_file, logger)
    path=pathlib.Path(system_file).parent.resolve()
    output_json=str(path) + "/Output.json"
    if result.solution is None:
        print("No solution is found.")
    else:
        result.print_result(S,output_json)

if __name__ == '__main__':
    error = Logger(stream = sys.stderr, verbose=1, error=True)

    parser = argparse.ArgumentParser(description="SPACE4AI-D")

    parser.add_argument("-s", "--system_file",
                        help="System configuration file")
    parser.add_argument('-L', "--Lambda",
                        help="Lambda")
    parser.add_argument('-c', "--config",
                        help="Test configuration file",
                        default="ConfigFiles/Input_file.json")
    parser.add_argument('-v', "--verbose",
                        help="Verbosity level",
                        type=int,
                        default=0)
    parser.add_argument('-e', '--evaluation_lambda', nargs=2, help="Evaluate the solution with Lambda")
    parser.add_argument('-l', "--log_directory",
                        help="Directory for logging",
                        default="")

    args = parser.parse_args()

    # initialize error stream
    #error = Logger(stream = sys.stderr, verbose=1, error=True)
    dic={}
    # check if the system configuration file exists
    if not os.path.exists(args.system_file):
        error.log("{} does not exist".format(args.system_file))
        sys.exit(1)
    else:
        dic["system_file"]=args.system_file
     # check if the test configuration file exists or we need an evaluation
    if args.evaluation_lambda is None:

        if not os.path.exists(args.config):
            error.log("{} does not exist".format(args.config))
            sys.exit(1)
        else:
            dic["config_file"]=args.config

            if args.Lambda is not None:
                dic["Lambda"]=float(args.Lambda)


    else:
        try:
             Lambda = float(args.evaluation_lambda[0])
             dic["Lambda"]=Lambda
        except ValueError:
             error.log("{} must be a number".format(args.evaluation_lambda[0]))
             sys.exit(1)
        if not os.path.exists(args.evaluation_lambda[1]):
            error.log("{} (solution to evaluate) does not exist".format(args.evaluation_lambda[1]))
            sys.exit(1)
        else:
            solution_file=args.evaluation_lambda[1]
            dic["solution_file"]=solution_file

    # check if the log directory exists and create it otherwise
    if args.log_directory != "":
        if os.path.exists(args.log_directory):
            print("Directory {} already exists. Terminating...".\
                  format(args.log_directory))
            sys.exit(0)
        else:
            createFolder(args.log_directory)

    #log_directory = "/Users/hamtasedghani/space4ai-d"
    #dic={}
    #dic["system_file"]= "/Users/hamtasedghani/space4ai-d/ConfigFiles_demo/space4ai-d/system_description.json"
    #dic["config_file"]="/Users/hamtasedghani/space4ai-d/ConfigFiles_demo/space4ai-d/Input.json"
    #dic["Lambda"]=1
    #dic["solution_file"]= "/Users/hamtasedghani/space4ai-d/ConfigFiles_demo/space4ai-d/Output.json"

    main(dic, args.log_directory)

