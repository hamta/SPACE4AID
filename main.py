from classes.Logger import Logger
from classes.System import System
from classes.Algorithm import RandomGreedy, Algorithm
import time
import sys
import os
import json
import numpy as np
import multiprocessing as mpp
from multiprocessing import Pool
import argparse
import functools
import sys


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
                        str(round(float(Lambda), 5)) + \
                            '_output_json.json'
    else:
        output_json = ""
    
    #stream = open("output.log", "a")
    #result.solution.logger = Logger(stream, verbose=7)
    result.print_result(S, solution_file=output_json)
    #stream.close()


## Function to create an instance of Algorithm class and run the random greedy 
# method
#   @param S An instance of System.System class including system description
#   @param seed Seed for random number generation
#   @param MaxIt Maximum number of RandomGreedy iterations
#   @return The results returned by Algorithm.RandomGreedy.random_greedy
def fun_greedy(MaxIt, S, seed):
    
    proc = mpp.current_process()
    pid = proc.pid
    GA = RandomGreedy(S,seed*pid, log=Logger(verbose=2))
    return GA.random_greedy( MaxIt)


## Function to get a list whose n-th element stores the number of iterations
# to be performed by the n-th cpu core
#   @param iteration Total number of iterations to be performed
#   @param cpuCore Total number of cpu cores
#   @return The list of iterations to be performed by each core
def get_n_iterations(iteration, cpuCore):
    iterations = []
    local = int(iteration / cpuCore)
    remainder = iteration % cpuCore
    for r in range(cpuCore):
        if r < remainder:
            iterations.append(local + 1)
        else:
            iterations.append(local)
    return iterations


## Main function
#   @param system_file Name of the file storing the system description
#   @param iteration Number of iterations to be performed by the RandomGreedy
#   @param seed Seed for random number generation
#   @param start_lambda Left extremum of the interval Lambda belongs to
#   @param end_lambda Right extremum of the interval Lambda belongs to
#   @param step Lambda is generated in start_lambda:step:end_lambda
def main(system_file, iteration, seed, start_lambda, end_lambda, step):
    
    # load system description
    system_file = create_pure_json(system_file)
    with open(system_file, "r") as a_file:
        json_object = json.load(a_file)
    
    # loop over Lambdas
    for Lambda in np.arange(start_lambda, end_lambda, step):
        
        # set the current Lambda in the system description
        json_object["Lambda"] = Lambda
        
        # initialize system
        S = System(system_json=json_object)#, log=Logger(verbose=2))
        
        ################## Multiprocessing ###################
        cpuCore = int(mpp.cpu_count())
        iterations = get_n_iterations(iteration, cpuCore)
        
        if __name__ == '__main__':
            
            start = time.time()
            
            with Pool(processes=cpuCore) as pool:
                
                partial_gp = functools.partial(fun_greedy, S=S, seed=seed)
                
                full_result = pool.map(partial_gp, iterations)
            
            end = time.time()
            
            # get final list combining the results of all threads
            elite = full_result[0][1]
            for tid in range(1, cpuCore):
                elite.merge(full_result[tid][1])
                
            # compute elapsed time
            tm1 = end-start
            
        # print result
        generate_output_json(Lambda, elite.elite_results[0], S)
        print("Lambda: ", Lambda, " --> elapsed_time: ", tm1)
        

    
if __name__ == '__main__':
    
    # system_file = sys.argv[1]
    # iteration = int(sys.argv[2])
    # start_lambda = float(sys.argv[3])
    # end_lambda = float(sys.argv[4])
    # step = float(sys.argv[5])
    # seed = int(sys.argv[6])
    
    system_file = "ConfigFiles/Random_Greedy.json"
    iteration = 100
    start_lambda = 0.15
    end_lambda = 0.16
    step = 0.01
    seed = 2
    
    
    #  # load system description
    system_file = create_pure_json(system_file)
    with open(system_file, "r") as a_file:
        json_object = json.load(a_file)
    
    
    # set the current Lambda in the system description
   # json_object["Lambda"] = Lambda
    
    # initialize system
   
    S = System(system_json=json_object)#, log=Logger(verbose=2))
    # best_result_no_update, elite, random_params=fun_greedy(iteration, S, seed)
    # generate_output_json(S.Lambda, elite.elite_results[0], S)
    
    solution_file="Output_Files/Lambda_0.16_output_json.json"
    # random_greedy_result=fun_greedy(100, S, seed)
    breakpoint()
    # random_greedy_result[0].print_result(S, solution_file=solution_file)
    A = Algorithm(S,seed, log=Logger(verbose=2))
    result=A.create_solution_by_file(solution_file)
    result.print_result(S, solution_file=solution_file)

    parser = argparse.ArgumentParser(description="SPACE4AI-D")

    parser.add_argument("-s", "--system_file", 
                        help="System configuration file")
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
    error = Logger(stream = sys.stderr, verbose=1, error=True)
    
    # check if the system configuration file exists
    if not os.path.exists(args.system_file):
        error.log("{} does not exist".format(args.system_file))
        sys.exit(1)
    
    # check if the test configuration file exists
    if args.evaluation_lambda == "":
        if not os.path.exists(args.config):
            error.log("{} does not exist".format(args.config))
            sys.exit(1)
        else:
            config = {}
            with open(args.config, "r") as config_file:
                config = json.load(config_file)
    else:
        try:
             Lambda = float(args.evaluation_lambda[0])
        except ValueError:
             error.log("{} must be a number".format(args.config))
             sys.exit(1)
        if not os.path.exists(args.evaluation_lambda[1]):
            error.log("{} does not exist".format(args.config))
            sys.exit(1)
        else:
            solution_file=args.evaluation_lambda[1]
    
    # check if the log directory exists and create it otherwise
    if args.log_directory != "":
        if os.path.exists(args.log_directory):
            print("Directory {} already exists. Terminating...".\
                  format(args.log_directory))
            sys.exit(0)
        else:
            createFolder(args.log_directory)
    
    # load data from the test configuration file
    
   
   # main(args.system_file, config, args.log_directory)
     
    # 
    #main(system_file, iteration, seed, start_lambda, end_lambda, step)
