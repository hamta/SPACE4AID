from classes.Logger import Logger
from classes.System import System
from classes.Algorithm import RandomGreedy
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
#   @param core_params Tuple of parameters required by the current core: 
#                      (number of iterations, seed, logger)
#   @param S An instance of System.System class including system description
#   @param verbose Verbosity level
#   @return The results returned by Algorithm.RandomGreedy.random_greedy
def fun_greedy(core_params, S, verbose):
    log_file = open(core_params[2], "a")
    GA = RandomGreedy(S, log=Logger(stream=log_file, verbose=verbose))
    result = GA.random_greedy(seed=core_params[1], MaxIt=core_params[0], K=2)
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
def get_core_params(iteration, seed, cpuCore, logger):
    core_params = []
    local = int(iteration / cpuCore)
    remainder = iteration % cpuCore
    for r in range(cpuCore):
        if cpuCore > 1 and logger.verbose > 0:
            log_file = ".".join(logger.stream.name.split(".")[:-1])
            log_file += "_" + str(r) + ".log"
        else:
            log_file = logger.stream.name
        r_seed = r * r * cpuCore * cpuCore * seed
        if r < remainder:
            core_params.append((local + 1, r_seed, log_file))
        else:
            core_params.append((local, r_seed, log_file))
    return core_params


## Main function
#   @param system_file Name of the file storing the system description
#   @param config Dictionary of configuration parameters
#   @param log_directory Directory for logging
def main(system_file, config, log_directory):
    
    # initialize logger
    logger = Logger(verbose=args.verbose)
    if log_directory != "":
        log_file = open(os.path.join(log_directory, "LOG.log"), "a")
        logger.stream = log_file
    
    # configuration parameters
    method = config["Method"]
    iteration = config["IterationNumber"]
    seed = config["Seed"]
    start_lambda = config["LambdaBound"]["start"]
    end_lambda = config["LambdaBound"]["end"]
    step = config["LambdaBound"]["step"]
    
    # load system description
    system_file = create_pure_json(system_file)
    with open(system_file, "r") as a_file:
        json_object = json.load(a_file)
    
    # loop over Lambdas
    for Lambda in np.arange(start_lambda, end_lambda, step):
        
        # initialize logger for current lambda
        separate_loggers = (logger.verbose > 0 and log_directory != "")
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
        core_params = get_core_params(iteration,seed,cpuCore,logger_lambda)
        
        if __name__ == '__main__':
            
            start = time.time()
            
            with Pool(processes=cpuCore) as pool:
                
                partial_gp = functools.partial(fun_greedy, S=S, 
                                               verbose=logger_lambda.verbose)
                
                full_result = pool.map(partial_gp, core_params)

            end = time.time()
            
            # get final list combining the results of all threads
            elite = full_result[0][1]
            for tid in range(1, cpuCore):
                elite.merge(full_result[tid][1])
                
            # compute elapsed time
            tm1 = end-start
            
        # print result
        generate_output_json(Lambda, elite.elite_results[0], S)
        logger.log("Lambda: {} --> elapsed_time: {}".format(Lambda, tm1))
        
        if separate_loggers:
            log_file_lambda.close()
    
    if log_directory != "":
        log_file.close()
        

    
if __name__ == '__main__':
    
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
    if not os.path.exists(args.config):
        error.log("{} does not exist".format(args.config))
        sys.exit(1)
    
    # check if the log directory exists and create it otherwise
    if args.log_directory != "" and os.path.exists(args.log_directory):
        print("Directory {} already exists. Terminating...".\
              format(args.log_directory))
        sys.exit(0)
    else:
        createFolder(args.log_directory)
    
    # load data from the test configuration file
    config = {}
    with open(args.config, "r") as config_file:
        config = json.load(config_file)
    
    # system_file = "ConfigFiles/Random_Greedy_Pacsltk.json"
    # system_file = "ConfigFiles/Random_Greedy.json"
    # iteration = 1000
    # start_lambda = 0.14
    # end_lambda = 0.15
    # step = 0.01
    # seed = 2
    # Lambda = start_lambda
   
    main(args.system_file, config, args.log_directory)
    
    