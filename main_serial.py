from classes.Logger import Logger
from classes.System import System
from classes.Algorithm import RandomGreedy
import time
import sys
import os
import json
import numpy as np
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
#   @param MaxIt Maximum number of iterations
#   @param seed Seed for random numbers generation
#   @param S An instance of System.System class including system description
#   @param logger Logger.Logger object
#   @return The results returned by Algorithm.RandomGreedy.random_greedy
def fun_greedy(MaxIt, seed, S, logger):
    GA = RandomGreedy(S, log=logger)
    return GA.random_greedy(seed, MaxIt, 2)


## Main function
#   @param system_file Name of the file storing the system description
#   @param config Dictionary of configuration parameters
#   @param log_directory Directory for logging
def main(system_file, config, log_directory):
    
    # initialize logger
    logger = Logger(verbose=args.verbose)
    if log_directory != "":
        createFolder(log_directory)
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
        separate_loggers = (logger.verbose > 0)
        if separate_loggers:
            logger_lambda = Logger(stream=logger.stream, 
                                   verbose=logger.verbose,
                                   level=logger.level)
            if log_directory != "":
                log_file_lambda = "LOG_" + str(Lambda) + ".log"
                log_file_lambda = open(os.path.join(log_directory, 
                                                    log_file_lambda), "a")
                logger_lambda.stream = log_file_lambda
        else:
            logger_lambda = logger
        
        # set the current Lambda in the system description
        json_object["Lambda"] = Lambda
        
        # initialize system
        S = System(system_json=json_object, log=logger_lambda)
        
        start = time.time()
            
        full_result = fun_greedy(iteration, seed, S, logger_lambda)

        end = time.time()
            
        # get elite solutions
        elite = full_result[1]
                
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
    