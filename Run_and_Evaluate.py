import pdb

from classes.Logger import Logger
from classes.System import System
from classes.Algorithm import Algorithm, RandomGreedy
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
def fun_greedy(core_params, S, verbose):
    core_logger = Logger(verbose=verbose)
    if core_params[2] != "":
        log_file = open(core_params[2], "a")
        core_logger.stream = log_file
    GA = RandomGreedy(S,seed=core_params[1], log=core_logger)
    result = GA.random_greedy( MaxIt=core_params[0], K=2)
    if core_params[2] != "":
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
        if logger.stream != sys.stdout:
            if cpuCore > 1 and logger.verbose > 0:
                log_file = ".".join(logger.stream.name.split(".")[:-1])
                log_file += "_" + str(r) + ".log"
            else:
                log_file = logger.stream.name
        else:
            log_file = ""
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
def main(dic, log_directory):
    # initialize logger
    logger = Logger(verbose=args.verbose)
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
        #method = config["Method"]
        iteration = config["IterationNumber"]
        seed = config["Seed"]
        if not "Lambda" in dic.keys():

            start_lambda = config["LambdaBound"]["start"]
            end_lambda = config["LambdaBound"]["end"]
            step = config["LambdaBound"]["step"]
        else:
            start_lambda = dic["Lambda"]
            end_lambda = start_lambda+1
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
            logger_lambda.log("Printing final result", 1)
            elite.elite_results[0].solution.logger = logger_lambda
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
    error = Logger(stream = sys.stderr, verbose=1, error=True)
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
            pdb.set_trace()
            if args.Lambda is not None:
                dic["Lambda"]=float(args.Lambda)

    else:
        try:
             Lambda = float(args.evaluation_lambda[0])
             dic["Lambda"]=Lambda
        except ValueError:
             error.log("{} must be a number".format(args.config))
             sys.exit(1)
        if not os.path.exists(args.evaluation_lambda[1]):
            error.log("{} does not exist".format(args.config))
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

    # load data from the test configuration file
    ''' config = {}
    with open(args.config, "r") as config_file:
        config = json.load(config_file)
    
    system_file = "ConfigFiles/Random_Greedy_Pacsltk.json"
    system_file = "ConfigFiles/RG-MaskDetection.json" # "ConfigFiles/Random_Greedy.json"
    iteration = 1000
    start_lambda = 0.14
    end_lambda = 0.15
    step = 0.01
    seed = 2
    Lambda = start_lambda
    system_file = create_pure_json(system_file)
    with open(system_file, "r") as a_file:
        json_object = json.load(a_file)


    # set the current Lambda in the system description
   # json_object["Lambda"] = Lambda

    # initialize system
    parser = argparse.ArgumentParser(description="SPACE4AI-D")
    dic={}
    args = parser.parse_args()
    S = System(system_json=json_object)#, log=Logger(verbose=2))
    # best_result_no_update, elite, random_params=fun_greedy(iteration, S, seed)
    # generate_output_json(S.Lambda, elite.elite_results[0], S)

    solution_file="Output_Files/Lambda_10.0_output_json.json"#"Output_Files/Lambda_0.16_output_json.json"
    GA = RandomGreedy(S,seed )
    MD_solution_file="Output_Files/RG-MaskDetection.json"
    result = GA.random_greedy( MaxIt=10)
    result[0].print_result(S,MD_solution_file)

    A = Algorithm(S,seed)
    result=A.create_solution_by_file(solution_file)
    result.print_result(S, solution_file=solution_file)'''






    main(dic, args.log_directory)

