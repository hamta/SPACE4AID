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
#   @param MaxIt_and_seed List whose n-th element stores a tuple with the 
#                         number of iterations to be performed by the n-th 
#                         cpu core and the seed it should use for random 
#                         numbers generation
#   @return The results returned by Algorithm.RandomGreedy.random_greedy
def fun_greedy(MaxIt_and_seed, S):
    GA = RandomGreedy(S, log=Logger(verbose=2))
    return GA.random_greedy(MaxIt_and_seed[1], MaxIt_and_seed[0], 2)


## Function to get a list whose n-th element stores a tuple with the number 
# of iterations to be performed by the n-th cpu core and the seed it should 
# use for random numbers generation
#   @param iteration Total number of iterations to be performed
#   @param seed Seed for random number generation
#   @param cpuCore Total number of cpu cores
#   @return The list of iterations to be performed by each core and the 
#           corresponding seed
def get_iterations_and_seed(iteration, seed, cpuCore):
    iterations_and_seeds = []
    local = int(iteration / cpuCore)
    remainder = iteration % cpuCore
    for r in range(cpuCore):
        r_seed = r * r * cpuCore * cpuCore * seed
        if r < remainder:
            iterations_and_seeds.append((local + 1, r_seed))
        else:
            iterations_and_seeds.append((local, r_seed))
    return iterations_and_seeds


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
        iterations_and_seeds = get_iterations_and_seed(iteration, seed, cpuCore)
        
        if __name__ == '__main__':
            
            start = time.time()
            
            with Pool(processes=cpuCore) as pool:
                
                partial_gp = functools.partial(fun_greedy, S=S)
                
                full_result = pool.map(partial_gp, iterations_and_seeds)
            
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
    
    system_file = sys.argv[1]
    iteration = int(sys.argv[2])
    start_lambda = float(sys.argv[3])
    end_lambda = float(sys.argv[4])
    step = float(sys.argv[5])
    seed = int(sys.argv[6])
    
    # system_file = "ConfigFiles/Random_Greedy.json"
    # iteration = 1000
    # start_lambda = 0.14
    # end_lambda = 0.15
    # step = 0.01
    # seed = 2
    # Lambda = start_lambda
   
    main(system_file, iteration, seed, start_lambda, end_lambda, step)
    
    