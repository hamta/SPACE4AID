import sys
import random
import pdb
import numpy as np
import os
import time
import copy
import networkx as nx
from random import uniform
from collections import defaultdict
import re
import copy


    


def extract_raw_data_as_matrix(file,param): 
    'This method creat a matrix according to given file and parameter '
    'Param should be the last parameter before ":=" in the '
    my_file = open(file, "r")
    s = my_file.read()
    x = re.search(param +' :=(.*?);', s,flags=re.DOTALL).group(1)
    lists=list(x.split("\n"))

    result=[]
    #pdb.set_trace()
    for line in lists:
        if line :
           l=line.split()
           res=[]
           for i in l:
                res.append(i)
           result.append(copy.deepcopy(res))
            
    return result
    
x=extract_raw_data_as_matrix("F:/Project code/SPACE4AI/ConfigFiles/Resources_Param_Conf.txt",'Execution_time')

