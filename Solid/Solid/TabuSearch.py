from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import deque
from numpy import argmin
import numpy as np
import time


class TabuSearch:
    """
    Conducts tabu search
    """
    __metaclass__ = ABCMeta

    cur_steps = None

    tabu_size = None
    tabu_list = None

    initial_state = None
    current = None
    best = None

    max_steps = None
    max_score = None
    max_time=None

    def __init__(self, initial_state, tabu_size, max_steps,max_time=None, max_score=None):
        """

        :param initial_state: initial state, should implement __eq__ or __cmp__
        :param tabu_size: number of states to keep in tabu list
        :param max_steps: maximum number of steps to run algorithm for
        :param max_score: score to stop algorithm once reached
        """
        self.initial_state = initial_state

        if isinstance(tabu_size, int) and tabu_size > 0:
            self.tabu_size = tabu_size
        else:
            raise TypeError('Tabu size must be a positive integer')

        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        if isinstance(max_time, (int, float)) and max_time > 0:
            self.max_time = max_time
        elif not self.max_steps > 0:
            raise ValueError('Maximum time or steps must be positive')
        if max_score is not None:
            if isinstance(max_score, (int, float)):
                self.max_score = float(max_score)
            else:
                raise TypeError('Maximum score must be a numeric type')

    def __str__(self):
        return ('TABU SEARCH: \n' +
                'CURRENT STEPS: %d \n' +
                'BEST SCORE: %f \n' +
                'BEST MEMBER: %s \n\n') % \
               (self.cur_steps, self._score(self.best), str(self.best))

    def __repr__(self):
        return self.__str__()

    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm

        :return: None
        """
        self.cur_steps = 0
        self.tabu_list = deque(maxlen=self.tabu_size)
        self.current = deepcopy(self.initial_state)
        self.best = deepcopy(self.initial_state)

    @abstractmethod
    def _score(self, state):
        """
        Returns objective function value of a state

        :param state: a state
        :return: objective function value of state
        """
        pass

    @abstractmethod
    def _neighborhood(self):
        """
        Returns list of all members of neighborhood of current state, given self.current

        :return: list of members of neighborhood
        """
        pass

    def _best(self, neighborhood,method):
        """
        Finds the best member of a neighborhood

        :param neighborhood: a neighborhood
        :return: best member of neighborhood
        
        """
        if method=="best":
            neighbor=neighborhood[argmin([self._score(x) for x in neighborhood])]
        else:
            idx=np.random.randint(0,len(neighborhood))
            neighbor=neighborhood[idx]
            
        return neighbor
    
    def check_in_tabu_list(self,solution):
        find=False
        i=0
        while not find and i<len(self.tabu_list):
            if solution==self.tabu_list[i]:
                find=True
            i+=1
       
        return find
    def run(self, verbose=True,method="best"):
        """
        Conducts tabu search

        :param verbose: indicates whether or not to print progress regularly
        :return: best state and objective function value of best state
        """
        best_sol_cost_list=[]
        current_solution_cost_list=[]
        time_list=[]

        self._clear()
        best_sol_cost_list.append(self._score(self.best))
        current_solution_cost_list.append(self._score(self.current))
        time_list.append(time.time())
        start=time.time()
        while self.cur_steps<self.max_steps or time.time()-start<self.max_time:
            self.cur_steps += 1

            if ((self.cur_steps + 1) % 100 == 0) and verbose:
                print(self)

            neighborhood = self._neighborhood()
            if len(neighborhood)<1:
                break
            neighborhood_best = self._best(neighborhood,method)
            
            while True:
                best_sol_cost_list.append(self._score(self.best))
                current_solution_cost_list.append(self._score(self.current))
                time_list.append(time.time())
                #  if all([x in self.tabu_list for x in neighborhood]):
                if all([self.check_in_tabu_list(x) for x in neighborhood] ):
                    print("TERMINATING - NO SUITABLE NEIGHBORS")
                    return self.best, self._score(self.best) , current_solution_cost_list, best_sol_cost_list, time_list
               
                if self.check_in_tabu_list(neighborhood_best):
                # if neighborhood_best in self.tabu_list:
                    if self._score(neighborhood_best) < self._score(self.best):
                        self.tabu_list.append(neighborhood_best)
                        self.best = deepcopy(neighborhood_best)

                        break
                    else:
                       
                        neighborhood.remove(neighborhood_best)
                        neighborhood_best = self._best(neighborhood,method)
                else:
                    self.tabu_list.append(neighborhood_best)
                    self.current = deepcopy(neighborhood_best)
                    
                   
                    if self._score(self.current) < self._score(self.best):
                        self.best = deepcopy(self.current)
                      
                    break
           
            if self.max_score is not None and self._score(self.best) < self.max_score:
                print("TERMINATING - REACHED MAXIMUM SCORE")
                return self.best, self._score(self.best)
        print("TERMINATING - REACHED MAXIMUM STEPS")
        
        return self.best, self._score(self.best), current_solution_cost_list, best_sol_cost_list, time_list
