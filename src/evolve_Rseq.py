# -*- coding: utf-8 -*-


# ================
# Evolve Rsequence
# ================


#import cProfile
import time

import random
import numpy as np
import json
#import time
#import matplotlib.pyplot as plt
#import copy
import os

from genome import Genome
#from expected_entropy import expected_entropy



def read_json_file(filename):
    ''' Returns the content of a specified JSON file as a python object. '''
    with open(filename) as json_content:
        return json.load(json_content)

def reproduce(organisms):
    ''' The Genome objects in `organisms` (a list) are cloned, and a the clones
    (a list) are returned. '''
    return [Genome(clone=parent) for parent in organisms]

def end_run(gen, solution_gen, drift_time, max_n_gen):
    if solution_gen:
        # A solution was already found:
        # Stop after `drift_time`
        return gen >= solution_gen + drift_time
    else:
        # No solution found so far:
        # Stop if no solution is found before `max_n_gen`
        return gen > max_n_gen





def main():
    
    # SET UP
    
    config_filename = 'config.json'
    config_dict = read_json_file(config_filename)
    
    run_tag = time.strftime("%Y%m%d%H%M%S")
    run_mode = config_dict['run_mode']
    
    # Ensure run_tag uniqueness when running in parallel
    if run_mode == 'parallel':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        run_tag = run_tag + '_' + str(rank)
    
    pop_size = config_dict['pop_size']
    motif_n = config_dict['motif_n']
    update_period = config_dict['update_period']
    
    # Results directory
    results_dirpath = '../results/' + run_tag + '/'
    os.makedirs(results_dirpath, exist_ok=True)
    
    
    # Initialize population
    population = [Genome(config_dict, []) for i in range(pop_size)]
    
    
    
    
    
    



if __name__ == '__main__':
    
    # Start a new run as soon as the previous one is done, until the process is killed
    while True:
        main()













