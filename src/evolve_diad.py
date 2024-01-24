# -*- coding: utf-8 -*-

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

def generate_diad_plcm_map(config_dict):
    # Map diad placement indexes to genomic position of centroid
    G = config_dict['G']
    mot_len = config_dict['motif_len']
    plcm_idx_to_gnom_pos = []
    gnom_pos_to_plcm_idx = [[] for i in range(G)]
    for idx in range(G**2):
        x, y = divmod(idx, G)
        if y < x:
            y += G
        # Genome position (centroid)
        pos = int((x + y + mot_len)/2) % G
        plcm_idx_to_gnom_pos.append(pos)
        gnom_pos_to_plcm_idx[pos].append(idx)
    return plcm_idx_to_gnom_pos, gnom_pos_to_plcm_idx

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
    
    gnom_pos_to_plcm_idx = None
    if motif_n == 2:
        if config_dict['targets_type'] == 'centroids':
            # Map diad placement indexes to genomic position of centroid
            gnom_pos_to_plcm_idx = generate_diad_plcm_map(config_dict)[1]
        elif config_dict['targets_type'] == 'placements':
            gnom_pos_to_plcm_idx = []
        else:
            raise ValueError("'targets_type' must be 'centroids' or 'placements'.")
        
    
    # Initialize population
    population = [Genome(config_dict, gnom_pos_to_plcm_idx) for i in range(pop_size)]
    
    
    # START
    
    min_Rseq_list = []
    avg_Rseq_list = []
    max_Rseq_list = []
    best_org_Rseq_list = []
    best_org_Rseq_ev_list = []
    solution_gen = None
    
    drift_time = config_dict['drift_time']
    max_n_gen = config_dict['max_n_gen']
    
    gen = 0
    
    while not end_run(gen, solution_gen, drift_time, max_n_gen):
        
        gen += 1
        print("Gen:", gen)
        
        # Avoid second-order selection towards higher IC than necessary
        random.shuffle(population)
        
        fitness_list = []
        R_seq_list = []
        
        # Mutation and fitness evaluation
        # -------------------------------
        for org in population:
            #org.mutate_with_rate()
            org.mutate_ev()
            #fitness_list.append(org.get_fitness())
            fitness_list.append(org.get_fitness_new())
        
        # Sort population based on fitness (descending: from best to worst)
        ranking = sorted(zip(fitness_list, population), key=lambda x: x[0], reverse=True)
        
        sorted_pop = []
        sorted_fit = []
        for fitness, org in ranking:
            sorted_pop.append(org)
            sorted_fit.append(fitness)
        best_fitness = sorted_fit[0]
        print('sorted_fit:', sorted_fit)
        print('\tMax Fitness:', best_fitness)
        
        # If the model is a single motif, keep track of Rseq through time
        # ---------------------------------------------------------------
        if motif_n == 1:
            R_seq_list = [org.get_R_sequence_ev() for org in sorted_pop]
            
            # R_seq of all the organisms with best fitness: `best_organisms_R_seq`
            
            # Index of the last organism that has fitness equal to `best_fitness`
            rev_idx = sorted_fit[::-1].index(best_fitness)
            if rev_idx == 0:
                # all organisms are 'best organism'
                best_organisms_R_seq = R_seq_list[:]
            else:
                best_organisms_R_seq = R_seq_list[:-rev_idx]
            
            print('\tAvg R_sequence:\t', np.array(R_seq_list).mean())
            print('\tAvg best R_sequence:\t', np.array(best_organisms_R_seq).mean())
            print('\tR_frequency:\t', org.get_R_frequency())
            
            if update_period:
                if gen % update_period == 0:
                    min_Rseq_list.append(np.array(R_seq_list).min())
                    avg_Rseq_list.append(np.array(R_seq_list).mean())
                    max_Rseq_list.append(np.array(R_seq_list).max())
                    best_org_Rseq_list.append(np.array(best_organisms_R_seq).mean())
                    best_org_Rseq_ev_list.append(R_seq_list[0])
        
        # Selection
        # ---------
        middle = pop_size//2
        good, bad = sorted_pop[:middle], sorted_pop[middle:]
        
        # Number of ties
        if sorted_fit[middle-1] == sorted_fit[middle]:
            tie_fit_val = sorted_fit[middle]
            # XXX OPTIMIZE THIS CODE! ------------------------
            n_in_good = 0
            n_in_bad = 0
            for fitness in sorted_fit[middle:]:
                if fitness != tie_fit_val:
                    break
                n_in_bad += 1
    
            for fitness in sorted_fit[::-1][middle:]:
                if fitness != tie_fit_val:
                    break
                n_in_good += 1
            # ------------------------------------------------
            n_ties = min(n_in_good, n_in_bad)
        else:
            n_ties = 0
        
        # Replacement of bad organisms with good organisms
        if n_ties == 0:
            #population = good + copy.deepcopy(good)
            population = good + reproduce(good)
        else:
            #population = good + copy.deepcopy(good[:-n_ties]) + bad[:n_ties]
            population = good + reproduce(good[:-n_ties]) + bad[:n_ties]
        
        if motif_n == 2:
            # Save earliest solution
            if best_fitness == 0 and not solution_gen:
                org = population[0]
                org.export(results_dirpath + 'gen_{}_org.json'.format(gen))
                org.print_genome_map(results_dirpath + 'gen_{}_map.txt'.format(gen))
                # IC report (CSV) and Gaps report (JSON)
                org.study_diad(results_dirpath + 'gen_{}'.format(gen))
                solution_gen = gen
                
                # Save the settings for this run
                with open(results_dirpath + 'parameters.json', 'w') as f:
                    json.dump(config_dict, f)
            
            if update_period:
                if gen % update_period == 0:
                    org = population[0]
                    org.export(results_dirpath + 'ev_gen_{}_org.json'.format(gen))
                    org.print_genome_map(results_dirpath + 'ev_gen_{}_map.txt'.format(gen))
                    # IC report (CSV) and Gaps report (JSON)
                    org.study_diad(results_dirpath + 'ev_gen_{}'.format(gen))
                    solution_gen = gen
    
    # Export latest solution
    if solution_gen:
        org = population[0]
        org.export(results_dirpath + 'gen_{}_org.json'.format(gen))
        org.print_genome_map(results_dirpath + 'gen_{}_map.txt'.format(gen))
        # IC report (CSV) and Gaps report (JSON)
        org.study_diad(results_dirpath + 'gen_{}'.format(gen))
    else:
        os.rmdir(results_dirpath)
        print('{}: No solution obtained.'.format(results_dirpath))



if __name__ == '__main__':
    
    # Start a new run as soon as the previous one is done, until the process is killed
    while True:
        main()













