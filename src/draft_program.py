# -*- coding: utf-8 -*-

import random
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import copy

from genome import Genome
from expected_entropy import expected_entropy


config_filename = 'config.json'


def read_json_file(filename):
    with open(filename) as json_content:
        return json.load(json_content)

config_dict = read_json_file(config_filename)


run_tag = time.strftime("%Y%m%d%H%M%S")

pop_size = config_dict['pop_size']

# population = []
# for i in range(pop_size):
#     new_genome = Genome(config_dict)
#     population.append(new_genome)
population = [Genome(config_dict) for i in range(pop_size)]

motif_n = config_dict['motif_n']

update_period = 1

# =============================================================================
#     ORIGINAL METHOD
# =============================================================================


min_Rseq_list = []
avg_Rseq_list = []
max_Rseq_list = []
best_org_R_seq_list = []
best_org_Rseq_ev_list = []

for gen in range(10000):
    print("Gen:", gen)
    

    fitness_list = []
    R_seq_list = []
    
    # Mutation and fitness evaluation
    # -------------------------------
    for org in population:
        #org.mutate_with_rate()
        org.mutate_ev()
        ##################################################################
        fitness_list.append(org.get_fitness())
    
    ranking = sorted(zip(fitness_list, population), key=lambda x: x[0], reverse=True)
    
    # sorted_pop = [org for _, org in ranking]
    sorted_pop = []
    sorted_fit = []
    for fitness, org in ranking:
        sorted_pop.append(org)
        sorted_fit.append(fitness)
    print('sorted_fit:', sorted_fit)
    
    best_fitness = sorted_fit[0]
    
    if motif_n == 1:
        R_seq_list = [org.get_R_sequence_ev() for org in sorted_pop]
        
        # XXX OPTIMIZE THIS CODE! ------------------------
        for idx in range(len(sorted_pop)):
            if sorted_fit[idx] != best_fitness:
                break
        best_organisms_R_seq = R_seq_list[:idx]    
    
    print('\tMax Fitness:', best_fitness)
    print('\tAvg Fitness:', np.array(fitness_list).mean())
    if motif_n == 1:
        print('\tAvg R_sequence:\t', np.array(R_seq_list).mean())
        print('\tAvg best R_sequence:\t', np.array(best_organisms_R_seq).mean())
        print('\tR_frequency:\t', org.get_R_frequency())
    
        if gen % update_period == 0:
            min_Rseq_list.append(np.array(R_seq_list).min())
            avg_Rseq_list.append(np.array(R_seq_list).mean())
            max_Rseq_list.append(np.array(R_seq_list).max())
            best_org_R_seq_list.append(np.array(best_organisms_R_seq).mean())
            best_org_Rseq_ev_list.append(R_seq_list[0])
    
    # Selection
    # ---------
    middle = pop_size//2
    good, bad = sorted_pop[:middle], sorted_pop[middle:]
    
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
    
    # Replacement
    if n_ties == 0:
        population = good + copy.deepcopy(good)
    else:
        population = good + copy.deepcopy(good[:-n_ties]) + bad[:n_ties]
    #population = sorted_pop[middle + n_ties:] + bad[-n_ties:] + sorted_pop[middle:]


# To numpy arrays
min_Rseq = np.array(min_Rseq_list)
avg_Rseq = np.array(avg_Rseq_list)
max_Rseq = np.array(max_Rseq_list)
best_org_R_seq = np.array(best_org_R_seq_list)
best_org_Rseq_ev = np.array(best_org_Rseq_ev_list)


# R_frequence
R_freq = org.get_R_frequency()
# Maximum information possible (for plot)
max_IC_possible = org.max_possible_IC()


'''
for org in population:
    print(org.acgt)

for org in population:
    counts = list(org.acgt.values())
    freqs = [c/config_dict['G'] for c in counts]
    EH = expected_entropy(config_dict['gamma'], base_probabilities=freqs)
    R_freq = org.get_R_frequency()
    R_freq += (2 - EH) * config_dict['motif_len']
    print(R_freq)

for org in population:
    print(org.seq[:100])

'''

# !!! NOT EXACTLY CORRECT:
#     This is using base probabilities 25% each
#     (they may change throughout the simulation)
EH = expected_entropy(config_dict['gamma'])
baseline_info = (2 - EH) * config_dict['motif_len']


# Apply correction
min_Rseq = min_Rseq - baseline_info
avg_Rseq = avg_Rseq - baseline_info
max_Rseq = max_Rseq - baseline_info
best_org_R_seq = best_org_R_seq - baseline_info
best_org_Rseq_ev = best_org_Rseq_ev - baseline_info


# SIMPLE PLOT
#plt.plot(best_org_R_seq, label="R_sequence of best organism")
plt.plot(best_org_Rseq_ev, label="R_sequence of best organism (evo)")
plt.axhline(y=R_freq, color='r', linestyle='-', label="R_frequency")
plt.legend()
plt.title("G={}, gamma={}, pop={}, mot_res={}".format(org.G, org.gamma, pop_size, org.motif_res))
plt.ylim((- baseline_info,max_IC_possible))
plt.show()

'''
filename = run_tag + "_simple_G_{}_gamma_{}_pop_{}_res_{}.png".format(population[i].G, population[i].gamma, pop_size, population[i].motif_res)
plt.savefig("../results/" + filename, bbox_inches='tight')
plt.close()
'''

# PLOT EVERYTHING
plt.plot(min_Rseq, label="Minimum R_sequence")
plt.plot(avg_Rseq, label="Average R_sequence")
plt.plot(max_Rseq, label="Maximum R_sequence")
plt.plot(best_org_R_seq, alpha=0.5, label="R_sequence of best organism")
#plt.plot(best_org_Rseq_ev, alpha=0.9, label="R_sequence of best organism (ev)")
plt.axhline(y=R_freq, color='r', linestyle='-', label="R_frequency")
plt.legend()
plt.title("G={}, gamma={}, pop={}, mot_res={}".format(org.G, org.gamma, pop_size, org.motif_res))
plt.ylim((- baseline_info,max_IC_possible))
plt.show()

'''
filename = run_tag + "G_{}_gamma_{}_pop_{}_res_{}.png".format(population[i].G, population[i].gamma, pop_size, population[i].motif_res)
plt.savefig("../results/" + filename, bbox_inches='tight')
plt.close()
'''































# =============================================================================
#     FIRST METHOD
# =============================================================================


min_Rseq_list = []
avg_Rseq_list = []
max_Rseq_list = []
best_org_R_seq_list = []
best_org_Rseq_ev_list = []

for gen in range(10000):
    print("Gen:", gen)
    
    fitness_list = []
    R_seq_list = []
    
    for i in range(pop_size):
        org = population[i]
        offspring = org.replicate()
        offspring.mutate_ev()
        parent_fitness = org.get_fitness()
        child_fitness = offspring.get_fitness()
        if child_fitness > parent_fitness:
            population[i] = offspring
        elif child_fitness == parent_fitness:
            if random.random() < 0.5:
                population[i] = offspring
        
        fitness_list.append(population[i].get_fitness())
        if motif_n == 1:
            R_seq_list.append(population[i].get_R_sequence())
    
    best_fitness = max(fitness_list)
    if motif_n == 1:
        best_organisms_indexes = [i for i in range(pop_size) if fitness_list[i]==best_fitness]
        # best_organism_index = random.choice(best_organisms_indexes)
        # best_organism_R_seq = R_seq_list[best_organism_index]
        best_organisms_R_seq = [R_seq_list[i] for i in best_organisms_indexes]
    
    print('\tMax Fitness:', max(fitness_list))
    print('\tAvg Fitness:', np.array(fitness_list).mean())
    if motif_n == 1:
        print('\tAvg R_sequence:\t', np.array(R_seq_list).mean())
        print('\tAvg best R_sequence:\t', np.array(best_organisms_R_seq).mean())
        print('\tR_frequency:\t', org.get_R_frequency())
    
        if gen % update_period == 0:
            min_Rseq_list.append(np.array(R_seq_list).min())
            avg_Rseq_list.append(np.array(R_seq_list).mean())
            max_Rseq_list.append(np.array(R_seq_list).max())
            best_org_R_seq_list.append(np.array(best_organisms_R_seq).mean())
            #best_org_Rseq_ev_list.append(R_seq_list[0])
            best_org_Rseq_ev_list.append(random.choice(best_organisms_R_seq))




# To numpy arrays
min_Rseq = np.array(min_Rseq_list)
avg_Rseq = np.array(avg_Rseq_list)
max_Rseq = np.array(max_Rseq_list)
best_org_R_seq = np.array(best_org_R_seq_list)
best_org_Rseq_ev = np.array(best_org_Rseq_ev_list)


# R_frequence
R_freq = org.get_R_frequency()
# Maximum information possible (for plot)
max_IC_possible = org.max_possible_IC()


# !!! NOT EXACTLY CORRECT:
#     This is using base probabilities 25% each
#     (they may change throughout the simulation)
EH = expected_entropy(config_dict['gamma'])
baseline_info = (2 - EH) * config_dict['motif_len']


# Apply correction
min_Rseq = min_Rseq - baseline_info
avg_Rseq = avg_Rseq - baseline_info
max_Rseq = max_Rseq - baseline_info
best_org_R_seq = best_org_R_seq - baseline_info
best_org_Rseq_ev = best_org_Rseq_ev - baseline_info



# SIMPLE PLOT
plt.plot(best_org_R_seq, label="R_sequence of best organism")
plt.axhline(y=R_freq, color='r', linestyle='-', label="R_frequency")
plt.legend()
plt.title("G={}, gamma={}, pop={}, mot_res={}".format(org.G, org.gamma, pop_size, org.motif_res))
plt.ylim((- baseline_info,max_IC_possible))
plt.show()
'''
filename = run_tag + "simple_G_{}_gamma_{}_pop_{}_res_{}.png".format(org.G, org.gamma, pop_size, org.motif_res)
plt.savefig("../results/" + filename, bbox_inches='tight')
plt.close()
'''

# PLOT EVERYTHING
plt.plot(min_Rseq, label="Minimum R_sequence")
plt.plot(avg_Rseq, label="Average R_sequence")
plt.plot(max_Rseq, label="Maximum R_sequence")
plt.plot(best_org_R_seq, alpha=0.5, label="R_sequence of best organism")
plt.axhline(y=R_freq, color='r', linestyle='-', label="R_frequency")
plt.legend()
plt.title("G={}, gamma={}, pop={}, mot_res={}".format(org.G, org.gamma, pop_size, org.motif_res))
plt.ylim((- baseline_info,max_IC_possible))
plt.show()
'''
filename = run_tag + "G_{}_gamma_{}_pop_{}_res_{}.png".format(org.G, org.gamma, pop_size, org.motif_res)
plt.savefig("../results/" + filename, bbox_inches='tight')
plt.close()
'''

























