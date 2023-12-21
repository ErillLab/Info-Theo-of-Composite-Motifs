# -*- coding: utf-8 -*-

import cProfile
import time

import random
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import copy

from genome import Genome
from expected_entropy import expected_entropy


def read_json_file(filename):
    with open(filename) as json_content:
        return json.load(json_content)




# =============================================================================
#     ORIGINAL METHOD
# =============================================================================


def main():
    # SET UP
    
    config_filename = 'config.json'
    
    run_tag = time.strftime("%Y%m%d%H%M%S")
    config_dict = read_json_file(config_filename)
    pop_size = config_dict['pop_size']
    motif_n = config_dict['motif_n']
    update_period = config_dict['update_period']
    population = [Genome(config_dict) for i in range(pop_size)]
    
    # START
    
    min_Rseq_list = []
    avg_Rseq_list = []
    max_Rseq_list = []
    best_org_Rseq_list = []
    best_org_Rseq_ev_list = []
    
    start = time.time()
    for gen in range(10):
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
            fitness_list.append(org.get_fitness())
        
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
        print('\tAvg Fitness:', np.array(fitness_list).mean())
        
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
            population = good + copy.deepcopy(good)
        else:
            population = good + copy.deepcopy(good[:-n_ties]) + bad[:n_ties]
    end = time.time()
    print(end-start)

if __name__ == '__main__':
    
    cProfile.run('main()', sort='tottime')


"""

# To numpy arrays
min_Rseq = np.array(min_Rseq_list)
avg_Rseq = np.array(avg_Rseq_list)
max_Rseq = np.array(max_Rseq_list)
best_org_Rseq = np.array(best_org_Rseq_list)
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
best_org_Rseq = best_org_Rseq - baseline_info
best_org_Rseq_ev = best_org_Rseq_ev - baseline_info


# SIMPLE PLOT
#plt.plot(best_org_Rseq, label="R_sequence of best organism")
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
plt.plot(best_org_Rseq, alpha=0.5, label="R_sequence of best organism")
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



# ===========
# STUDY DIADS
# ===========


import itertools

def study_diad(org):
    
    # HITS
    pwm_arrays = [org.pwm_scan(i+1) for i in range(org.motif_n)]
    plcm_pwm_scores = list(itertools.product(*pwm_arrays))
    plcm_scores = []
    plcm_pos = []
    pwms_pos = []
    for i in range(len(plcm_pwm_scores)):
        q, r = divmod(i, org.G)
        pwms_pos.append((q, r))
        # The distance between the two recognizers is r - q
        # Genome is circular, so distances are ambiguous.
        # We chose non-negative distances.
        # (e.g. the distance between 8 and 2 on a genome of length 10 is 4, instead of -6)
        # So the effective distance will be (r-q) % G, instead of r-q.
        plcm_scores.append(sum(plcm_pwm_scores[i]) + org.regulator['connectors'][0].score((r - q) % org.G))
        plcm_pos.append(int((r + q + org.motif_len)/2))  # motif center
    hits_indexes = np.argwhere(np.array(plcm_scores) > org.regulator['threshold']).flatten()
    hits = [plcm_pos[idx] for idx in hits_indexes]
    elements_pos = [pwms_pos[idx] for idx in hits_indexes]
    
    # TARGTETS
    targets = org.targets
    
    # CORRECT
    set(hits).intersection(set(targets))
    
    # PWM1 binding sites
    pwm1_tg_pos = [e[0] for e in elements_pos]
    pwm1_tg_seq = [org.seq[pos:pos+org.motif_len] for pos in pwm1_tg_pos]
    
    # PWM2 binding sites
    pwm2_tg_pos = [e[1] for e in elements_pos]
    pwm2_tg_seq = [org.seq[pos:pos+org.motif_len] for pos in pwm2_tg_pos]
    
    # R_sequence
    
    # !!! NOT EXACTLY CORRECT:
    #     This is using base probabilities 25% each
    #     (they may change throughout the simulation)
    EH = expected_entropy(config_dict['gamma'])
    baseline_info = (2 - EH) * config_dict['motif_len']
    
    # Rsequence1
    Rseq1 = 0
    for i in range(org.motif_len):
        obs_bases = [target_seq[i] for target_seq in pwm1_tg_seq]
        
        for base in org._bases:
            freq = obs_bases.count(base) / len(obs_bases)
            if freq != 0:
                bg_freq = org.acgt[base] / org.G
                Rseq1 += freq * (np.log2(freq) - np.log2(bg_freq))
    
    # Rsequence2
    Rseq2 = 0
    for i in range(org.motif_len):
        obs_bases = [target_seq[i] for target_seq in pwm2_tg_seq]
        
        for base in org._bases:
            freq = obs_bases.count(base) / len(obs_bases)
            if freq != 0:
                bg_freq = org.acgt[base] / org.G
                Rseq2 += freq * (np.log2(freq) - np.log2(bg_freq))
    
    # Rspacer
    gaps = []
    for l, r in elements_pos:
        distance = r - l
        gap = distance - org.motif_len
        gaps.append(gap)
    print('gaps:', gaps)
    conn_obj = org.regulator['connectors'][0]
    print('\nconnector: mu={}, sigma={}\n'.format(conn_obj.mu, conn_obj.sigma))
    
    import collections
    counter = collections.Counter(gaps)
    gap_counts = np.array(list(counter.values()))
    gap_freqs = gap_counts / len(gaps)
    gap_H = - sum([f * np.log2(f) for f in gap_freqs])
    Rspacer = np.log2(org.G) - gap_H
    
    print('  Rseq1 + Rseq2 + Rspacer =\n= {:.3f} + {:.3f} + {:.3f}   = {:.3f}  ~  {} = Rfreq\n'.format(
        Rseq1, Rseq2, Rspacer, Rseq1 + Rseq2 + Rspacer, org.get_R_frequency()))
    
    # CORRECTED
    print('After correction:')
    print('  Rseq1 + Rseq2 + Rspacer =\n= {:.3f} + {:.3f} + {:.3f}   = {:.3f}  ~  {} = Rfreq\n'.format(
        Rseq1-baseline_info, Rseq2-baseline_info, Rspacer,
        Rseq1-baseline_info + Rseq2-baseline_info + Rspacer, org.get_R_frequency()))




print('fitness_list:', fitness_list)

# Select diad organisms to be studied
indexes = []
for idx in range(len(fitness_list)):
    if fitness_list[idx] == 0:
        indexes.append(idx)
org_to_study = [population[idx] for idx in indexes]

for i, organism in enumerate(org_to_study):
    print('=' * 10)
    print('  Org', i)
    print('=' * 10 + '\n')
    study_diad(organism)

    






# =============================================================================
#     FIRST METHOD
# =============================================================================


min_Rseq_list = []
avg_Rseq_list = []
max_Rseq_list = []
best_org_Rseq_list = []
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
            best_org_Rseq_list.append(np.array(best_organisms_R_seq).mean())
            #best_org_Rseq_ev_list.append(R_seq_list[0])
            best_org_Rseq_ev_list.append(random.choice(best_organisms_R_seq))




# To numpy arrays
min_Rseq = np.array(min_Rseq_list)
avg_Rseq = np.array(avg_Rseq_list)
max_Rseq = np.array(max_Rseq_list)
best_org_Rseq = np.array(best_org_Rseq_list)
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
best_org_Rseq = best_org_Rseq - baseline_info
best_org_Rseq_ev = best_org_Rseq_ev - baseline_info



# SIMPLE PLOT
plt.plot(best_org_Rseq, label="R_sequence of best organism")
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
plt.plot(best_org_Rseq, alpha=0.5, label="R_sequence of best organism")
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




#





"""





















