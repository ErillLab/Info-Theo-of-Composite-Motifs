
'''

Evolve transcriptional regulatory system.

TODO: Rename file 'evolve_reg_sys.py'

'''


# import cProfile
# import numpy as np
import time
import random
import json
import os
import warnings
from genome import Genome



def read_json_file(filename):
    ''' Returns the content of a specified JSON file as a python object. '''
    with open(filename) as json_content:
        return json.load(json_content)

def check_settings(config_dict):
    ''' Raises Errors if the settings are inconsistent. '''
    
    # Check `run_mode`
    if config_dict['run_mode'] not in ['serial', 'parallel']:
        raise ValueError("run_mode should be 'serial' or 'parallel'.")
    
    # Check `fitness_mode`
    if config_dict['fitness_mode'].lower() not in ['errors_penalty', 'auprc']:
        raise ValueError("fitness_mode should be 'errors_penalty' or 'auprc'.")
    
    # Check `targets_type`
    if config_dict['targets_type'] not in ['placements', 'centroids']:
        raise ValueError("targets_type should be 'placements' or 'centroids'.")
    if config_dict['motif_n'] == 1 and config_dict['targets_type'] == 'centroids':
        raise ValueError("targets_type can be set to 'centroids' only when motif_n > 1.")
    
    # Check `spacers`
    if not isinstance(config_dict['spacers'], list):
        raise ValueError("spacers should be a list.")
    else:
        for spacer in config_dict['spacers']:
            if not isinstance(spacer, int):
                raise ValueError("Each spacer value should be an integer.")
    if config_dict["targets_type"] == "placements":
        if len(config_dict['spacers']) != config_dict['gamma']:
            raise ValueError("The list `spacers` specified in the settings " +
                             "should contain one spacer value per site. The " +
                             "number of elements should therefore be equal " +
                             "to the parameter `gamma`.")
    
    # Check `connector_type`
    if config_dict['connector_type'] not in ['uniform', 'gaussian']:
        raise ValueError("connector_type should be 'uniform' or 'gaussian'.")
    
    # Check all `fix_*` parameters
    for key in ['fix_mu', 'fix_sigma', 'fix_left', 'fix_right']:
        if isinstance(config_dict[key], bool):
            if config_dict[key]:
                raise ValueError(key + " should be either a number or None or False")
    
    # Check `mut_mode`
    if config_dict['mut_mode'] not in ['ev', 'rate']:
        raise ValueError("mut_mode should be 'ev' or 'rate'.")
    
    # Check population origin parameters
    if config_dict['pop_origin'] not in ['random', 'files']:
        raise ValueError("pop_origin should be 'random' or 'files'.")
    if config_dict['pop_origin'] == 'files':
        if not isinstance(config_dict['pop_dir_path'], str):
            raise ValueError("pop_dir_path must be a string when pop_origin is 'files'.")

def generate_diad_plcm_map(config_dict):
    '''
    For a dimeric TF there are G^2 possible placements (because each of the two
    elements of the diad can be in G positions). Each of the G^2 possible diad
    placements has a centroid, i.e. the center of the placement (rounded down
    when it is in between two genomic positions).
    
    This function returns a dictionary that has two elements:
    
    (1) KEY:
            "plcm_idx_to_gnom_pos" : string
        VALUE:
            plcm_idx_to_gnom_pos : list (of G^2 integers)
                Maps the i-th diad placement to the position of its centroid.
                The i-th element in the list stores the centroid position.
    
    (2) KEY:
            "gnom_pos_to_plcm_idx" : string
        VALUE:
            `gnom_pos_to_plcm_idx` : list (of G lists)
                Maps each genomic position to the list of all the diad placements
                that have that genomic position as centroid. The diad placements
                are identified by their index [from 0 to (G^2)-1].
    '''
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
    return {'plcm_idx_to_gnom_pos': plcm_idx_to_gnom_pos,
            'gnom_pos_to_plcm_idx': gnom_pos_to_plcm_idx}

def reproduce(organisms):
    ''' The Genome objects in `organisms` (a list) are cloned, and the clones
    (a list) are returned. '''
    return [Genome(clone=parent) for parent in organisms]

def sort_pop_by_fit(population):
    ''' Sorts population based on fitness (descending: from best to worst). Returns
    the sorted population and the corresponding (sorted) list of fitness values. '''
    
    fitness_list = [org.get_fitness() for org in population]
    ranking = sorted(zip(fitness_list, population), key=lambda x: x[0], reverse=True)
    sorted_pop = []
    sorted_fit = []
    for fitness, org in ranking:
        sorted_pop.append(org)
        sorted_fit.append(fitness)
    return sorted_pop, sorted_fit

def export_org_data(org, gen, results_dirpath, files_tag='', verbose=True):
    '''
    Save output files for the organism `org` into the `results_dirpath` folder.
    Four files are generated:
        - *_gaps_report.json
        - *_ic_report.csv
        - *_map.txt
        - *_org.json

    Parameters
    ----------
    org : object of the Genome class
        Organism to be saved.
    gen : int
        Generation number. Used to define the file names.
    results_dirpath : str
        Path of the directory where output files will be saved.
    files_tag : str, optional
        This tag can be used to easily identify output files from special
        circumstances. For example, the tag "sol_latest_" can be used when
        saving the last organism of the run. The default is '' (no tag).
    '''
    # Each output file name starts with the following string
    path_start = os.path.join(results_dirpath, '{}_gen_{}'.format(files_tag, gen))
    # Save files into `results_dirpath`
    org.export(path_start + '_org.json')
    org.make_genome_map(path_start + '_map.txt', verbose)
    # IC report (CSV) and Gaps report (JSON)
    org.study_info(path_start, gen, verbose)

def end_run(gen, solution_gen, drift_time, max_n_gen):
    ''' Returns True if the run reached the end (according to input parameters).
    Returns False otherwise. '''
    if solution_gen:
        # A solution was already found:
        # Stop after `drift_time`
        return gen >= solution_gen + drift_time
    else:
        # No solution found so far:
        # Stop if no solution is found before `max_n_gen`
        return gen > max_n_gen

def is_max_fitness(fitness, fitness_mode):
    ''' Returns True if `fitness` is the maximum possible fitness (perfect classifier). '''
    if fitness_mode == 'errors_penalty':
        return fitness == 0
    elif fitness_mode.lower() == 'auprc':
        return fitness == 1
    else:
        raise ValueError("fitness_mode should be 'errors_penalty' or 'auprc'.")

def initialize_poplutaion(config_dict, diad_plcm_map, verbose):
    '''
    Returns a population (list of Genome objects). The Genomes are random or
    they are read from files, depending on the parameters in the config.
    '''
    pop_size = config_dict['pop_size']
    pop_origin = config_dict['pop_origin']
    pop_dir_path = config_dict['pop_dir_path']
    org_tag = config_dict['org_tag']
    
    # Generate population of random organisms
    if pop_origin == 'random':
        if verbose:
            print('Generating {} random organisms ...'.format(pop_size))
        return [Genome(config_dict, diad_plcm_map) for i in range(pop_size)]
    
    # Import population from files
    elif pop_origin == 'files':
        if verbose:
            print('Importing population from {} ...'.format(pop_dir_path))
        files = os.listdir(pop_dir_path)
        if len(files) != pop_size:
            warnings.warn(("pop_size parameter is set to " + str(pop_size) +
                           ", but " + str(len(files)) + " files were found in " +
                           pop_dir_path + ". Effective pop size in this run will be " +
                           str(len(files))))
        return [Genome(config_dict, diad_plcm_map, input_file=pop_dir_path + '/' + f, org_tag=org_tag) for f in files]

def is_competition(population):
    ''' Returns True if population contains organisms with different tags, which
    is used to keep track of subpopulations during competition experiments.
    Returns False otherwise. '''
    return len(set([org.tag for org in population])) > 1


def main():
    '''
    Main function that implements the evolutionary simulation.
    '''
    
    # SET UP
    
    config_filename = 'config.json'
    config_dict = read_json_file(config_filename)
    check_settings(config_dict)
    
    run_tag = time.strftime("%Y%m%d%H%M%S")
    run_mode = config_dict['run_mode']
    verb = config_dict['print_to_stdout']
    
    # Ensure run_tag uniqueness when running in parallel
    if run_mode == 'parallel':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        run_tag = run_tag + '_' + str(rank)
    
    motif_n = config_dict['motif_n']
    update_period = config_dict['update_period']
    drift_time = config_dict['drift_time']
    max_n_gen = config_dict['max_n_gen']
    fitness_mode = config_dict['fitness_mode']
    mut_mode = config_dict['mut_mode']
    prob_indel = config_dict['prob_indel']
    
    # Results directory
    results_dirpath = '../results/' + run_tag + '/'
    os.makedirs(results_dirpath, exist_ok=True)
    
    # Save the settings for this run
    with open(results_dirpath + 'parameters.json', 'w') as f:
        json.dump(config_dict, f)
    
    if motif_n == 2:
        # Map diad placement indexes to genomic position of centroid
        diad_plcm_map = generate_diad_plcm_map(config_dict)
    else:
        diad_plcm_map = None
    
    # INITIALIZE POPULATION
    population = initialize_poplutaion(config_dict, diad_plcm_map, verb)
    competition_experiment = is_competition(population)
    
    # START EVOLUTIONARY SIMULATION
    solution_gen = None
    gen = 0
    
    # If it's a competition experiment
    if competition_experiment:
        
        # Initialize competition report (CSV file) 
        tag_types = list(set([org.tag for org in population]))
        competition_report_fp = results_dirpath + 'competition_report.csv'
        with open(competition_report_fp, 'w') as f:
            for tag in tag_types:
                f.write(str(tag) + ',')
            f.write('\n')
        
        # Set `mutable_regulator` to False to ensure that the competing
        # strategists don't modify their strategy over time
        mutable_regulator = False
    
    else:
        mutable_regulator = True
    
    # Save initial results for Generation 0
    if verb:
        print("\nGen:", gen)
    sorted_pop, sorted_fit = sort_pop_by_fit(population)
    export_org_data(sorted_pop[0], gen, results_dirpath, 'ev', verb)
    # If it's a competition experiment, write starting frequencies in
    # competition report (CSV file)
    if competition_experiment:
        # Store tag frequencies at Generation 0
        tags = [org.tag for org in population]
        N = len(population)
        tag_freqs = []
        for tag in tag_types:
            tag_freqs.append(tags.count(tag)/N)
        # Append to CSV file
        with open(competition_report_fp, 'a') as f:
            # Write one line of the CSV report
            for freq in tag_freqs:
                f.write(str(freq) + ',')
            f.write('\n')
        if verb:
            print("Sub-populations:", dict(zip(tag_types, tag_freqs)))
    
    while not end_run(gen, solution_gen, drift_time, max_n_gen):
        
        gen += 1
        
        # Avoid second-order selection towards higher IC than necessary
        random.shuffle(population)
        
        # Mutation 
        # --------
        for org in population:
            org.mutate(prob_indel, mut_mode, mutable_regulator)
        
        # Fitness evaluation
        # ------------------
        sorted_pop, sorted_fit = sort_pop_by_fit(population)
        best_fitness = sorted_fit[0]
        
        if verb:
            print("\nGen:", gen)
            print('\tBest organism:')
            print('\t\tfitness =', best_fitness)
            if motif_n == 2 and config_dict['connector_type']=='gaussian':
                bc = sorted_pop[0].regulator['connectors'][0]
                print('\t\tconnector: (mu = {}, sigma = {:.3f})'.format(bc.mu, bc.sigma))
        
        # Selection
        # ---------
        middle = len(sorted_pop)//2
        good, bad = sorted_pop[:middle], sorted_pop[middle:]
        
        # Number of ties
        if sorted_fit[middle-1] == sorted_fit[middle]:
            tie_fit_val = sorted_fit[middle]
            # XXX OPTIMIZE THIS CODE! -----------------------------------------
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
            # -----------------------------------------------------------------
            n_ties = min(n_in_good, n_in_bad)
        else:
            n_ties = 0
        
        # Replacement of bad organisms with good organisms
        if n_ties == 0:
            population = good + reproduce(good)
        else:
            population = good + reproduce(good[:-n_ties]) + bad[:n_ties]
        
        # Store results
        # -------------
        if update_period:
            # Export current best organism
            if gen % update_period == 0:
                export_org_data(population[0], gen, results_dirpath, 'ev', verb)
        
        # Update competition report (CSV file) if it's a competition experiment
        if competition_experiment:
            tags = [org.tag for org in population]
            tag_freqs = []
            for tag in tag_types:
                tag_freqs.append(tags.count(tag)/N)
            # Append to CSV file
            with open(competition_report_fp, 'a') as f:
                # Write one line of the CSV report
                for freq in tag_freqs:
                    f.write(str(freq) + ',')
                f.write('\n')
            if verb:
                print("Sub-populations:", dict(zip(tag_types, tag_freqs)))
                
        # Export first solution
        if is_max_fitness(best_fitness, fitness_mode) and not solution_gen:
            export_org_data(population[0], gen, results_dirpath, 'sol_first', verb)
            solution_gen = gen
    
    # Export latest solution
    if solution_gen:
        export_org_data(population[0], gen, results_dirpath, 'sol_latest', verb)
        print('\nDone. Results in ', results_dirpath)
    else:
        for filename in os.listdir(results_dirpath):
            os.remove(results_dirpath + filename)
        os.rmdir(results_dirpath)
        print('{}: No solution obtained.'.format(results_dirpath))



if __name__ == '__main__':
    
    '''
    # Start a new run as soon as the previous one is done, until the process is killed
    while True:
        main()
    '''
    main()












