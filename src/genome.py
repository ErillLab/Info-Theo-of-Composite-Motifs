
'''
Genome class.

'''


import numpy as np
import random
import copy
import math
import json
import numbers
# import itertools
import collections
import pandas as pd
from Bio import motifs

from connector import ConnectorGauss, ConnectorUnif
from expected_entropy import expected_entropy



class Genome():
    
    def __init__(self, config_dict=None, diad_plcm_map=None, clone=None):
        
        if clone is None:
            self.non_copy_constructor(config_dict, diad_plcm_map)
        else:
            self.copy_constructor(clone)
    
    def non_copy_constructor(self, config_dict, diad_plcm_map):
        
        # Set parameters from config file
        self.G = config_dict['G']
        self.gamma = config_dict['gamma']
        self.mut_rate = config_dict['mut_rate']
        self.motif_len = config_dict['motif_len']
        self.motif_res = config_dict['motif_res']
        self.motif_n = config_dict['motif_n']
        self.threshold_res = config_dict['threshold_res']
        self.max_threshold = config_dict['motif_len'] * 2
        self.min_threshold = -self.max_threshold
        
        if self.motif_n == 2:
            if diad_plcm_map is None:
                raise ValueError("A 'diad_plcm_map' list must be passed.")
            elif not type(diad_plcm_map) is list:
                raise ValueError("diad_plcm_map must be a list.")
            else:
                self._diad_plcm_map = diad_plcm_map
        
        self.pseudocounts = 0.01  # XXX Temporarily hardcoded
        
        self.connector_type = config_dict['connector_type']
        # Gaussian connectors parameters
        self.fix_mu = config_dict['fix_mu']
        self.fix_sigma = config_dict['fix_sigma']
        self.min_mu = config_dict['min_mu']
        self.max_mu = config_dict['max_mu']
        self.min_sigma = 0.01
        self.max_sigma = self.G * 2  # Approximates a uniform over the genome
        self._sigma_vals = np.logspace(
            np.log2(self.min_sigma), np.log2(self.max_sigma), base=2, num=64)
        # Uniform connectors parameters
        self.fix_left  = config_dict['fix_left']
        self.fix_right = config_dict['fix_right']
        self.min_left  = config_dict['min_left']
        self.max_right = config_dict['max_right']
        if self.min_left is None:
            self.min_left = 0
        if self.max_right is None:
            self.max_right = self.G
        
        
        self.seq = None
        self.regulator = None
        self.threshold = None
        self.acgt = None
        
        self._bases = ['a', 'c', 'g', 't']
        self._nucl_to_int = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        
        # Set seq
        self.synthesize_genome_seq()
        self.set_acgt_content()
        
        # Set target sites
        self.targets_type = config_dict['targets_type']
        self.spacers = config_dict['spacers']
        self.targets = None
        self.set_targets()
        
        self.translate_regulator()
    
    def copy_constructor(self, parent):
        
        # Set parameters from parent
        self.G = parent.G
        self.gamma = parent.gamma
        self.mut_rate = parent.mut_rate
        self.motif_len = parent.motif_len
        self.motif_res = parent.motif_res
        self.motif_n = parent.motif_n
        self.threshold_res = parent.threshold_res
        self.max_threshold = parent.max_threshold
        self.min_threshold = parent.min_threshold
        
        if self.motif_n == 2:
            self._diad_plcm_map = parent._diad_plcm_map
        
        self.pseudocounts = parent.pseudocounts
        
        self.connector_type = parent.connector_type
        # Gaussian connectors parameters
        self.fix_mu = parent.fix_mu
        self.fix_sigma = parent.fix_sigma
        self.min_mu = parent.min_mu
        self.max_mu = parent.max_mu
        self.min_sigma = parent.min_sigma
        self.max_sigma = parent.max_sigma
        self._sigma_vals = copy.deepcopy(parent._sigma_vals)
        # Uniform connectors parameters
        self.fix_left = parent.fix_left
        self.fix_right = parent.fix_right
        self.min_left = parent.min_left
        self.max_right = parent.max_right
        
        self.seq = parent.seq
        self.regulator = copy.deepcopy(parent.regulator)
        self.threshold = parent.threshold
        self.acgt = copy.deepcopy(parent.acgt)
        
        self._bases = parent._bases[:]
        self._nucl_to_int = copy.deepcopy(parent._nucl_to_int)
        
        self.targets_type = parent.targets_type
        self.spacers = copy.deepcopy(parent.spacers)
        self.targets = parent.targets[:]
    
    def synthesize_genome_seq(self):
        ''' Sets the `seq` attribute. '''
        self.seq = "".join(random.choices(self._bases, k=self.G))
    
    def get_seq(self):
        '''
        Returns the genome sequence as a string, where the first L-1 letters are
        repeated at the end of the string (where L is the length of each PSSM).
        In this way, the function provides an argument for the pssm.calculate
        function such that it will return G scores, instead of only G-L+1 scores.
        This accounts for genome circularity: PSSMs can be placed across the 'end'.
        '''
        return self.seq + self.seq[:self.motif_len-1]
    
    def get_pwm_gene_len(self):
        return self.motif_res * self.motif_len
    
    def get_conn_gene_len(self):
        
        if self.connector_type == 'gaussian':
            # Encoding mu
            n_mu_vals = self.max_mu - self.min_mu + 1
            # Required number of bp to encode the mu value
            n_bp_for_mu = int(np.ceil(math.log(n_mu_vals, 4)))
            # Encoding sigma
            # We allow for 64 sigma values, spanning (in log space) from 0.01 to G
            # Therefore, we only need 3 bp (because 4^3=64)
            return n_bp_for_mu + 3
        
        elif self.connector_type == 'uniform':
            # Required number of bp to encode any number up to G is:
            # int(np.ceil(math.log(self.G, 4))), therefore, we need twice as
            # much in order to encode both the left and the right bounds of
            # the uniform
            return 2 * int(np.ceil(math.log(self.G, 4)))
    
    def get_threshold_gene_len(self):
        '''
        Returns the required number of bp to encode the threshld value. If we
        want to encode N distinct threshold values we need at least base-4-log(N)
        digits (in base 4). That's the minimum number of bp needed (actually it's
        the closest integer larger than that).
        
        `self.threshold_res` is the number of distinc threshold values within a
        range of 1 bit.

        '''
        # The number of positive threshold values is self.max_threshold * self.threshold_res
        # There are as many negative threshold values. Therefore, the total number
        # of distinct threshold values is 2 * self.max_threshold * self.threshold_res.
        return int(np.ceil(math.log(2 * self.max_threshold * self.threshold_res, 4)))
    
    def get_pwm_gene_pos(self, pwm_number):
        if pwm_number > self.motif_n or pwm_number < 1:
            raise ValueError(('pwm_number should be a value between 1 and ' +
                              str(self.motif_n) + ' (both included).'))
        # Gene coordinates
        offset = (pwm_number - 1) * (self.get_pwm_gene_len() + self.get_conn_gene_len())
        return (offset, offset + self.get_pwm_gene_len())
    
    def get_conn_gene_pos(self, conn_number):
        if conn_number >= self.motif_n:
            if self.motif_n < 2:
                raise ValueError('There are no connectors.')
            else:
                raise ValueError('There are only {} connectors'.format(self.motif_n-1))
        # Gene coordinates
        offset = ((conn_number * self.get_pwm_gene_len()) +
                  ((conn_number - 1) * self.get_conn_gene_len()))
        return (offset, offset + self.get_conn_gene_len())
    
    def get_threshold_gene_pos(self):
        offset = ((self.motif_n * self.get_pwm_gene_len()) +
                  ((self.motif_n - 1) * self.get_conn_gene_len()))
        return (offset, offset + self.get_threshold_gene_len())
        
    def get_non_coding_start_pos(self):
        return self.get_threshold_gene_pos()[1]
    
    def get_pwm_gene_seq(self, pwm_number):
        pwm_start, pwm_stop = self.get_pwm_gene_pos(pwm_number)
        return self.seq[pwm_start:pwm_stop]
    
    def get_conn_gene_seq(self, conn_number):
        conn_start, conn_stop = self.get_conn_gene_pos(conn_number)
        return self.seq[conn_start:conn_stop]
    
    def get_thrsh_gene_seq(self):
        thrsh_start, thrsh_stop = self.get_threshold_gene_pos()
        return self.seq[thrsh_start:thrsh_stop]
    
    def translate_pwm_gene(self, pwm_number):
        gene_seq = self.get_pwm_gene_seq(pwm_number)
        # Swap A with T
        gene_seq = gene_seq.replace("a", "x")
        gene_seq = gene_seq.replace("t", "a")
        gene_seq = gene_seq.replace("x", "t")
        # Swap C with G
        gene_seq = gene_seq.replace("c", "x")
        gene_seq = gene_seq.replace("g", "c")
        gene_seq = gene_seq.replace("x", "g")
        instances = [gene_seq[i:i+self.motif_len] for i in range(0,len(gene_seq),self.motif_len)]
        return motifs.create([inst.upper() for inst in instances])
    
    def _is_number(self, x):
        ''' Checks whether the input is a number (and not a boolean). '''        
        return isinstance(x, numbers.Number) and not isinstance(x, bool)
            
    
    def translate_conn_gene(self, conn_number):
        ''' Returns a connector object with mu and sigma translated from the
        connector gene, unless they are 'fixed' in the settings.
        '''
        
        # UNIFORM connector
        if self.connector_type == 'uniform':
            
            # No gene translation necessary
            if self._is_number(self.fix_left) and self._is_number(self.fix_right):
                return ConnectorUnif(self.fix_left, self.fix_right, self.G, self.motif_len)
            
            # Gene translation
            else:
                gene_seq = self.get_conn_gene_seq(conn_number)
                left_locus, right_locus = gene_seq[:len(gene_seq)//2], gene_seq[len(gene_seq)//2:]
                
                # Define left
                if self._is_number(self.fix_left):
                    left = self.fix_left
                else:
                    # Translate left
                    left = self.min_left + self.nucl_seq_to_int(left_locus)
                    left = left % (self.max_right + 1)
                
                # Define right
                if self._is_number(self.fix_right):
                    right = self.fix_right
                else:
                    # Transalte right
                    right = self.min_left + self.nucl_seq_to_int(right_locus)
                    right = right % (self.max_right + 1)
                
                if left > right:
                    left = right
            
            return ConnectorUnif(left, right, self.G, self.motif_len)
        
        # GAUSSIAN connector
        elif self.connector_type == 'gaussian':
            
            # No gene translation necessary
            if self._is_number(self.fix_mu) and self._is_number(self.fix_sigma):
                return ConnectorGauss(self.fix_mu, self.fix_sigma, self.G, self.motif_len)
            
            # Gene translation
            else:
                gene_seq = self.get_conn_gene_seq(conn_number)
                mu_locus, sigma_locus = gene_seq[:-3], gene_seq[-3:]
                
                # Define mu
                if self._is_number(self.fix_mu):
                    mu = self.fix_mu
                else:
                    # Translate mu
                    if len(mu_locus)==0:
                        mu = self.min_mu  # Fixed-mu case (min_mu = max_mu)
                    else:
                        mu = self.nucl_seq_to_int(mu_locus)
                        if mu > self.max_mu:
                            mu = self.max_mu
                
                # Define sigma
                if self._is_number(self.fix_sigma):
                    sigma = self.fix_sigma
                else:
                    '''
                    # Transalte sigma
                    sigma_idx = self.nucl_seq_to_int(sigma_locus)
                    sigma = self._sigma_vals[sigma_idx]
                    '''
                    
                    # !!! Alternative definition of sigma based on spring constant
                    # 0 <= x <= 5
                    x = 5 * self.nucl_seq_to_int(sigma_locus)/63
                    # 10^-5 <= k <= 1
                    k = 10**(-x)
                    # 0.019235 <= sigma <= 6.082641
                    sigma = 0.019235/(k**(1/2))
                    
                return ConnectorGauss(mu, sigma, self.G, self.motif_len)
        
        else:
            raise ValueError("connector_type should be 'uniform' or 'gaussian'.")
    
    def translate_threshold_gene(self):
        
        #return (self.nucl_seq_to_int(self.get_thrsh_gene_seq()) / self.threshold_res) + self.min_threshold
        '''
        By doing `- self.max_threshold` instead of `+ self.min_threshold`, the
        self.min_threshold attribute is no longer needed ...
        XXX Remove self.min_threshold
        '''
        return (self.nucl_seq_to_int(self.get_thrsh_gene_seq()) / self.threshold_res) - self.max_threshold
    
    def translate_regulator(self):
        ''' Sets the `regulator` attribute. '''
        recog_list = []
        conn_list = []
        for i in range(self.motif_n - 1):
            # Add PWM
            recog_list.append(self.translate_pwm_gene(i+1))
            # Add connector
            conn_list.append(self.translate_conn_gene(i+1))
        # Add last PWM
        recog_list.append(self.translate_pwm_gene(self.motif_n))
        # Translate threshold
        threshold = self.translate_threshold_gene()
        # Set regulator
        self.regulator = {'recognizers': recog_list,
                          'connectors': conn_list,
                          'threshold': threshold}
    
    def set_targets(self):
        ''' Sets the `targets` attribute. '''
        
        if self.targets_type == 'centroids':
            # Avoid setting targets within the coding sequences (the first part of the genome)
            end_CDS = self.get_non_coding_start_pos()
            # Avoid overlapping sites
            tmp = random.sample(range(end_CDS, self.G-self.motif_len, self.motif_len),
                                k=self.gamma)
            tmp.sort()
            for i in range(len(tmp)-1):
                gap = tmp[i+1] - (tmp[i] + self.motif_len)
                tmp[i] += random.randint(0, min(gap, self.motif_len))
            self.targets = tmp
        
        elif self.targets_type == 'placements':
            if self.gamma != len(self.spacers):
                raise ValueError('The number of specified spacers is different ' +
                                 'from the number of requested targets (gamma).')
            
            # Occupancy
            occ = sum([(self.motif_len * 2) + s for s in self.spacers])
            noncod_bp = (self.G - self.get_non_coding_start_pos())
            if occ > noncod_bp:
                raise ValueError('Genome is too short for such spacers.')
            free_bp = noncod_bp - occ
            # Distribute the 'free bp' randomly (from Unif) among the inter-site spaces
            # (there's also a space before the first and after the last site,
            # so it's gamma+1 intervals in total)
            grouped = [int(x) for x in np.random.uniform(0, self.gamma+1, free_bp)]
            intervals = []
            for x in set(grouped):
                intervals.append(grouped.count(x))
            if len(intervals) < self.gamma+1:
                # If some bins are empty the become zeros in the counts vector
                intervals += [0]*(self.gamma+1 - len(intervals))
            # Define targets' placements
            start = self.get_non_coding_start_pos()
            placements = []
            for i in range(self.gamma):                
                left = start + intervals[i]
                right = left + self.motif_len + self.spacers[i]
                placements.append((left, right))  # where the two elements start
                start = right + self.motif_len
            
            # Transform (left, right) placements into a single index up to G^2
            self.targets = [left*self.G + right for (left, right) in placements]
        
        else:
            raise ValueError("targets_type must be 'centroids' or 'placements'.")
    
    def nucl_seq_to_int(self, nucl_seq):
        '''
        Turns DNA sequences into unique integers. The integer is read as base-4
        number, where the digits from 0 to 3 are the four nucleotide letters.
        `nucl_seq`: a string.
        Returns: an integer.
        '''
        nucl_to_int_dict = self._nucl_to_int
        number_base_four = nucl_seq[:]
        for key in nucl_to_int_dict.keys():
            number_base_four = number_base_four.replace(key, str(nucl_to_int_dict[key]))
        number_base_ten = int(number_base_four, 4)
        return number_base_ten
    
    def pwm_scan(self, pwm_number):
        # Generate PSSM
        recog = self.regulator['recognizers'][pwm_number - 1]
        pwm = recog.counts.normalize(pseudocounts=self.pseudocounts)
        pssm = pwm.log_odds()
        # Scan genome
        return pssm.calculate(self.get_seq())
    
    def scan(self):
        '''
        !!! Obsolete function.
        '''
        pwm_arrays = [self.pwm_scan(i+1) for i in range(self.motif_n)]
        
        # Shortcut for the single motif case
        if self.motif_n == 1:
            return list(np.argwhere(pwm_arrays[0] > self.regulator['threshold']).flatten())
        
        # Shortcut for the diad case
        elif self.motif_n == 2:
            
            # The distance between the two recognizers is r - q
            # Genome is circular, so distances are ambiguous.
            # We chose non-negative distances.
            # [e.g., the distance between 8 (left) and 2 (right) on a genome
            # of length 10 is 4, instead of -6]
            # So the effective distance will be (j-i) % G, instead of j-i.
            # [e.g., (2-8)%10 = 4, instead of 2-8 = -6]
            _G = self.G
            x1 = np.repeat(pwm_arrays[0], _G)
            x2 = np.tile(pwm_arrays[1], _G)
            x3 = np.array([self.regulator['connectors'][0].get_score((j - i) % _G) for i in range(_G) for j in range(_G)])
            
            plcm_scores = x1 + x2 + x3
            
            hits_indexes = np.argwhere(plcm_scores > self.regulator['threshold']).flatten()
            
            # 'position' is the placement 'center'
            mot_len = self.motif_len
            
            plcm_pos = []
            for idx in hits_indexes:
                left, right = divmod(idx, _G)
                if right < left:
                    right += _G
                plcm_pos.append(int((left + right + mot_len)/2) % _G)
            
            return plcm_pos
        
        # Code for the general case (works for any value of `motif_n`)
        else:
            # !!! TO BE RE-CODED
            
            # plcm_pwm_scores = list(itertools.product(*pwm_arrays))
            # plcm_pwm_pos = list(itertools.product(range(self.G), repeat=self.motif_n))
            # plcm_spcr_scores = []
            
            # for plcm in plcm_pwm_pos:
            #     distances = [t - s for s, t in zip(plcm, plcm[1:])]
            #     connscores = [self.regulator['connectors'][i].get_score(distances[i]) for i in range(len(distances))]
            #     plcm_spcr_scores.append(connscores)
            
            # plcm_scores = []
            # plcm_pos = []
            # for i in range(len(plcm_pwm_scores)):
            #     plcm_scores.append(sum(plcm_pwm_scores[i]) + sum(plcm_spcr_scores[i]))
                
                
            #     # !!! WRONG
            #     plcm_pos.append(int((plcm_pwm_pos[i][0] + plcm_pwm_pos[i][-1] + self.motif_len)/2) % self.G)  # site center
                
                
            # hits_indexes = np.argwhere(np.array(plcm_scores) > self.regulator['threshold']).flatten()
            # return [plcm_pos[idx] for idx in hits_indexes]
            raise ValueError('This code needs to be re-coded.')

    def get_fitness(self):
        hits_positions = self.scan()
        # Fitness is (-1) * number-of-errors
        return -(self.count_false_positives(hits_positions) +
                 self.count_false_negatives(hits_positions))
    
    def count_false_positives(self, hits_positions):
        ''' Returns the number of False Positives. '''
        # Count type_I_errors
        type_I_errors  = set(hits_positions).difference(set(self.targets))
        if self.motif_n == 1:
            return len(type_I_errors)
        else:
            n_fp = sum([hits_positions.count(err) for err in type_I_errors])
            # By def, there are only gamma correct placements. Enforce by counting
            # redundant placements as false positives (fp)
            for target in self.targets:
                if hits_positions.count(target) > 1:
                    n_fp += hits_positions.count(target) - 1
            return n_fp
    
    def count_false_negatives(self, hits_positions):
        ''' Returns the number of False Negatives. '''
        # Count type_II_errors
        return len(set(self.targets).difference(set(hits_positions)))
    
    # ===========
    # NEW FITNESS
    # ===========
    
    def get_fitness_new(self):
        '''
        !!! Work in progress ...
        '''
        
        # Scan genome
        # -----------
        
        pwm_arrays = [self.pwm_scan(i+1) for i in range(self.motif_n)]
        
        # Single motif case
        if self.motif_n == 1:
            
            # Define `hits_positions`
            # -----------------------
            hits_positions =  list(np.argwhere(pwm_arrays[0] > self.regulator['threshold']).flatten())
            
            # Calculate fitness
            # -----------------
            return -(self.count_false_positives(hits_positions) +
                     self.count_false_negatives(hits_positions))
        
        # Diad case
        elif self.motif_n == 2:
            
            # Define `hits_positions`
            # -----------------------
            
            # The distance between the two recognizers is r - q
            # Genome is circular, so distances are ambiguous.
            # We chose non-negative distances. [e.g., the distance between
            # 8 (left) and 2 (right) on a genome of length 10 is 4, instead of -6]
            # So the effective distance will be (j-i) % G, instead of j-i.
            # [e.g., (2-8)%10 = 4, instead of 2-8 = -6]
            _G = self.G
            x1 = np.repeat(pwm_arrays[0], _G)
            x2 = np.tile(pwm_arrays[1], _G)
            x3 = np.array([self.regulator['connectors'][0].get_score((j - i) % _G) for i in range(_G) for j in range(_G)])
            
            plcm_scores = x1 + x2 + x3
            
            hits_indexes = np.argwhere(plcm_scores > self.regulator['threshold']).flatten()
            
            # Calculate fitness
            # -----------------
            
            if self.targets_type == 'placements':
                
                # False Positives penalty (penalty is 1 per FP)
                fp_penalty = len(set(hits_indexes).difference(set(self.targets)))
                
                # False Negatives penalty (penalty is between 1 and 2 per FN)
                fn_penalty = 0
                tr = self.regulator['threshold']
                for missed in list(set(self.targets).difference(set(hits_indexes))):
                    # score
                    s = plcm_scores[missed]
                    # False Negatives Penalty (penalty is 1 per FN)
                    fn_penalty += 1
                    # Extra FN penalty based on the score (between 0 and 1 per FN)
                    fn_penalty += self.extra_FN_penalty(s, tr)
                
                return -(fp_penalty + fn_penalty)
            
            elif self.targets_type == 'centroids':
                
                # 'position' is the placement 'centroid'
                hits_positions = []
                for idx in hits_indexes:
                    left, right = divmod(idx, _G)
                    if right < left:
                        right += _G
                    hits_positions.append(int((left + right + self.motif_len)/2) % _G)
                
                # False Positives penalty (penalty is 1 per FP)
                fp_penalty = self.count_false_positives(hits_positions)
                
                # False Negatives penalty (penalty is between 0 and 1 per FN)
                fn_penalty = 0
                tr = self.regulator['threshold']
                # Candidate placements on targets
                for missed_target in list(set(self.targets).difference(set(hits_positions))):
                    
                    # Maximum score among the placements that map onto that genomic position
                    ms = max(map(plcm_scores.__getitem__, self._diad_plcm_map[missed_target]))
                    # False Negatives Penalty (penalty is 1 per FN)
                    fn_penalty += 1
                    # Extra FN penalty based on the score (between 0 and 1 per FN)
                    #####fn_penalty += (tr-ms)/(tr-ms+1)
                    fn_penalty += self.extra_FN_penalty(ms, tr)
                
                return -(fp_penalty + fn_penalty)
        
        # Code for the general case (works for any value of `motif_n`)
        else:
            raise ValueError('This code needs to be re-coded.')
    
    def extra_FN_penalty(self, score, threshold):
        '''
        Extra penalty to apply to false negatives (FNs). FNs are targets with a
        score below the threshold. Instead of counting as a penalty of just
        1 point, the extra penalty penalizes FNs even more, based on how far
        the score of the FN was from reaching the threshold.
        
         - As the the score approaches the threshold, the extra penalty
           approaches 0.
         - As the difference between the threshold and the score increases, the
           extra penalty approaches 1.
        '''
        if score == -np.inf:
            return 1
        else:
            return (threshold-score)/(threshold-score+1)
    
    def mutate_base(self, base_position):
        ''' Point mutation of a randomly chosen nucleotide. '''
        curr_base = self.seq[base_position]
        new_base = random.choice(self._bases)
        self.seq = self.seq[:base_position] + new_base + self.seq[base_position+1:]
        # Update ACGT content
        self.acgt[curr_base] -= 1
        self.acgt[new_base] += 1
    
    # ==== INDELS ====================
    # ================================
    
    def insert_base(self):
        ''' Inserts a random nucleotide. '''
        # Randomly choose a position (outside the regulator CDS)
        pos = random.randint(self.get_non_coding_start_pos(), self.G-1)
        # Randomly choose a base
        base = random.choice(self._bases)
        # Update genome sequence
        self.seq = self.seq[:pos] + base + self.seq[pos:]
        # Update coordinates
        for i in range(len(self.targets)):
            if self.targets[i] >= pos:
                self.targets[i] += 1
        # Update ACGT content
        self.acgt[base] += 1
    
    def delete_base(self):
        ''' Deletes a random nucleotide. '''
        # Randomly chose a position (outside the regulator CDS)
        pos = random.randint(self.get_non_coding_start_pos(), self.G-1)
        # Targeted base
        base = self.seq[pos]
        # Update genome sequence
        self.seq = self.seq[:pos] + self.seq[pos+1:]
        # Update coordinates
        for i in range(len(self.targets)):
            if self.targets[i] > pos:
                self.targets[i] -= 1
        # Update ACGT content
        self.acgt[base] -= 1
    
    def apply_indel(self):
        ''' Applies one insertion and one deletion. In this way, the value of G
        (the length of the genome) is preserved. '''
        self.insert_base()
        self.delete_base()
    
    # ================================
    # ================================
    
    
    def mutate_ev(self):
        rnd_pos = random.randint(0, self.G - 1)
        self.mutate_base(rnd_pos)
        # !!!
        if rnd_pos < self.get_non_coding_start_pos():
            self.translate_regulator()
    
    def mutate_with_rate(self):
        '''
        Alternative mutation strategy, based on a mutation rate. Instead of one
        mutation per organism per generation, the number of mutations is a random
        number that depends on the mutation rate.
        '''
        n_mut_bases = np.random.binomial(self.G, self.mut_rate)
        #n_mut_bases = int(self.G * self.mut_rate)
        #n_mut_bases = np.random.poisson(self.G * self.mut_rate)
        if n_mut_bases > 0:
            mut_bases_positions = random.sample(range(self.G), k=n_mut_bases)
            for pos in mut_bases_positions:
                self.mutate_base(pos)
            
            if min(mut_bases_positions) < self.get_non_coding_start_pos():
                self.translate_regulator()
    
    def set_acgt_content(self, both_strands=False):
        ''' Sets the `acgt` attribute. '''
        a = self.seq.count('a')
        c = self.seq.count('c')
        g = self.seq.count('g')
        t = self.G - (a+c+g)
        self.acgt = {'a': a, 'c': c, 'g': g, 't': t}   
    
    # def get_R_sequence_old(self):
    #     '''
    #     Older version of the function: Background frequencies are fixed at 0.25.
    #     Check new version of this function: "get_R_sequence_ev".
    #     '''
    #     target_sequences = [self.get_seq()[pos:pos+self.motif_len] for pos in self.targets]
    #     H = 0
    #     for i in range(self.motif_len):
    #         obs_bases = [target_seq[i] for target_seq in target_sequences]
    #         counts = {}
    #         for base in self._bases:
    #             counts[base] = obs_bases.count(base)
    #         frequencies = np.array(list(counts.values()))/sum(counts.values())
    #         for f in frequencies:
    #             if f != 0:
    #                 H -= f * np.log2(f)
    #     return (2 * self.motif_len) - H    
    
    def get_R_sequence_ev(self):
        '''
        Background frequencies as in "Information Content of Binding Sites on
        Nucleotide Sequences" Schneider, Stormo, Gold, Ehrenfeucht.
        This is the method used in "Evolution of biological information" (Schneider, 2000).
        '''
        self.set_acgt_content()  # Update A/C/G/T content
        
        target_sequences = [self.get_seq()[pos:pos+self.motif_len] for pos in self.targets]
        Rsequence = 0
        for i in range(self.motif_len):
            obs_bases = [target_seq[i] for target_seq in target_sequences]
            for base in self._bases:
                freq = obs_bases.count(base) / self.gamma
                if freq != 0:
                    bg_freq = self.acgt[base] / self.G
                    Rsequence += freq * (np.log2(freq) - np.log2(bg_freq))
        return Rsequence
    
    # def get_R_sequence_ev_new(self):
    #     '''
    #     !!! Work in progress ...
        
    #     Function that takes into account small sample bias.
    #     As described in "Evolution of biological information".
    #     '''
    #     self.set_acgt_content()  # Update A/C/G/T content
        
    #     Hg = 0
    #     for base in self._bases:
    #         p = self.acgt[base] / self.G
    #         if p != 0:
    #             Hg -= p * np.log2(p)
    #     # !!!
    #     # Skip for now: correction is negligible for large genomes
    #     # Hg += e(self.G)
    #     Hbefore = Hg * self.motif_len
        
    #     Hafter = 0
        
        
        
        
    
    def get_R_frequency(self):
        return -np.log2(self.gamma/(self.G**self.motif_n))
    
    def max_possible_IC(self):
        ''' Maximum information possible for the (composite) motif of the regulator,
        i.e., 2L*number_of_PSSMs + log2(G)*number_of_spacers. '''
        return (2 * self.motif_len * self.motif_n) + (np.log2(self.G) * (self.motif_n - 1))
    
    def export(self, outfilepath=None):
        '''
        Exports the organism as a JSON file. If the path of the output file
        `outfilepath` is not specified, a python dictionary is returned, instead.
        '''
        out_dict = {'seq': self.seq,
                    'G': self.G,
                    'gamma': self.gamma,
                    'targets': self.targets,
                    'motif_len': self.motif_len,
                    'motif_res': self.motif_res,
                    'motif_n': self.motif_n,
                    'threshold_res': self.threshold_res,
                    'min_mu': self.min_mu, 'max_mu': self.max_mu,
                    'min_sigma': self.min_sigma, 'max_sigma': self.max_sigma,
                    'pseudocounts': self.pseudocounts}
        if self.motif_n == 2:
            if self.connector_type == 'gaussian':
                out_dict['mu']    = self.regulator['connectors'][0].mu
                out_dict['sigma'] = self.regulator['connectors'][0].sigma
            elif self.connector_type == 'uniform':
                out_dict['min_gap'] = self.regulator['connectors'][0].min_gap
                out_dict['max_gap'] = self.regulator['connectors'][0].max_gap
        if outfilepath:
            with open(outfilepath, 'w') as f:
                json.dump(out_dict, f)
        else:
            return out_dict
    
    def _get_gene_string(self, name, length):
        ''' Called by `print_genome_map`. Returns a string of the given length
        that represents a gene as a name followed by an arrow. '''
        name += '-' * (length - len(name))
        return name[:length-1] + '>'
    
    def _add_diad_plcm_line(self, outlist, elements_pos, elements_idx):
        ''' Called by `print_genome_map`.
        !!! ... complete documentation ...
        '''
        leftovers = []
        leftovers_idx = []
        for i in range(len(elements_pos)):
            l, r = elements_pos[i]
            idx = elements_idx[i]
            
            if (set(outlist[l:l+self.motif_len]) != {' '} or
                set(outlist[r:r+self.motif_len]) != {' '}):
                leftovers.append((l,r))
                leftovers_idx.append(idx)
            else:
                outlist[l:l+self.motif_len] = [c for c in str(idx+1)] + ['L'] * (self.motif_len - len(str(idx+1)))
                outlist[r:r+self.motif_len] = [c for c in str(idx+1)] + ['R'] * (self.motif_len - len(str(idx+1)))
        return outlist, leftovers, leftovers_idx
    
    def print_genome_map(self, outfilepath=None):
        '''
        Prints the genome map to standard output and, if specified, it writes it
        into an output file with path `outfilepath`. The map includes the location
        of genes and target positions for transcriptional regulation, as well as
        the actual binding positions of the encoded transcriptional regulator.
        '''
        # Annotate genes
        # --------------
        out_string = ''
        prev_stop = 0
        for i in range(self.motif_n):
            # Print PWM gene
            start, stop = self.get_pwm_gene_pos(i+1)
            if start != prev_stop:
                raise ValueError('Inconsistent gene map.')            
            out_string += self._get_gene_string('PWM' + str(i+1), stop - start)
            prev_stop = stop
            # Print Connector gene
            if i+1 > self.motif_n-1:
                break
            start, stop = self.get_conn_gene_pos(i+1)
            if start != prev_stop:
                raise ValueError('Inconsistent gene map.')
            out_string += self._get_gene_string('CON' + str(i+1), stop - start)
            prev_stop = stop
        # Print Threshold gene
        start, stop = self.get_threshold_gene_pos()
        if start != prev_stop:
            raise ValueError('Inconsistent gene map.')
        out_string += self._get_gene_string('THRS', stop - start)
        prev_stop = stop
        
        # Annotate targets
        # ----------------
        if self.targets_type == 'centroids':
            for i, pos in enumerate(self.targets):
                out_string += ' ' * (pos - prev_stop) + str(i+1)
                prev_stop = pos + len(str(i+1))
            out_string += ' ' * (self.G - prev_stop) + '\n'
        
        elif self.targets_type == 'placements':
            mot_len = self.motif_len
            for i, idx in enumerate(self.targets):
                left, right = divmod(idx, self.G)
                out_string += ' ' * (left - prev_stop)
                out_string += (str(i+1) + 'L'*mot_len)[:mot_len]
                out_string += ' ' * (right - (left+mot_len))
                out_string += (str(i+1) + 'R'*mot_len)[:mot_len]
                prev_stop = right + mot_len
            out_string += ' ' * (self.G - prev_stop) + '\n'
        
        # Genome sequence
        # ---------------
        out_string += self.seq + '\n'
        
        # Annotate hits placements
        # ------------------------
        pwm_arrays = [self.pwm_scan(i+1) for i in range(self.motif_n)]
        
        # Shortcut for the single motif case
        if self.motif_n == 1:
            hits_positions = list(np.argwhere(pwm_arrays[0] > self.regulator['threshold']).flatten())
        
        # Shortcut for the diad case
        elif self.motif_n == 2:
            
            # The distance between the two recognizers is r - q
            # Genome is circular, so distances are ambiguous.
            # We chose non-negative distances.
            # [e.g., the distance between 8 (left) and 2 (right) on a genome
            # of length 10 is 4, instead of -6]
            # So the effective distance will be (j-i) % G, instead of j-i.
            # [e.g., (2-8)%10 = 4, instead of 2-8 = -6]
            _G = self.G
            x1 = np.repeat(pwm_arrays[0], _G)
            x2 = np.tile(pwm_arrays[1], _G)
            x3 = np.array([self.regulator['connectors'][0].get_score((j - i) % _G) for i in range(_G) for j in range(_G)])
            plcm_scores = x1 + x2 + x3
            
            hits_indexes = np.argwhere(plcm_scores > self.regulator['threshold']).flatten()
            
            hits_positions = []
            elements_pos = []
            for hit_idx in hits_indexes:
                left, right = divmod(hit_idx, _G)
                elements_pos.append((left, right))
                if right < left:
                    right += _G
                hits_positions.append(int((left + right + self.motif_len)/2) % _G)
                
        # General case
        else:
            raise ValueError('To be coded ...')
        
        # Hits positions (center)
        prev_stop = 0
        for idx, pos in enumerate(hits_positions):
            out_string += ' ' * (pos - prev_stop) + str(idx+1)
            prev_stop = pos + len(str(idx+1))
        out_string += ' ' * (_G - prev_stop) + '\n'
        
        # Hits placements
        if self.motif_n == 2:
            #plcm_list = [' '] * _G
            to_be_printed = elements_pos
            to_be_printed_idx = list(range(len(elements_pos)))
            while len(to_be_printed) > 0:
                new_plcm_list, to_be_printed, to_be_printed_idx = self._add_diad_plcm_line(
                    [' '] * _G, to_be_printed, to_be_printed_idx)
                out_string += ''.join(new_plcm_list) + '\n'
        
        # End of diagram
        out_string += '-' * self.G + '\n'
        
        # Write to specified output file
        if outfilepath:
            with open(outfilepath, 'w') as f:
                f.write(out_string)
        
        # Print to standard output
        print(out_string)
    
    def study_diad(self, outfilepath=None):
        
        if self.motif_n != 2:
            raise ValueError("The study_diad function is meant for diad-based regulators " +
                             "(motif_n should be 2). 'motif_n' is " + str(self.motif_n))
        
        pwm_arrays = [self.pwm_scan(i+1) for i in range(self.motif_n)]
        _G = self.G
        x1 = np.repeat(pwm_arrays[0], _G)
        x2 = np.tile(pwm_arrays[1], _G)
        x3 = np.array([self.regulator['connectors'][0].get_score((j - i) % _G) for i in range(_G) for j in range(_G)])
        plcm_scores = x1 + x2 + x3
        
        hits_indexes = np.argwhere(plcm_scores > self.regulator['threshold']).flatten()
        
        # Case without hits
        if len(hits_indexes) == 0:
            if outfilepath:
                
                # Save (empty) IC report
                ic_report = pd.DataFrame(
                    {'Rseq1': [0, 0],
                     'Rseq2': [0, 0],
                     'Rspacer': [0, 0],
                     'Rtot': [0, 0],
                     'Rfrequency': [self.get_R_frequency(), self.get_R_frequency()]})
                ic_report.index = ['corrected', 'uncorrected']
                ic_report.to_csv(outfilepath + '_ic_report.csv')
                
                # Save (empty) Gaps report
                with open(outfilepath + '_gaps_report.json', 'w') as f:
                    json.dump([], f)
            return
        
        hits_positions = []
        elements_pos = []
        for hit_idx in hits_indexes:
            left, right = divmod(hit_idx, _G)
            elements_pos.append((left, right))
            if right < left:
                right += _G
            hits_positions.append(int((left + right + self.motif_len)/2) % _G)
        
        # Rsequence of the two elements
        
        L = self.motif_len
        
        # PWM1 binding sites
        pwm1_tg_pos = [e[0] for e in elements_pos]
        pwm1_tg_seq = [self.get_seq()[pos:pos+L] for pos in pwm1_tg_pos]
        
        # PWM2 binding sites
        pwm2_tg_pos = [e[1] for e in elements_pos]
        pwm2_tg_seq = [self.get_seq()[pos:pos+L] for pos in pwm2_tg_pos]
        
        # R_sequence
        
        # !!! NOT EXACTLY CORRECT:
        #     This is using base probabilities 25% each
        #     (they may change throughout the simulation)
        EH = expected_entropy(self.gamma)
        baseline_info = (2 - EH) * L
        
        # Rsequence(1)
        Rseq1 = 0
        for i in range(L):
            obs_bases = [target_seq[i] for target_seq in pwm1_tg_seq]
            
            for base in self._bases:
                freq = obs_bases.count(base) / len(obs_bases)
                if freq != 0:
                    bg_freq = self.acgt[base] / self.G
                    Rseq1 += freq * (np.log2(freq) - np.log2(bg_freq))
        
        # Rsequence(2)
        Rseq2 = 0
        for i in range(L):
            obs_bases = [target_seq[i] for target_seq in pwm2_tg_seq]
            
            for base in self._bases:
                freq = obs_bases.count(base) / len(obs_bases)
                if freq != 0:
                    bg_freq = self.acgt[base] / self.G
                    Rseq2 += freq * (np.log2(freq) - np.log2(bg_freq))
        
        # Rspacer
        gaps = [(r - l) % _G - L for l, r in elements_pos]
        counter = collections.Counter(gaps)
        gap_counts = np.array(list(counter.values()))
        gap_freqs = gap_counts / len(gaps)
        gap_H = - sum([f * np.log2(f) for f in gap_freqs])
        Rspacer = np.log2(_G) - gap_H
        
        # IC report
        # print('  Rseq1 + Rseq2 + Rspacer =\n= {:.3f} + {:.3f} + {:.3f}   = {:.3f}  ~  {} = Rfreq\n'.format(
        #     Rseq1-baseline_info, Rseq2-baseline_info, Rspacer,
        #     Rseq1-baseline_info + Rseq2-baseline_info + Rspacer, self.get_R_frequency()))
        
        # Save IC report
        if outfilepath:
            ic_report = pd.DataFrame(
                {'Rseq1': [Rseq1-baseline_info, Rseq1],
                 'Rseq2': [Rseq2-baseline_info, Rseq2],
                 'Rspacer': [Rspacer, Rspacer],
                 'Rtot': [Rseq1+Rseq2+Rspacer-(2*baseline_info), Rseq1+Rseq2+Rspacer],
                 'Rfrequency': [self.get_R_frequency(), self.get_R_frequency()]})
            ic_report.index = ['corrected', 'uncorrected']
            ic_report.to_csv(outfilepath + '_ic_report.csv')
        
        # Gaps report
        # print('Gaps: {}\nConnector: mu={}, sigma={}\n'.format(
        #     gaps, self.regulator['connectors'][0].mu, self.regulator['connectors'][0].sigma))
        
        # Save gaps report
        if outfilepath:
            with open(outfilepath + '_gaps_report.json', 'w') as f:
                json.dump([int(gap) for gap in gaps], f)
        
        
        
        

                    












