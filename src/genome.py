
'''
Genome class.

'''


import numpy as np
import random
import copy
import math
import json
import numbers
#import itertools
import collections
import pandas as pd
from Bio import motifs
from sklearn.metrics import auc, precision_recall_curve


from connector import ConnectorGauss, ConnectorUnif
from expected_entropy import expected_entropy, entropy



class Genome():
    
    def __init__(self, config_dict=None, diad_plcm_map=None, clone=None):
        
        if clone is None:
            self.non_copy_constructor(config_dict, diad_plcm_map)
        else:
            self.copy_constructor(clone)
    
    def non_copy_constructor(self, config_dict, diad_plcm_map):
        
        # Set parameters from config file
        self.fitness_mode = config_dict['fitness_mode']
        self.extra_FN_penalty = config_dict['extra_FN_penalty']
        self.G = config_dict['G']
        self.gamma = config_dict['gamma']
        self.mut_rate = config_dict['mut_rate']
        self.motif_n = config_dict['motif_n']
        self.motif_len = config_dict['motif_len']
        self.motif_res = config_dict['motif_res']
        self.pseudocounts = config_dict['pseudocounts']
        self.threshold_res = config_dict['threshold_res']
        self.max_threshold = config_dict['motif_len'] * 2
        self.min_threshold = -self.max_threshold
        
        # Placements map for diads
        if self.motif_n == 2:
            if diad_plcm_map is None:
                self._set_diad_plcm_map()
            else:
                self._check_diad_plcm_map(diad_plcm_map)  # check map validity
                self._diad_plcm_map = diad_plcm_map
        else:
            self._diad_plcm_map = None
        
        # Connector
        self.connector_type = config_dict['connector_type']
        # Gaussian connectors parameters
        self.fix_mu = config_dict['fix_mu']
        self.fix_sigma = config_dict['fix_sigma']
        self.min_mu = config_dict['min_mu']
        self.max_mu = config_dict['max_mu']
        self.min_sigma = 0.01
        self.max_sigma = self.G * 2  # Approximates a uniform over the genome
        self.sigma_res = config_dict['sigma_res']
        self._sigma_vals = np.logspace(
            np.log2(self.min_sigma), np.log2(self.max_sigma), base=2, num=64)  # XXX Obsolete ?
        # Uniform connectors parameters
        self.fix_left  = config_dict['fix_left']
        self.fix_right = config_dict['fix_right']
        self.min_left  = config_dict['min_left']
        self.max_right = config_dict['max_right']
        if self.min_left is None:
            self.min_left = 0
        if self.max_right is None:
            self.max_right = self.G
        
        # Set genome content
        
        self.seq = None
        self.regulator = None
        self.threshold = None
        self.acgt = {'a': None, 'c': None, 'g': None, 't': None} 
        
        self._bases = ['a', 'c', 'g', 't']
        self._nucl_to_int = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        
        # Set seq
        self.synthesize_genome_seq()
        self.set_acgt_content()
        
        # Set target sites
        self.targets_type = config_dict['targets_type']
        self.spacers = config_dict['spacers']
        self.targets = None
        self.targets_binary = None
        self.set_targets()
        
        self.translate_regulator()
    
    def copy_constructor(self, parent):
        
        # Set parameters from parent
        self.fitness_mode = parent.fitness_mode
        self.extra_FN_penalty = parent.extra_FN_penalty
        self.G = parent.G
        self.gamma = parent.gamma
        self.mut_rate = parent.mut_rate
        self.motif_n = parent.motif_n
        self.motif_len = parent.motif_len
        self.motif_res = parent.motif_res
        self.pseudocounts = parent.pseudocounts
        self.threshold_res = parent.threshold_res
        self.max_threshold = parent.max_threshold
        self.min_threshold = parent.min_threshold
        
        # Placements map for diads
        self._diad_plcm_map = parent._diad_plcm_map
        
        # Connector
        self.connector_type = parent.connector_type
        # Gaussian connectors parameters
        self.fix_mu = parent.fix_mu
        self.fix_sigma = parent.fix_sigma
        self.min_mu = parent.min_mu
        self.max_mu = parent.max_mu
        self.min_sigma = parent.min_sigma
        self.max_sigma = parent.max_sigma
        self.sigma_res = parent.sigma_res
        self._sigma_vals = copy.deepcopy(parent._sigma_vals)  # XXX Obsolete ?
        # Uniform connectors parameters
        self.fix_left = parent.fix_left
        self.fix_right = parent.fix_right
        self.min_left = parent.min_left
        self.max_right = parent.max_right
        
        # Set genome content
        
        self.seq = parent.seq
        self.regulator = copy.deepcopy(parent.regulator)
        self.threshold = parent.threshold
        self.acgt = copy.deepcopy(parent.acgt)
        
        self._bases = parent._bases[:]
        self._nucl_to_int = copy.deepcopy(parent._nucl_to_int)
        
        self.targets_type = parent.targets_type
        self.spacers = copy.deepcopy(parent.spacers)
        self.targets = parent.targets[:]
        self.targets_binary = copy.deepcopy(parent.targets_binary)
    
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
            # We allow for sigma_res^4 different sigma values.
            # So we need only sigma_res many bp
            return n_bp_for_mu + self.sigma_res
        
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
                mu_locus, sigma_locus = gene_seq[:-self.sigma_res], gene_seq[-self.sigma_res:]
                
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
                    
                    # Alternative definition of sigma based on spring constant (k)
                    # 0 <= x <= 5
                    x = 5 * self.nucl_seq_to_int(sigma_locus)/(4**self.sigma_res - 1)
                    '''
                    _sigmas = []
                    for i in range(4**self.sigma_res):
                        k = 10**(-5 * i / (4**self.sigma_res - 1))
                        _sigmas.append(0.019235/(k**(1/2)))
                    print("sigmas: ")
                    print(_sigmas)
                    '''
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
        '''        
        Sets the `targets` attribute (as a list of length self.gamma).
        It also calls the `_set_targets_binary` function, which sets the
        `targets_binary` attribute.
        
        The content of the `targets` attribute depend on `targets_type`.
        
        If the `targets_type` attribute is "centroids":
            Each element of the list is
                0 <= e < G
            (actually the initial coding region of the genome is avoided, so it
            would be S <= e < G, where S is the start of the non-coding region).
        
        If the `targets_type` attribute is "placements":
            Each element of the list is
                0 <= e < G^n
            Each of the G^n possible TF-complex placements can be represented
            as an integer.
            
            Example:
            In the diad case (n=2), index i represents a diad placing the start
            of the left element at position l and the start of the right
            element at position r, where l and r are the following quotient and
            remainder:
                l = i//G
                r = i%G
            or
                l, r = divomd(i, G)    
        '''
        
        if self.motif_n == 1 or self.targets_type == 'centroids':
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
            if self.motif_n > 2:
                raise ValueError("To be coded")
            
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
        
        self._set_targets_binary()
    
    def _set_targets_binary(self):
        if self.targets_type == 'centroids':
            length = self.G
        else:
            length = self.G**self.motif_n
        bin_vec = np.zeros(length).astype(int)
        bin_vec[self.targets] = 1
        self.targets_binary = bin_vec
    
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
        ''' Scans the genome using the `pwm_number`-th PWM and returns the scores. '''
        # Generate PSSM
        recog = self.regulator['recognizers'][pwm_number - 1]
        pwm = recog.counts.normalize(pseudocounts=self.pseudocounts)
        pssm = pwm.log_odds()
        # Scan genome
        return pssm.calculate(self.get_seq())
    
    def scan_old(self):
        '''
        Obsolete function.
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
            # TO BE RE-CODED
            # --------------
            raise ValueError('This code block needs to be re-coded.')
            
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
            
    
    # ===========
    # NEW FITNESS
    # ===========
    
    def scan(self):
        '''
        Returns the tuple (scores, hits_indexes).
        
        scores : numpy 1D array
            a vector of scores from the genome scan. For a single motif (n=1),
            the vector will be of length G. For composite motifs (n>1), the
            vector will be of length G^n.
        
        !!! Maybe get_hits_indexes should be called outside scan? <<<<<<<<<<<<<<<<<<<<<
        hits_indexes : list
            the indexes of the scores that are above the threshold.
        '''
        
        if self.motif_n == 1:
            scores =  self.pwm_scan(1)
        elif self.motif_n == 2:
            # The distance between the two recognizers is i - j.
            # But the genome is circular, so distances are ambiguous.
            # We chose non-negative distances. [e.g., the distance between
            # 8 (left) and 2 (right) on a genome of length 10 is 4, instead of -6]
            # So the effective distance will be (j-i) % G, instead of j-i.
            # [e.g., (2-8)%10 = 4, instead of 2-8 = -6]
            _G = self.G
            x1 = np.repeat(self.pwm_scan(1), _G)
            x2 = np.tile(self.pwm_scan(2), _G)
            x3 = np.array([self.regulator['connectors'][0].get_score((j - i) % _G) for i in range(_G) for j in range(_G)])
            scores = x1 + x2 + x3
        else:
            # pwm_arrays = [self.pwm_scan(i+1) for i in range(self.motif_n)]
            raise ValueError('Needs to be coded')
        
        return scores, self.get_hits_indexes(scores)
    
    def get_hits_indexes(self, scores):
        if self.fitness_mode.lower() == 'auprc':
            return list(np.argpartition(scores,-self.gamma)[-self.gamma:])
        else:
            return list(np.argwhere(scores > self.regulator['threshold']).flatten())
    
    def count_fp(self, hits_positions):
        ''' Returns the number of False Positives (FP) or "Type I Errors". '''
        
        if self.targets_type == 'centroids':
            # Type I errors (false positives)
            type_I_errors  = set(hits_positions).difference(set(self.targets))
            # Each FP could be repeated (multiple placements can have same centroid)
            n_fp = sum([hits_positions.count(err) for err in type_I_errors])
            # Each TP could be repeated too. Extra placements on a target are
            # counted as false positives.
            for target in self.targets:
                n_fp += hits_positions.count(target) - 1
            return n_fp
        
        else:
            return len(set(hits_positions).difference(set(self.targets)))
    
    def count_fn(self, hits_positions):
        ''' Returns the number of False Negatives (FN) or "Type II Errors". '''
        return len(set(self.targets).difference(set(hits_positions)))
    
    def _get_fn_penalty(self, hits_indexes, plcm_scores):
        # False Negatives penalty (penalty is between 1 and 2 per FN)
        fn_penalty = 0
        tr = self.regulator['threshold']
        for missed in list(set(self.targets).difference(set(hits_indexes))):
            if self.targets_type == 'centroids':
                # Maximum score among the placements that map onto missed target
                s = max(map(plcm_scores.__getitem__, self._diad_plcm_map['gnom_pos_to_plcm_idx'][missed]))
            else:
                # Score of missed target
                s = plcm_scores[missed]
            # FN penalty + Extra penalty based on the score
            # The extra FN penalty is between 0 and 1, so `fn_penalty` is between 1 and 2
            fn_penalty += 1 + self._calculate_FN_penalty(s, tr)
        return fn_penalty
    
    def _get_errors_penalty(self, hits_indexes, plcm_scores):
        if self.extra_FN_penalty:
            return self.count_fp(hits_indexes) + self._get_fn_penalty(hits_indexes, plcm_scores)
        else:
            return self.count_fp(hits_indexes) + self.count_fn(hits_indexes)
    
    def _get_auprc(self, scores):
        ''' Returns the Area Under the Precision-Recall Curve (AUPRC). '''
        scores[scores == -np.inf] = -10**10
        prec, rec, thrs = precision_recall_curve(self.targets_binary, scores)
        # Round AUPRC to 14 decimal places to avoid AUPRC>1 due to float point errors
        return round(auc(rec, prec), 14)
    
    def get_fitness(self):        
        # Scan genome
        plcm_scores, hits_indexes = self.scan()
        
        # AUPRC-based fitness
        # -------------------
        if self.fitness_mode.lower() == 'auprc':
            if self.targets_type == 'centroids':
                raise ValueError('AUPRC fitness cannot be currently applied ' +
                                 'to centroid-based target definition')
            else:
                return self._get_auprc(plcm_scores)
        
        # Errors penalty-based fitness
        # ----------------------------
        elif self.fitness_mode == 'errors_penalty':
            if self.targets_type == 'centroids':
                return - self._get_errors_penalty(self.idx_to_centroids(hits_indexes), plcm_scores)
            else:
                return - self._get_errors_penalty(hits_indexes, plcm_scores)
    
    def _calculate_FN_penalty(self, score, threshold):
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
    
    
    def mutate(self, mode='ev'):
        ''' Mutates the organism according to the mutation mode. '''
        if mode == 'ev':
            self.mutate_ev()
        elif mode == 'rate':
            self.mutate_with_rate()
        else:
            raise ValueError("mode should be 'ev' or 'rate'.")
    
    def mutate_ev(self):
        ''' Applyies one and only one mutation (like in the ev program). '''
        rnd_pos = random.randint(0, self.G - 1)
        self.mutate_base(rnd_pos)
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
    
    def get_acgt_freqs(self, both_strands=False):
        ''' Returns a list of the genomic frequencies of the four bases (A,C,G,T). '''
        return [self.acgt[base] / self.G for base in self._bases]
    
    # def get_R_sequence_old(self):
        
    #     Rsequence = 0
    #     for i in range(self.motif_len):
    #         obs_bases = [target_seq[i] for target_seq in target_sequences]
    #         for base in self._bases:
    #             freq = obs_bases.count(base) / self.gamma
    #             if freq != 0:
    #                 bg_freq = self.acgt[base] / self.G
    #                 Rsequence += freq * (np.log2(freq) - np.log2(bg_freq))
    #     return Rsequence
    
    
    
    def _get_H_sequences(self, sequences):
        '''
        Returns Shannon's entropy for a set of aligned DNA sequences.
        Hs, defined in equation (2) of "Information Content of Binding Sites
        on Nucleotide Sequences" (Schneider, Stormo, Gold, Ehrenfeucht), is the
        entropy for a specific position in the sequence alignment. This function
        returns the total entropy, i.e., the sum of all the Hs over all positions.
        '''
        n_seq = len(sequences)
        if n_seq == 0:
            return None
        H = 0
        L = len(sequences[0])
        for i in range(L):
            obs_bases = [seq[i] for seq in sequences]
            counts = [obs_bases.count(base) for base in self._bases]
            H += entropy(counts)
        return H
    
    def get_R_sequence_alternative(self, sequences):
        ''' KLD-based ... '''
        if len(sequences) == 0:
            return None
        
        Rseq = 0
        for i in range(len(sequences[0])):
            obs_bases = [target_seq[i] for target_seq in sequences]
            for base in self._bases:
                freq = obs_bases.count(base) / len(obs_bases)
                if freq != 0:
                    bg_freq = self.acgt[base] / self.G
                    Rseq += freq * (np.log2(freq) - np.log2(bg_freq))
        return Rseq
    
    def get_R_sequence_ev(self, sequences):
        '''
        !!! Work in progress ...
        
        Function that takes into account small sample bias.
        As described in "Evolution of biological information" (Schneider, 2000)
        and "Information Content of Binding Sites on Nucleotide Sequences"
        (Schneider, Stormo, Gold, Ehrenfeucht).
        '''
        
        if len(sequences) == 0:
            return None
        
        # Hg = 0
        # for base in self._bases:
        #     p = self.acgt[base] / self.G
        #     if p != 0:
        #         Hg -= p * np.log2(p)
        
        ###Hbefore = Hg * self.motif_len
        
        
        EH = expected_entropy(self.gamma, self.get_acgt_freqs())  # !!! Check
        
        
        # Definition of Rsequence as in Equation (6) in "Information Content of
        # Binding Sites on Nucleotide Sequences" (Schneider et al., 1986)
        # Instead of the sum over L differences, here it's computed as the
        # difference between the total EH (over the L positions) and the total
        # Hs (over the L positions).
        return (EH * len(sequences[0])) - self._get_H_sequences(sequences)
    
    def get_all_tg_Rseq(self):
        if self.targets_type == 'centroids':
            raise ValueError("For this method targets_type must be 'placments'.")
        return [self.get_R_sequence_ev(seq_list) for seq_list in self.idx_to_seq(self.targets)]
    
    def get_R_spacer(self, gaps):
        ''' Returns R_spacer (calculated as H_before - H_after). '''
        if gaps in [None, []]:
            return None
        else:
            return np.log2(self.G) - entropy(list(collections.Counter(gaps).values()))
    
    def get_R_connector(self, conn_idx=0):
        return np.log2(self.G) - self.regulator['connectors'][conn_idx].get_conn_entropy()
    
    def get_R_frequency(self):
        ''' Returns R_frequency according to our generalized framework. '''
        return -np.log2(self.gamma/(self.G**self.motif_n))
    
    def max_possible_IC(self):
        ''' Maximum information possible for the (composite) motif of the regulator,
        i.e., 2L*number_of_PSSMs + log2(G)*number_of_spacers. '''
        return (2 * self.motif_len * self.motif_n) + (np.log2(self.G) * (self.motif_n - 1))
    
    def _set_diad_plcm_map(self):
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
        G = self.G
        plcm_idx_to_gnom_pos = []
        gnom_pos_to_plcm_idx = [[] for i in range(G)]
        for idx in range(G**2):
            x, y = divmod(idx, G)
            if y < x:
                y += G
            # Genome position (centroid)
            pos = int((x + y + self.motif_len)/2) % G
            plcm_idx_to_gnom_pos.append(pos)
            gnom_pos_to_plcm_idx[pos].append(idx)
        self._diad_plcm_map = {'plcm_idx_to_gnom_pos': plcm_idx_to_gnom_pos,
                               'gnom_pos_to_plcm_idx': gnom_pos_to_plcm_idx}
    
    def _check_diad_plcm_map(self, diad_plcm_map):
        if not type(diad_plcm_map) is dict:
            raise ValueError("diad_plcm_map must be a dictionary.")
        if set(diad_plcm_map.keys()) != {'plcm_idx_to_gnom_pos', 'gnom_pos_to_plcm_idx'}:
            raise ValueError("diad_plcm_map keys must be 'plcm_idx_to_gnom_pos' " +
                             "and 'gnom_pos_to_plcm_idx'.")
        if len(diad_plcm_map['plcm_idx_to_gnom_pos']) != self.G**self.motif_n:
            raise ValueError("plcm_idx_to_gnom_pos should be a list of " +
                             str(self.G**self.motif_n) + " elements.")
        if len(diad_plcm_map['gnom_pos_to_plcm_idx']) != self.G:
            raise ValueError("gnom_pos_to_plcm_idx should be a list of " +
                             str(self.G) + " elements.")
    
    def export(self, outfilepath=None):
        '''
        Exports the organism as a JSON file. If the path of the output file
        `outfilepath` is not specified, a python dictionary is returned, instead.
        '''
        # XXX Update this dictionary
        out_dict = {'seq': self.seq,
                    'G': self.G,
                    'gamma': self.gamma,
                    'motif_n': self.motif_n,
                    'motif_len': self.motif_len,
                    'targets': self.targets,
                    'motif_res': self.motif_res,
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
    
    def idx_to_seq(self, indexes):
        '''
        Returns a list of sequences (when motif_n = 1) or pairs of sequences
        (when motif_n = 2). Each element of the returned list corresponds to
        one index in the `indexes` input list.
        
        indexes : list
            - If the regulator is monomeric (motif_n = 1), each index is the
              genomic start positions of the sequences to be returned (each
              sequence has a length of `motif_len` bp).
            - If the regulator is a dimer (motif_n = 2), each index is
              converted into a pair of start positions, for the left and the
              right elements of the diad (both sequences will have a length
              of `motif_len` bp).
        '''
        if self.motif_n == 1:
            return [[self.get_seq()[idx:idx+self.motif_len] for idx in indexes]]
        elif self.motif_n == 2:
            seq1 = []
            seq2 = []
            for idx in indexes:
                l, r = divmod(idx, self.G)
                seq1.append(self.get_seq()[l:l+self.motif_len])
                seq2.append(self.get_seq()[r:r+self.motif_len])
            return [seq1, seq2]
        else:
            raise ValueError("To be coded ...")
    
    def _get_hits_spacers(self, hits_indexes):
        if self.motif_n != 2:
            return None
        hits_spacers = []
        for idx in hits_indexes:
            l, r = divmod(idx, self.G)
            hits_spacers.append((r - l) % self.G - self.motif_len)
        return hits_spacers
        
    
    def idx_to_centroids(self, indexes):
        if self.motif_n == 1:
            return indexes
        elif self.motif_n == 2:
            return [self._diad_plcm_map['plcm_idx_to_gnom_pos'][idx] for idx in indexes]
        else:
            raise ValueError("To be coded ...")
    
    def _get_gene_string(self, name, length):
        ''' Called by `print_genome_map`. Returns a string of the given length
        that represents a gene as a name followed by an arrow. '''
        name += '-' * (length - len(name))
        return name[:length-1] + '>'
    
    def _add_diad_plcm_line(self, outlist, elements_pos, elements_idx):
        '''
        Called by `print_genome_map`. Adds one line of characters to the output.
        Writes characters in the right positions of `outlist` (list of characters)
        if it finds space characters. Otherwise, it doesn't overwrite the characters
        found at those positions: the info about the elements that were supposed
        to be written into `outlist` will rather be saved into `leftovers` and
        `leftovers_idx` (will be displayed in the next line of the output).
        
        elements_pos : list
            Each element of the list is a tuple of two integers
                - start position of the left element of the diad
                - start position of the right element of the diad
        
        elements_idx : list
            Each element of the list is an index (int) for the diad placement
            to be displayed.
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
        if self.motif_n == 1 or self.targets_type == 'centroids':
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
        plcm_scores, hits_indexes = self.scan()
        _G = self.G
        
        if self.motif_n == 1:
            hits_positions = hits_indexes  # !!! Check hits_positions definition when motif_n=1
        
        elif self.motif_n == 2:
            hits_positions = [self._diad_plcm_map['plcm_idx_to_gnom_pos'][idx] for idx in hits_indexes]
            elements_pos = [divmod(idx, _G) for idx in hits_indexes]
        
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
    
    def study_info(self, outfilepath=None, gen=None):
        
        if self.motif_n > 2:
            raise ValueError("To be coded.")
        
        plcm_scores, hits_indexes = self.scan()
        
        # TARGETS
        # -------
        if self.targets_type == 'placements':
            tg_sequences = self.idx_to_seq(self.targets)        
            tg_Rseq_true = [self.get_R_sequence_ev(s) for s in tg_sequences]
            tg_Rseq_alt = [self.get_R_sequence_alternative(s) for s in tg_sequences]
            if self.motif_n == 2:
                tg_Rspacer = self.get_R_spacer(self.spacers)
        else:
            # Targets are not defined when using 'centroids' mode
            tg_Rseq_true = None, None
            tg_Rseq_alt = None, None
            tg_Rspacer = None
        
        # Targets baseline information
        tg_EH = expected_entropy(self.gamma, self.get_acgt_freqs())
        tg_baseline_info = (2 - tg_EH) * self.motif_len
        # This is assuming base probabilities 25% each
        tg_EH_unif = expected_entropy(self.gamma)
        tg_baseline_info_unif = (2 - tg_EH_unif) * self.motif_len
        # 'Targets' stats (only makes sense for 'placements' mode)
        if self.targets_type == 'placements':
            tg_cols = []
            # print('>>>>\tTargets:')
            for i in range(self.motif_n):
                # print('>>>>\t\tRseq({})\t{:.4f}\t{:.4f}'.format(i+1, tg_Rseq_true[i], tg_Rseq_alt[i]-tg_baseline_info))
                tg_cols.append([tg_Rseq_true[i],
                                tg_Rseq_alt[i]-tg_baseline_info,
                                tg_Rseq_alt[i]-tg_baseline_info_unif,
                                tg_Rseq_alt[i]])
        else:
            tg_cols = [[None]*4]*self.motif_n
        
        # HITS
        # ----
        hit_sequences = self.idx_to_seq(hits_indexes)        
        hit_Rseq_true = [self.get_R_sequence_ev(s) for s in hit_sequences]
        hit_Rseq_alt = [self.get_R_sequence_alternative(s) for s in hit_sequences]
        if self.motif_n == 2:
            hits_spacers = self._get_hits_spacers(hits_indexes)
            hit_Rspacer = self.get_R_spacer(hits_spacers)
        
        if len(hits_indexes) > 0:
            # Hits baseline information
            hit_EH = expected_entropy(len(hits_indexes), self.get_acgt_freqs())
            hit_baseline_info = (2 - hit_EH) * self.motif_len
            # This is assuming base probabilities 25% each
            hit_EH_unif = expected_entropy(len(hits_indexes))
            hit_baseline_info_unif = (2 - hit_EH_unif) * self.motif_len
            # if self.motif_n == 2:
            #     print('>>>>\tHits:')
            #     print('>>>>\t\tRseq(1)\t{:.4f}\t{:.4f}'.format(hit_Rseq_true[0], hit_Rseq_alt[0]-hit_baseline_info))
            #     print('>>>>\t\tRseq(2)\t{:.4f}\t{:.4f}'.format(hit_Rseq_true[1], hit_Rseq_alt[1]-hit_baseline_info))
            
            hit_cols = []
            for i in range(self.motif_n):
                hit_cols.append([hit_Rseq_true[i], hit_Rseq_alt[i]-hit_baseline_info, hit_Rseq_alt[i]-hit_baseline_info_unif, hit_Rseq_alt[i]])
        else:
            hit_cols = [[None]*4] * self.motif_n
        
        # Save IC report
        # --------------
        if self.motif_n == 1:
            df = pd.DataFrame(
                {'Rseq_targets': tg_cols[0],
                 'Rseq_hits': hit_cols[0],
                 'Rfrequency': [self.get_R_frequency()] * 4})
            df.index = ['true_Rseq', 'alt_corrected', 'alt_corrected_unif', 'alt_uncorrected']
            df.to_csv(outfilepath + '_ic_report.csv')
            print('')
            print(df.transpose())
        
        elif self.motif_n == 2:
            if self.connector_type == 'gaussian':
                sigma = self.regulator['connectors'][0].sigma
            else:
                sigma = None
            df = pd.DataFrame(
                {'Rseq1_targets': tg_cols[0],
                 'Rseq1_hits': hit_cols[0],
                 'Rseq2_targets': tg_cols[1],
                 'Rseq2_hits': hit_cols[1],
                 'Rspacer_targets': [tg_Rspacer] * 4,
                 'Rspacer_hits': [hit_Rspacer] * 4,
                 'Rconnector': [self.get_R_connector()] * 4,
                 'connector_sigma': [sigma] * 4,
                 'generation': [gen] * 4})
            df['Rtot'] = df['Rseq1_targets'] + df['Rseq2_targets'] + df['Rspacer_targets']
            df['Reffective'] = df['Rseq1_targets'] + df['Rseq2_targets'] + df['Rconnector']
            df['Rfrequency'] = [self.get_R_frequency()] * 4
            df.index = ['true_Rseq', 'alt_corrected', 'alt_corrected_unif', 'alt_uncorrected']
            df.to_csv(outfilepath + '_ic_report.csv')
            print(df.drop(['generation'], axis=1).transpose())
        
        if self.motif_n == 2:
            # Save gaps report
            with open(outfilepath + '_gaps_report.json', 'w') as f:
                json.dump([int(gap) for gap in hits_spacers], f)













