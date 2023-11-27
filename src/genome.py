# -*- coding: utf-8 -*-


import numpy as np
import random
import copy
import math
import itertools
from Bio import motifs
# from Bio import SeqIO
# from Bio.Seq import Seq

from connector import Connector





class Genome():
    
    def __init__(self, config_dict):
        
        # Set parameters from config file
        self.G = config_dict['G']
        self.gamma = config_dict['gamma']
        self.mut_rate = config_dict['mut_rate']
        self.motif_len = config_dict['motif_len']
        self.motif_res = config_dict['motif_res']
        self.motif_n = config_dict['motif_n']
        self.threshold_res = config_dict['threshold_res']
        self.max_threshold = config_dict['motif_len'] * 2
        self.pseudocounts = 1  # XXX Temporarily hardcoded
        self.min_mu = config_dict['min_mu']
        self.max_mu = config_dict['max_mu']
        self.min_sigma = 0.01
        self.max_sigma = self.G * 2  # Approximates a uniform over the genome
        
        self._sigma_vals = np.logspace(np.log2(self.min_sigma), np.log2(self.max_sigma), base=2, num=64)
        
        self.seq = None
        self.regulator = None
        self.threshold = None
        self.acgt = None
        
        # XXX Move to `biochem` file and import biochem?
        self._bases = ['a', 'c', 'g', 't']
        self._mut_dict = {'a':['c', 'g', 't'], 'c':['a', 'g', 't'], 'g':['a', 'c', 't'], 't':['a', 'c', 'g']}
        self._nucl_to_int = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        
        # Set seq
        self.synthesize_genome_seq()
        self.set_acgt_content()
        
        
        '''
        # Define TF gene
        self.tf_gene_len = self.motif_res * self.motif_len
        self.tf_gene_loc = {'start': 0, 
                            'end': self.tf_gene_len}
        
        # Define threshold gene(s)
        self.threshold_gene_len = None
        self.set_threshold_gene_len()
        
        self.threshold_gene_loc = {'start': self.tf_gene_len,
                                   'end': self.tf_gene_len + self.threshold_gene_len}
        '''
        
        # Set target sites
        self.targets = None
        self.set_targets()
        
        # !!!
        self.translate_regulator()
    
    # !!!
    '''
    def set_genes(self):
        for i in range(self.motif_n):
            pwm_name = 'pwm_' + str(i+1)
            self.genes[pwm_name] = {'start': self.first_non_cd_pos,
                                      'stop': self.first_non_cd_pos + self.get_pwm_gene_len()}
            # Update first non-coding position
            self.first_non_cd_pos = self.genes[pwm_name]['stop']
            
            conn_name = 'connector_' + str(i+1)
            self.genes[conn_name] = {'start': self.genes[pwm_name]['stop'],
                                      'stop': self.genes[pwm_name]['stop'] + self.get_conn_gene_len()}
            
        
        # !!!
        # Gene for the threshold
        self.genes['threshold'] = {'start': self.first_non_cd_pos,
                                    'stop': self.first_non_cd_pos + self.get_threshold_gene_len()}
    '''
    
    def synthesize_genome_seq(self):
        self.seq = "".join(random.choices(self._bases, k=self.G))
        # Circular genome
        self.seq = self.seq + self.seq[:self.motif_len-1]
    
    def get_pwm_gene_len(self):
        return self.motif_res * self.motif_len
    
    def get_conn_gene_len(self):
        # Encoding mu
        n_mu_vals = self.max_mu - self.min_mu + 1
        # Required number of bp to encode the mu value
        n_bp_for_mu = int(np.ceil(math.log(n_mu_vals, 4)))
        # Encoding sigma
        # We allow for 64 sigma values, spanning (in log space) from 0.01 to G
        # Therefore, we only need 3 bp (because 4^3=64)
        return n_bp_for_mu + 3
    
    def get_threshold_gene_len(self):
        n_positive_thrs_vals = self.max_threshold * self.threshold_res
        n_negative_thrs_vals = n_positive_thrs_vals * 1
        n_thrs_vals = n_negative_thrs_vals + n_positive_thrs_vals
        # Required number of bp to encode the threshld value
        return int(np.ceil(math.log(n_thrs_vals, 4)))
    
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
                raise ValueError('')  # XXX Define error message
        # Gene coordinates
        offset = (conn_number * self.get_pwm_gene_len()) + ((conn_number - 1) * self.get_conn_gene_len())
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
    
    def translate_conn_gene(self, conn_number):
        gene_seq = self.get_conn_gene_seq(conn_number)
        mu_locus, sigma_locus = gene_seq[:-3], gene_seq[-3:]
        # Translate mu
        if len(mu_locus)==0:
            mu = self.min_mu
        else:
            mu = self.nucl_seq_to_int(mu_locus)
            if mu > self.max_mu:
                mu = self.max_mu
        # Transalte sigma
        sigma_idx = self.nucl_seq_to_int(sigma_locus)
        sigma = self._sigma_vals[sigma_idx]
        return Connector(mu, sigma, self.G, self.motif_len)
    
    def translate_threshold_gene(self):
        # !!! double-check code
        thrsh = (self.nucl_seq_to_int(self.get_thrsh_gene_seq()) / self.threshold_res) - self.max_threshold
        if thrsh > self.max_threshold:
            return self.max_threshold
        else:
            return thrsh
    
    def translate_regulator(self):
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
        self.regulator = {'recognizers': recog_list, 'connectors': conn_list, 'threshold': threshold}
    
    def set_targets(self):
        # Avoid setting targets within the coding sequences (the first part of the genome)
        end_CDS = self.get_non_coding_start_pos()
        # Avoid overlapping sites
        tmp = random.sample(range(end_CDS, self.G-self.motif_len, self.motif_len), k=self.gamma)
        tmp.sort()
        for i in range(len(tmp)-1):
            gap = tmp[i+1] - (tmp[i] + self.motif_len)
            tmp[i] += random.randint(0, min(gap, self.motif_len))
        self.targets = tmp
    
    def nucl_seq_to_int(self, nucl_seq):
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
        return pssm.calculate(self.seq)
    
    def scan(self):
        pwm_arrays = [self.pwm_scan(i+1) for i in range(self.motif_n)]
        
        # Shortcut for the single motif case
        if self.motif_n == 1:
            return list(np.argwhere(pwm_arrays[0] > self.regulator['threshold']).flatten())
        
        # Shortcut for the diad case
        elif self.motif_n == 2:
            plcm_pwm_scores = list(itertools.product(*pwm_arrays))
            plcm_scores = []
            plcm_pos = []
            print('\tfor ...')
            # XXX VECTORIZE! ...
            for i in range(len(plcm_pwm_scores)):
                q, r = divmod(i, self.G)
                # The distance between the two recognizers is r - q
                plcm_scores.append(sum(plcm_pwm_scores[i]) + self.regulator['connectors'][0].score(r - q))
                plcm_pos.append(int((r + q + self.motif_len)/2))  # motif center
            print('\tdone')
            hits_indexes = np.argwhere(np.array(plcm_scores) > self.regulator['threshold']).flatten()
            return [plcm_pos[idx] for idx in hits_indexes]
        
        # Code for the general case (works for any value of `motif_n`)
        else:
            plcm_pwm_scores = list(itertools.product(*pwm_arrays))
            plcm_pwm_pos = list(itertools.product(range(self.G), repeat=self.motif_n))
            plcm_spcr_scores = []
            
            for plcm in plcm_pwm_pos:
                distances = [t - s for s, t in zip(plcm, plcm[1:])]
                connscores = [self.regulator['connectors'][i].score(distances[i]) for i in range(len(distances))]
                plcm_spcr_scores.append(connscores)
            
            plcm_scores = []
            plcm_pos = []
            for i in range(len(plcm_pwm_scores)):
                # XXX Vectorize ?
                plcm_scores.append(sum(plcm_pwm_scores[i]) + sum(plcm_spcr_scores[i]))
                plcm_pos.append(int((plcm_pwm_pos[i][0] + plcm_pwm_pos[i][-1] + self.motif_len)/2))  # motif center
            hits_indexes = np.argwhere(np.array(plcm_scores) > self.regulator['threshold']).flatten()
            return [plcm_pos[idx] for idx in hits_indexes]
        
        
        
        '''
        idx = 23
        G = 4
        
        n = 3
        r = idx
        
        while n > 1:
            q, r = divmod(r, G**n)
        
        for plcm in all_plcm_pwm_pos:
            score = 0
            for i in range(len(plcm)-1):
                score += plcm[i] + connscore[d(i)]
            score += plcm[-1]
        '''
        
    
        # # Return hits positions
        # return np.argwhere(scores > self.threshold).flatten()
    
    def get_fitness(self):
        hits_positions = self.scan()
        targets_positions = self.targets
        # Fitness is (-1) * number-of-errors
        return -(self.count_false_positives(hits_positions, targets_positions) +
                 self.count_false_negatives(hits_positions, targets_positions))
        '''
        # False positives
        n_fp = self.count_false_positives(hits_positions, targets_positions)
        # False negatives
        n_fn = self.count_false_negatives(hits_positions, targets_positions)
        # Fitness is (-1) * number-of-errors
        return - (n_fp + n_fn)
        '''
    
    def count_false_positives(self, hits_positions, targets_positions):
        # Count type_I_errors
        type_I_errors  = set(hits_positions).difference(set(targets_positions))
        if self.motif_n == 1:
            return len(type_I_errors)
        else:
            n_fp = sum([hits_positions.count(err) for err in type_I_errors])
            # By def, there are only gamma correct placements. Enforce by counting
            # redundant placements as false positives (fp)
            for target in targets_positions:
                if hits_positions.count(target) > 1:
                    n_fp += hits_positions.count(target) - 1
            return n_fp
    
    def count_false_negatives(self, hits_positions, targets_positions):
        # Count type_II_errors
        return len(set(targets_positions).difference(set(hits_positions)))
    
    def mutate_base(self, base_position):
        curr_base = self.seq[base_position]
        new_base = random.choice(self._bases)
        self.seq = self.seq[:base_position] + new_base + self.seq[base_position+1:]
        # Update ACGT content
        self.acgt[curr_base] -= 1
        self.acgt[new_base] += 1
    
    def mutate_with_rate(self):
        # !!!
        #n_mut_bases = np.random.binomial(self.G, self.mut_rate)
        n_mut_bases = int(self.G * self.mut_rate)
        if n_mut_bases > 0:
            mut_bases_positions = random.sample(range(self.G), k=n_mut_bases)
            for pos in mut_bases_positions:
                self.mutate_base(pos)
            # !!!
            if min(mut_bases_positions) < self.get_non_coding_start_pos():
                self.translate_regulator()
        
        # # Re-set TF model and threshold
        # self.translate_tf()
        # self.translate_threshold_gene()
    '''
    INSERTIONS and DELETIONS:
        make them compensatory (#insertions == #deletions), so that the length
        of the genome G doesn't change. #insertions == #deletions will be drawn
        from a Poisson, with lambda depending on the specific mutation rate.
    '''
    
    
    def mutate_ev(self):
        rnd_pos = random.randint(0, self.G - 1)
        self.mutate_base(rnd_pos)
        # !!!
        if rnd_pos < self.get_non_coding_start_pos():
            self.translate_regulator()
    
    def replicate(self):
        return copy.deepcopy(self)
    
    def set_acgt_content(self, both_strands=False):
        a = self.seq.count('a')
        c = self.seq.count('c')
        g = self.seq.count('g')
        t = self.G - (a+c+g)
        self.acgt = {'a': a, 'c': c, 'g': g, 't': t}   
    
    def get_R_sequence(self):
        target_sequences = [self.seq[pos:pos+self.motif_len] for pos in self.targets]
        
        H = 0
        for i in range(self.motif_len):
            obs_bases = [target_seq[i] for target_seq in target_sequences]
            counts = {}
            for base in self._bases:
                counts[base] = obs_bases.count(base)
            frequencies = np.array(list(counts.values()))/sum(counts.values())
            for f in frequencies:
                if f != 0:
                    H -= f * np.log2(f)
        return (2 * self.motif_len) - H    
    
    def get_R_sequence_ev(self):
        '''
        Background frequencies as in "Information Content of Binding Sites on
        Nucleotide Sequences" Schneider, Stormo, Gold, Ehrenfeucht.
        This is the method used in "Evolution of biological information".
        '''
        target_sequences = [self.seq[pos:pos+self.motif_len] for pos in self.targets]
        
        H = 0
        for i in range(self.motif_len):
            obs_bases = [target_seq[i] for target_seq in target_sequences]
            
            for base in self._bases:
                freq = obs_bases.count(base) / self.gamma
                if freq != 0:
                    bg_freq = self.acgt[base] / self.G
                    H += freq * (np.log2(freq) - np.log2(bg_freq))
        return H
    
    def get_R_sequence_ev_new(self):
        '''
        As described in "Evolution of biological information".
        '''
        Hg = 0
        for base in self._bases:
            p = self.acgt[base] / self.G
            if p != 0:
                Hg -= p * np.log2(p)
        # !!!
        # Skip for now: correction is negligible for large genomes
        # Hg += e(self.G)
        Hbefore = Hg * self.motif_len
        
        Hafter = 0
        
        
        
        
    
    def get_R_frequency(self):
        return -np.log2(self.gamma/(self.G**self.motif_n))
    
    # XXX
    # def get_R_placement(self):
    #     return xxx
    
    def max_possible_IC(self):
        # (max recog IC * #recogs) + (max conn IC * #connectors)
        return (2 * self.motif_len * self.motif_n) + (np.log2(self.G) * (self.motif_n - 1))











