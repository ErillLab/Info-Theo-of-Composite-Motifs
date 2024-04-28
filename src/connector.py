
'''
Two classes:
    - ConnectorGauss (Connector modeled with a Gaussian distribution)
    - ConnectorUnif (Connector modeled with a Uniform distribution)

'''


import numpy as np
import math


class ConnectorGauss():
    ''' Connector modeled with a truncated Gaussian distribution '''
    
    def __init__(self, mu, sigma, G, motif_len):
        
        self.mu = mu
        self.sigma = sigma
        self.trunc_z = 3  # z-score where the Gaussian is truncated
        self._G = G
        self._motif_len = motif_len
        self.scores = None
        self.set_scores()
    
    def _spacer_score(self, distance):
        ''' Returns the connector score for the input distance between the start
        position of the two connected elements. The score is a log-likelihood
        ratio, where the null model assigns a probability of 1/G to each gap
        size, while the alternative model is a truncated Gaussian. '''
        
        # If the distance between the start positions is less than the motif
        # length it means that the two binding domains are overlapping (negative
        # gap size), which is not allowed, due to steric constraints (returns -inf)
        if abs(distance) < self._motif_len:
            return -np.inf
        
        # If the recog 'to the right' has a coordinate that is smaller than the
        # coordinate of the recog 'to the left' it means that R-L is negative due to
        # an artifact coming from genome circularity. Real distance is distance % G.
        # The actual gap size (spacer length) is real distance - motif_len.
        gap = (distance % self._G) - self._motif_len
        
        # Probability according to a Gaussian truncated at 3 standard deviations from mu
        prob = self._norm_pf(gap)
        
        # Return Log-likelihood ratio (LLR)
        if prob == 0:
            return -np.inf
        else:
            # LLR == log( prob / (1/G) ) == log( prob * G )
            return np.log2(prob * self._G)
    
    def set_scores(self):
        tot_auc = (self._norm_cdf(self.mu + (self.sigma * self.trunc_z)) -
                   self._norm_cdf(self.mu - (self.sigma * self.trunc_z)) )
        self.scores = [self._spacer_score(i)/tot_auc for i in range(self._G)]
    
    def get_score(self, distance):
        return self.scores[distance]
    
    
    # def set_scores_1(self):
        
    #     if self.sigma == 0:
    #         gap_scores = [-np.inf] * self._G
    #         if isinstance(self.mu, int):
    #             # Max_LLR == log( Max_prob / (1/G) ) == log( 1 / (1/G) ) == log(G)
    #             gap_scores[self.mu] = np.log2(self._G)
        
    #     else:
    #         # Define bins
    #         right_truncation = self.mu + (self.sigma * self.trunc_z)
    #         left_truncation  = self.mu - (self.sigma * self.trunc_z)
    #         min_gap = int(left_truncation + 0.5)
    #         max_gap = int(right_truncation + 0.5)
            
    #         if min_gap == max_gap:
    #             gap_scores = [-np.inf] * min_gap + [np.log2(self._G)] + [-np.inf] * (self._G - (max_gap + 1))
            
    #         else:
    #             interm_gaps = list(range(min_gap+1, max_gap))
    #             starts = ([left_truncation] +  # start of min_gap bin
    #                       [g-0.5 for g in interm_gaps] +
    #                       [max_gap - 0.5])  # start of max_gap bin
    #             # Bin probabilities
    #             probs = []
    #             for i in range(len(starts)-1):
    #                 probs.append(self._norm_cdf(starts[i+1]) -
    #                              self._norm_cdf(starts[i]) )
    #             probs.append(self._norm_cdf(right_truncation) -  # end of max_gap bin
    #                          self._norm_cdf(starts[-1]) )
    #             # Normalize by the total probability within the truncated region
    #             tot_auc = self._norm_cdf(right_truncation) - self._norm_cdf(left_truncation)
    #             if tot_auc == 0:
    #                 raise ValueError("AUC is 0")
    #             for i in range(len(probs) - 1):
    #                 p = probs[i]
    #                 l, r = starts[i], starts[i+1]
    #                 lcum, rcum = self._norm_cdf(starts[i]), self._norm_cdf(starts[i+1])
    #                 if p <= 0:
    #                     print('min_gap:', min_gap, 'max_gap:',max_gap)
    #                     print(starts)
    #                     print('interm_gaps:', interm_gaps)
    #                     print('L:', left_truncation, 'R:', right_truncation)
    #                     raise ValueError("Prob={}, Left={}, Right={}, LCum={}, RCum={}".format(
    #                         p, l, r, lcum, rcum))
    #             probs = np.array(probs) / tot_auc
                
    #             # LLR == log( prob / (1/G) ) == log( prob * G )
    #             LLRs = np.log2(probs * self._G)
                
    #             gap_scores = [-np.inf] * min_gap + list(LLRs) + [-np.inf] * (self._G - (max_gap + 1))
        
    #     self.scores = gap_scores
        
    # def get_score_1(self, distance):
        
    #     if abs(distance) < self._motif_len:
    #         # If the distance between the start positions is less than the motif
    #         # length it means that the two binding domains are overlapping (negative
    #         # gap size), which is not allowed, due to steric constraints (returns -inf)
    #         return -np.inf
        
    #     else:
    #         # If the recog 'to the right' has a coordinate that is smaller than the
    #         # coordinate of the recog 'to the left' it means that R-L is negative due to
    #         # an artifact coming from genome circularity. Real distance is distance % G.
    #         # The actual gap size (spacer length) is real distance - motif_len.
    #         return self.scores[(distance % self._G) - self._motif_len]
    
    
    def _norm_cdf(self, x):
        '''
        Cumulative density function for the normal distribution specified by the
        mu and sigma attributes.
        '''
        if self.sigma == 0:
            if x < self.mu:
                return 0
            elif x > self.mu:
                return 1
            elif x == self.mu:
                return 0.5
        else:
            z = (x - self.mu) / abs(self.sigma)  # z-score
            return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0
    
    def _norm_pf(self, x):
        '''
        Probablity function based on a normal distribution.
        Considering that the observed x is an integer, the probability of x is
        defined as the probability of observing a value within x - 0.5 and x + 0.5,
        given the normal distribution specified by the mu and sigma attributes.
        '''
        if self.sigma != 0:
            # Left and Right truncation of the Gaussian
            l = max(x-0.5, self.mu - (self.sigma * self.trunc_z))
            r = min(x+0.5, self.mu + (self.sigma * self.trunc_z))
            if l < r:
                return self._norm_cdf(r) - self._norm_cdf(l)
            else:
                return 0
        
        else:
            # Dirac Delta
            if x == self.mu:
                return 1
            else:
                return 0
    


class ConnectorUnif():
    ''' Connector modeled with a Uniform distribution '''
    
    def __init__(self, min_gap, max_gap, G, motif_len):
        
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.n_valid_gaps = max_gap - min_gap + 1
        self._G = G
        self._motif_len = motif_len
        self.scores = None
        self.set_scores()
    
    def _spacer_score(self, distance):
        ''' Returns the connector score for the input distance between the start
        position of the two connected elements. The score is a log-likelihood
        ratio, where the null model assigns a probability of 1/G to each gap
        size, while the alternative model is a Uniform. '''
        
        # If the distance between the start positions is less than the motif
        # length it means that the two binding domains are overlapping (negative
        # gap size), which is not allowed, due to steric constraints (returns -inf)
        if abs(distance) < self._motif_len:
            return -np.inf
        
        # If the recog 'to the right' has a coordinate that is smaller than the
        # coordinate of the recog 'to the left' it means that R-L is negative due to
        # an artifact coming from genome circularity. Real distance is distance % G.
        # The actual gap size (spacer length) is distance - motif_len.
        gap = (distance % self._G) - self._motif_len
        
        # Return Log-likelihood ratio (LLR)
        if self.min_gap <= gap <= self.max_gap:
            # LLR == log( (1/nvg) / (1/G) ) == log( G/nvg )
            return np.log2(self._G/self.n_valid_gaps)
        else:
            return -np.inf
    
    def set_scores(self):
        self.scores = [self._spacer_score(i) for i in range(self._G)]
            
    def get_score(self, distance):
        return self.scores[distance]












