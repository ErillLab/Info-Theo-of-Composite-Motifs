
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
        self._G = G
        self._motif_len = motif_len
        self.scores = None
        self.set_scores()
    
    def spacer_score(self, distance):
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
        # The actual gap size (spacer length) is distance - motif_len.
        gap = (distance % self._G) - self._motif_len
        
        # Probability according to a Gaussian truncated at 3 standard deviations from mu
        if abs((gap - self.mu)/self.sigma) > 3:
            prob = 0
        else:
            prob = self._norm_pf(gap)
        
        # Return Log-likelihood ratio (LLR)
        if prob == 0:
            return -np.inf
        else:
            # LLR == log( prob / (1/G) ) == log( prob * G )
            return np.log2(prob * self._G)
    
    def set_scores(self):
        self.scores = [self.spacer_score(i) for i in range(self._G)]
            
    def score(self, distance):
        return self.scores[distance]
    
    def _norm_cdf(self, x):
        ''' Cumulative distribution function for the normal distribution. '''
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
        given a normal distribution specified by the given mu and sigma.
        '''
        if self.sigma != 0:
            # Normalize: divide by the AUC (the sum over all the possible outcomes)
            return (self._norm_cdf(x+0.5) - self._norm_cdf(x-0.5))/(self._norm_cdf(self._G+0.5) - self._norm_cdf(-0.5))
        else:
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
    
    def spacer_score(self, distance):
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
        self.scores = [self.spacer_score(i) for i in range(self._G)]
            
    def score(self, distance):
        return self.scores[distance]



































