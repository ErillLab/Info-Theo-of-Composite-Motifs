# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:06:54 2024

@author: eliam
"""

import numpy as np
import json
import os
import math
import matplotlib.pyplot as plt



def read_json_file(filename):
    ''' Returns the content of a specified JSON file as a python object. '''
    with open(filename) as json_content:
        return json.load(json_content)




for f in os.listdir(results_dir):
    if not os.path.isdir(results_dir + f):
        continue
    for filename in os.listdir(results_dir + f):
        if filename.startswith('gen_') and filename.endswith('_org.json'):
            d = read_json_file(results_dir + f + '/' + filename)
            if d['mu'] == 5:
                print(f, 'sigma =', d['sigma'])







def entropy(counts):
    n = sum(counts)
    H = 0
    for c in counts:
        if c != 0:
            H -= (c/n) * np.log2(c/n)
    return H




# ====================================
# ~ Normally distributed Spacer Values
# ====================================

vals = [3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7]
#plt.hist(vals, bins=5)

vals = [3, 4, 4, 5, 5, 5, 6, 6, 7]


mu, sigma = np.mean(vals), np.std(vals)

def _norm_cdf(x, mu, sigma):
    ''' Cumulative distribution function for the normal distribution. '''
    z = (x - mu) / abs(sigma)  # z-score
    return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0

def _norm_pf(x, mu, sigma):
    '''
    Probablity function based on a normal distribution.
    Considering that the observed x is an integer, the probability of x is
    defined as the probability of observing a value within x - 0.5 and x + 0.5,
    given a normal distribution specified by the given mu and sigma.
    '''
    # Normalize: divide by the AUC (the sum over all the possible outcomes)
    return _norm_cdf(x+0.5, mu, sigma) - _norm_cdf(x-0.5, mu, sigma)


unique, counts = np.unique(vals, return_counts = True)
freqs = counts / sum(counts)

gauss_p = []
for v in unique:
    gauss_p.append(_norm_pf(v, mu, sigma))

for i in range(len(freqs)):
    print(freqs[i], '\t', gauss_p[i])

# ====================================





np.std([3,4,4,4,5,5,5,6])


# mu = 5, sigma = 0
vec = [5,5,5,5,5,5,5,5]

# mu = 4.5, sigma = 0.87
vec = [3,4,4,4,5,5,5,6]



np.std(vec)

counts = [vec.count(item) for item in list(set(vec))]
for i in range(len(vec) - len(counts)):
    counts += [0]
entropy(counts)









