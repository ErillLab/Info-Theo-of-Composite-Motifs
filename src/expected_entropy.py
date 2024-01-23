# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multinomial
import math


def entropy(counts):
    n = sum(counts)
    H = 0
    for c in counts:
        if c != 0:
            H -= (c/n) * np.log2(c/n)
    return H


def expected_entropy(n, base_probabilities=[0.25, 0.25, 0.25, 0.25]):
    '''
    Computes the Expected value of the entropy for a random collection of `n`
    bases. When `n` is the number of aligned sites, this function provides the
    expected value of bits per position (assuming the `n` sites are random).
    If `base_probabilities` is not specified, equal probabilities for the four
    bases are assumed.
    
    The concept, as well as the algorithm implemented here, are described in:
        
    Schneider, T. D., Stormo, G. D., Gold, L., & Ehrenfeucht, A. (1986).
    Information content of binding sites on nucleotide sequences.
    Journal of Molecular Biology, 188(3), 415-431.
    https://doi.org/10.1016/0022-2836(86)90165-8
    
    The returned quantity is the number of bits per position we expect for a
    collection of randomly selected sites.
    
    Parameters
    ----------
    n : int
        Sample size.
    base_probabilities : array-like, optional
        Probability of each base. The default is [0.25, 0.25, 0.25, 0.25].

    Returns
    -------
    E : float
        Expected value of bits per position.

    '''
    
    # Generate multinoial model
    rv = multinomial(n, base_probabilities)
    
    # Initialize while loop
    na, nc, ng, nt = n, 0, 0, 0
    done = False
    
    # Expected value of `Hnb` (entropy)
    E = 0
    while not done:
        # Probability of observing a combination with counts na, nc, ng, nt
        Pnb = rv.pmf([na, nc, ng, nt])
        # Entropy of a combination with counts na, nc, ng, nt
        Hnb = entropy([na, nc, ng, nt])
        # Contribution to the expected value of entropy
        E += Pnb * Hnb
        
        if nt > 0:
            if ng > 0:
                # Turn one G into a T
                ng -= 1
                nt += 1
            elif nc > 0:
                # Turn one C into a G, and all T to G
                nc -= 1
                ng = nt + 1
                nt = 0
            elif na > 0:
                # Turn one A into a C, and all G and T to C
                na -= 1
                nc = nt + 1
                nt = 0
            else:
                done = True  # because nt==n
        else:
            if ng > 0:
                ng -= 1
                nt += 1
            elif nc > 0:
                nc -= 1
                ng += 1
            else:
                na -= 1
                nc += 1
    return E




# ===================================================================
# Expected value of Entropy for Discrete Uniform Sample of size gamma
# ===================================================================



def append_partitions(out_list, n, *rest):
    '''
    Appends all the possible partitions of `n` to a pre-existing list `out_list`.
    Every partition is itself a list.
    '''
    out_list.append([n, *rest])
    min = rest[0] if rest else 1
    max = n // 2
    for i in range(min, max+1):
        append_partitions(out_list, n-i, i, *rest)


def count_distributions(partition, N_bins):
    '''
    Counts how many distributions correspond to a specified partition of the
    total number of elements T, where T is sum(partition).
    `partition` is assumed to be a vector of non-zero counts. The non-zero counts
    in `partition` refer to one of many possible bins. The total number of
    possible bins is `N_bins`.
    
    Example:
        N_bins = 8
        partition = [2,2,1]
        T = 5 (because 2+2+1=5)
        
        This function will return 168, because there are 168 distributions of
        five counts over 8 bins such that two bins contain 2 counts, one bin
        contains 1 count and all the ramaining five bins contain 0 counts.
        
        One of those 168 distributions would be:
            [0,0,2,1,0,2,0,0]
    '''
    # Non-empty bins
    m = len(partition)
    
    # The number of ways to map `m` non-empty bins onto `N_bins` possible bins is:
    # N_bins!/(N_bins-m)!  To avoid factorials (big nubers) I compute the ratio directly
    num = math.prod([i for i in range(N_bins, N_bins-m, -1)])
    
    # Duplicates: avoid counting multiple times the same distribution (some elements
    # are not unique).
    # Example: the third and sixth bins of [0,0,2,1,0,2,0,0] can be swapped
    den = 1
    for count_val in list(set(partition)):
        if partition.count(count_val) > 1:
            den *= math.factorial(partition.count(count_val))
    return int(num/den)


def prob_of_distribution(distribution, N_bins):
    '''
    Probability of obtaining the exact list of counts as in `distribution`,
    given that bins are equiprobable, and given that there were `N_bins` bins.
    '''
    n_empty_bins = N_bins - len(distribution)
    distribution = distribution + [0]*n_empty_bins  # Add empty bins
    bin_probs = [1/N_bins] * N_bins  # Discrete uniform over the bins
    return multinomial.pmf(distribution, n=sum(distribution), p=bin_probs)
    

def prob_of_partition(partition, N_bins):
    '''
    Returns the probability that T elements will fall into bins in a way that
    partitions T into a set of counts as `partition`. T is sum(partition).
    
    Example:
        N_bins = 8
        partition = [2,2,1]
        T = 5 (because 2+2+1=5)
        
        This function will return ~ 0.1538, which is the probability that, after
        randomly placing 5 elements in 8 bins, two bins will contain two elements,
        one bin will contain one element, and the remaining bins will be empty.
    '''
    return count_distributions(partition, N_bins) * prob_of_distribution(partition, N_bins)


def exp_entropy(n, N_bins):
    # All the partitions of n
    partitions = []
    append_partitions(partitions, n)
    # Expected value of Shannon Entropy (H)
    E = 0
    tot_p = 0
    for partition in partitions:
        if len(partition) <= N_bins:
            p = prob_of_partition(partition, N_bins)
            H = entropy(partition + [0]*(N_bins-len(partition)))
            E += p*H
            tot_p += p
    return E / tot_p

























