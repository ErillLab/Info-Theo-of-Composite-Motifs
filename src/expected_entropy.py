# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multinomial


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
        DESCRIPTION.
    base_probabilities : array-like, optional
        DESCRIPTION. The default is [0.25, 0.25, 0.25, 0.25].

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















