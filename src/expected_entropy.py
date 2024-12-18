# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multinomial
import math


def entropy(counts):
    '''
    Returns Shannon's Entropy for the `counts` vector.
    '''
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
    if n == 0:
        return None
    
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


# ===============================================================
# Expected value of Entropy for Discrete Uniform Sample of size n
# ===============================================================

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
    # Note that the zeros are not included in the lists that represent partitions,
    # so there's no need to account for the multiple zeros.
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
    '''
    Returns the expected value of Shannon's entropy for a discrete sample of
    size n, where every element is drawn from a uniform over `N_bins` many bins.
    '''
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
    print(E)
    print(tot_p)
    return E / tot_p








'''
n = 16
b = 50



def new_exp_entropy_function(b, n, *rest):
    part = [n, *rest]
    print(part)
    if len(part) <= b:
        p = prob_of_partition(part, b)
        H = entropy(part + [0]*(b-len(part)))
    i_min = rest[0] if rest else 1
    i_max = n // 2
    for i in range(i_min, i_max+1):
        new_exp_entropy_function(b, n-i, i, *rest)



new_exp_entropy_function(8)






def append_partitions_old(out_list, n, *rest):
    out_list.append([n, *rest])
    i_min = rest[0] if rest else 1
    i_max = n // 2
    for i in range(i_min, i_max+1):
        append_partitions_old(out_list, n-i, i, *rest)

def append_partitions_new(n, *rest):
    lll = [[n, *rest]]
    #print([n, *rest])
    i_min = rest[0] if rest else 1
    i_max = n // 2
    for i in range(i_min, i_max+1):
        lll += append_partitions_new(n-i, i, *rest)
    return lll


for i in range(70):
    out_list_old = []
    append_partitions_old(out_list_old, i)
    
    out_list_new = append_partitions_new(i)
    
    if out_list_old != out_list_new:
        print("!!!")
        raise ValueError('Different output')




# =======================
# Ramanujan's upper bound
# =======================

# def upper_bound(k):
#     """Ramanujan's upper bound for number of partitions of k"""
#     return int(exp(pi*sqrt(2.0*k/3.0))/(4.0*k*sqrt(3.0)))







def accel_asc(n):
    """
    By Jerome Kelleher
    (https://jeromekelleher.net/generating-integer-partitions.html)
    """
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]




import time

n = 70



for n in range(70, 95, 5):
    
    print('\n', n)
    
    out_list = []
    out_list2 = []
    out_list3 = []
    
    # 1
    start = time.time()
    append_partitions(out_list, n)
    for pp in out_list:
        pass
    end = time.time()
    print('first ', end - start)
    
    out_list = []
    
    
    # 3
    start = time.time()
    ppp = accel_asc(n)
    for pp in ppp:
        out_list3.append(pp)
    end = time.time()
    print('third ', end - start)
    
    ppp = []
    out_list3 = []
    











def partitions(n):
	# base case of recursion: zero is the sum of the empty list
	if n == 0:
		yield []
		return
		
	# modify partitions of n-1 to form partitions of n
	for p in partitions(n-1):
		yield [1] + p
		if p and (len(p) < 2 or p[1] > p[0]):
			yield [p[0] + 1] + p[1:]


newppp = partitions(16)

for pp in newppp:
    print(pp)






# ===============
# NOT EFFICIENT !
# ===============

q = { 1: [[1]] }

def decompose(n):
    try:
        return q[n]
    except:
        pass

    result = [[n]]

    for i in range(1, n):
        a = n-i
        R = decompose(i)
        for r in R:
            if r[0] <= a:
                result.append([a] + r)

    q[n] = result
    return result


# ===============
# NOT EFFICIENT enough
# ===============

def partitions_tuple(n):
    # tuple version
    if n == 0:
        yield ()
        return

    for p in partitions_tuple(n-1):
        yield (1, ) + p
        if p and (len(p) < 2 or p[1] > p[0]):
            yield (p[0] + 1, ) + p[1:]










# A utility function to print an
# array p[] of size 'n'
def printArray(p, n):
	for i in range(0, n):
		print(p[i], end = " ")
	print()

def printAllUniqueParts(n):
	p = [0] * n	 # An array to store a partition
	k = 0		 # Index of last element in a partition
	p[k] = n	 # Initialize first partition
				# as number itself

	# This loop first prints current partition, 
	# then generates next partition.The loop 
	# stops when the current partition has all 1s
	while True:
		
			# print current partition
			printArray(p, k + 1)

			# Generate next partition

			# Find the rightmost non-one value in p[]. 
			# Also, update the rem_val so that we know
			# how much value can be accommodated
			rem_val = 0
			while k >= 0 and p[k] == 1:
				rem_val += p[k]
				k -= 1

			# if k < 0, all the values are 1 so 
			# there are no more partitions
			if k < 0:
				print()
				return

			# Decrease the p[k] found above 
			# and adjust the rem_val
			p[k] -= 1
			rem_val += 1

			# If rem_val is more, then the sorted 
			# order is violated. Divide rem_val in 
			# different values of size p[k] and copy 
			# these values at different positions after p[k]
			while rem_val > p[k]:
				p[k + 1] = p[k]
				rem_val = rem_val - p[k]
				k += 1

			# Copy rem_val to next position 
			# and increment position
			p[k + 1] = rem_val
			k += 1

# Driver Code
print('All Unique Partitions of 2')
printAllUniqueParts(2)

print('All Unique Partitions of 3')
printAllUniqueParts(3)

print('All Unique Partitions of 4')
printAllUniqueParts(4)

# This code is contributed 
# by JoshuaWorthington


print('All Unique Partitions of 16')
printAllUniqueParts(16)


'''

















