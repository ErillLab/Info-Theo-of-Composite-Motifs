# -*- coding: utf-8 -*-
"""
Generate logos from *_map.txt files.

"""

import re
#from Bio import motifs

#filepath = '../results/Study_Spacer/ML6_SH0_G512_Gamma16/ML6_SH0_G512_Gamma16_20240210094831_5/gen_10403_map.txt'
filepath = '../results/Study_Spacer/ML6_SH4_G512_Gamma16/ML6_SH4_G512_Gamma16_20240213125554_0/gen_10847_map.txt'

# Read genome map file
with open(filepath, 'r') as f:
    lines = f.readlines()
seq_str, sites_str = lines[1], lines[3]



L_starts = [m.start() for m in re.finditer('LLLL', sites_str)]
R_starts = [m.start() for m in re.finditer('RRRR', sites_str)]
for i in range(16):
    if i < 10:
        L_starts[i] -= 1
        R_starts[i] -= 1
    else:
        L_starts[i] -= 2
        R_starts[i] -= 2
L_sequences = [seq_str[start:start+6] for start in L_starts]
R_sequences = [seq_str[start:start+6] for start in R_starts]

# motifs.create([inst.upper() for inst in L_sequences])

for inst in L_sequences:
    print(inst)

for inst in R_sequences:
    print(inst)



































