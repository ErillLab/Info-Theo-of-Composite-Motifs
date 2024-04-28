

import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_gen(filename):
    return int(filename.split('_')[1])

def parse_report(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['IC Correction'] + list(df.columns[1:])
    return df

def merge_dataframes(list_of_df):
    df = pd.concat(list_of_df, ignore_index=True)
    df = df[df.iloc[:,0]=='corrected']
    return df.reset_index(drop=True)    

def max_possible_Rseq(motif_len):
    return 2 * motif_len

def max_possible_Rspacer(G):
    return np.log2(G)

def max_possible_diad_IC(G, motif_len):
    return  2 * max_possible_Rseq(motif_len) + max_possible_Rspacer(G)

def diad_Rfreq(G, gamma):
    return np.log2(G**2/gamma)

def read_json_file(filename):
    with open(filename) as json_content:
        return json.load(json_content)

def stacked_barplots(initial_df, drifted_df, parameters, filename='barplots.png'):
    
    # Read parameters
    G = parameters['G']
    gamma = parameters['gamma']
    motif_len = parameters['motif_len']
    
    # Data for bars (average IC)
    x = [1, 2, 3]
    y1 = np.array([max_possible_Rseq(motif_len), initial_df['Rseq1'].mean(), drifted_df['Rseq1'].mean()])
    y2 = np.array([max_possible_Rseq(motif_len), initial_df['Rseq2'].mean(), drifted_df['Rseq2'].mean()])
    y3 = np.array([max_possible_Rspacer(G), initial_df['Rspacer'].mean(), drifted_df['Rspacer'].mean()])
    
    # Maximum IC and predicted IC
    maxIC = max_possible_diad_IC(G, motif_len)
    Rfreq = diad_Rfreq(G, gamma)
    
    # Plot horizontal line for Rfrequency
    plt.hlines(Rfreq, xmin=min(x)-0.5, xmax=max(x)+0.5, linestyles='dashed', color='r')
    
    # Plot bars with 95% Confidence Interval (CI)
    # Note that 95% CI = 1.96 * SEM  [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3487226/]
    initial_95ci = initial_df['Rtot'].sem() * 1.96
    drifted_95ci = drifted_df['Rtot'].sem() * 1.96
    plt.bar(x, y1)
    plt.bar(x, y2, bottom=y1)
    plt.bar(x, y3, bottom=y1+y2, yerr=(0, initial_95ci, drifted_95ci), capsize=4)
    # Labels and legend
    plt.ylabel("Information (bits)")
    plt.legend(["Rfreq(diad)", "Rseq1", "Rseq2", "Rspacer"])
    plt.title("Diad information content")
    plt.ylim((0,maxIC))
    plt.yticks(list(plt.yticks()[0]) + [Rfreq, maxIC])
    plt.gca().get_yticklabels()[-2].set_color("red")
    plt.gca().get_yticklabels()[-1].set_color("red")
    xlabels = ['Maximum IC','First solution','After drift time']
    plt.xticks(x, xlabels, rotation='vertical')
    
    # Save Figure
    filepath = results_dir + filename
    plt.savefig(filepath, bbox_inches="tight", dpi=600)
    plt.close()

def make_3d_plot(df, parameters):
    
    # Paramters
    gamma = parameters['gamma']
    G = parameters['G']
    motif_len = parameters['motif_len']
    
    # 3D scatter plot
    # ---------------
    
    ax = plt.axes(projection='3d')
    
    xdata = df['Rseq1']
    ydata = df['Rseq2']
    zdata = df['Rspacer']
    
    ax.scatter3D(xdata, ydata, zdata, c=df['Rtot'], cmap='jet')
    
    # Axes labels
    ax.set_xlabel('Rseq1')
    ax.set_ylabel('Rseq2')
    ax.set_zlabel('Rspacer')
    
    # Theoretical surface
    # -------------------
    
    # Bounds
    tot_info = -np.log2(gamma/(G**2))
    # max_x = -np.log2(gamma/G)
    # max_y = -np.log2(gamma/G)
    max_x = 2*motif_len
    max_y = 2*motif_len
    min_z = np.log2(gamma)
    max_z = np.log2(G)
    
    # Plot surface
    res = 300
    x_upper = -np.log2(gamma/G)
    y_upper = -np.log2(gamma/G)
    x = np.outer(np.linspace(0, x_upper, res), np.ones(res))
    y = np.outer(np.linspace(0, y_upper, res), np.ones(res))
    y = y.T
    z = tot_info - x - y
    z[z > max_z] = np.nan
    z[z < min_z] = np.nan
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor=None, alpha=0.5)
    
    # Minimum Rspacer plane
    min_z_plane = np.outer(np.array([min_z]*2), np.ones(2))
    ax.plot_surface([[0, max_x],[0, max_x]], [[0, 0],[max_y, max_y]], min_z_plane,
                    edgecolor='black', alpha=0.0)
    ax.set_zlim((0,max_z))
    
    # Plot "triangle" lines
    Rfreq_mono = -np.log2(gamma/G)
    ax.plot([Rfreq_mono,Rfreq_mono], [0,Rfreq_mono], [max_z,min_z], color='grey')
    ax.plot([0,Rfreq_mono], [Rfreq_mono,Rfreq_mono], [max_z,min_z], color='grey')
    ax.plot([Rfreq_mono,0], [0,Rfreq_mono], [max_z,max_z], color='grey')
    
    # Show interactive plot
    plt.show()

def process_data(resultsdir, sample_size=None):
    
    # CLEANUP FOLDER (remove empty/incomplete subfolders)
    for f in os.listdir(resultsdir):
        # Skip files if present
        if not os.path.isdir(resultsdir + f):
            continue
        # Remove empty subfolders
        if len(os.listdir(resultsdir + f)) == 0:
            print('Removing empty subfolder:', f)
            os.rmdir(resultsdir + f)
        # Remove incomplete subfolders ("complete" subfolders contain 9 items)
        elif len(os.listdir(resultsdir + f)) != 9:
            print('Removing incomplete subfolder:', f)
            shutil.rmtree(resultsdir + f)
    
    # Input folders
    folders = [f for f in os.listdir(resultsdir) if os.path.isdir(resultsdir + f)]
    print(folders)
    if not sample_size is None:
        folders = folders[:sample_size]
    
    # Generate Dataframes
    initial = []
    drifted = []
    for folder in folders[:sample_size]:
        ic_reports = [f for f in os.listdir(resultsdir + folder) if f[-4:] == '.csv']
        ic_reports_gen = [get_gen(f) for f in ic_reports]
        sorted_ic_reports = [report for gen, report in sorted(zip(ic_reports_gen, ic_reports))]
        initial.append(parse_report(resultsdir + folder + '/' + sorted_ic_reports[0]))
        drifted.append(parse_report(resultsdir + folder + '/' + sorted_ic_reports[1]))
    initial_df = merge_dataframes(initial)
    drifted_df = merge_dataframes(drifted)
    
    # Read parameters that were used
    parameters = read_json_file(resultsdir + folder + '/parameters.json')
    
    return initial_df, drifted_df, parameters

def study_spacer_stacked_barplot(drifted_df_list, parameters, sample_size=None):
    '''
    Saves a stacked barplot for the Spacer-Study as a PNG file.
      - `drifted_df_list` is a list of pandas dataframes (with the 'drifted' values)
      - `parameters` is a dictionary from the JSON parameters file used in the Spacer-Study
    
    '''
    
    Rseq1_vals   = []
    Rseq2_vals   = []
    Rspacer_vals = []
    Rtot_CI_vals = []
    for df in drifted_df_list:
        Rseq1_vals.append(df['Rseq1'].mean())
        Rseq2_vals.append(df['Rseq2'].mean())
        Rspacer_vals.append(df['Rspacer'].mean())
        Rtot_CI_vals.append(df['Rtot'].sem() * 1.96)
        # Note that 95% CI = 1.96 * SEM  [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3487226/]
    
    
    # Read parameters
    G = parameters['G']
    gamma = parameters['gamma']
    motif_len = parameters['motif_len']
    
    # Data for bars (average IC)
    x = list(range(len(drifted_df_list)+1))
    y1 = np.array([max_possible_Rspacer(G)] + Rspacer_vals)
    y2 = np.array([max_possible_Rseq(motif_len)] + Rseq1_vals)
    y3 = np.array([max_possible_Rseq(motif_len)] + Rseq2_vals)
    
    # Maximum IC and predicted IC
    maxIC = max_possible_diad_IC(G, motif_len)
    Rfreq = diad_Rfreq(G, gamma)
    
    # Plot horizontal line for Rfrequency
    plt.hlines(Rfreq, xmin=min(x)-0.5, xmax=max(x)+0.5, linestyles='dashed', color='r')
    
    plt.bar(x, y1)
    plt.bar(x, y2, bottom=y1)
    plt.bar(x, y3, bottom=y1+y2, yerr=([0] + Rtot_CI_vals), capsize=4)
    # Labels and legend
    plt.ylabel("Information (bits)")
    plt.legend(["Rfreq(diad)", "Rspacer", "Rseq1", "Rseq2"])
    plt.title("Diad information content")
    plt.ylim((0,maxIC))
    plt.yticks(list(plt.yticks()[0]) + [Rfreq, maxIC])
    plt.gca().get_yticklabels()[-2].set_color("red")
    plt.gca().get_yticklabels()[-1].set_color("red")
    
    xlabels = ['Maximum IC'] + ['Rspacer = ' + str(int(s)) + ' bits' for s in y1[1:]]
    plt.xticks(x, xlabels, rotation='vertical')
    
    # Save Figure
    if sample_size:
        filename = 'Study_Spacer_Barplot_' + str(sample_size) + '.png'
    else:
        filename = 'Study_Spacer_Barplot_ALL.png'
    filepath = results_dir + filename
    plt.savefig(filepath, bbox_inches="tight", dpi=600)
    plt.close()


# ===============
# Analyze results
# ===============

# !!! Settings
results_dir = '../results/ML6_SH4/'
sample_size = None


initial_df, drifted_df, parameters = process_data(results_dir, sample_size)



# Save Stacked Bar Plot
if sample_size:
    filename = 'barplots_{}.png'.format(sample_size)
else:
    filename = 'barplots_ALL_{}.png'.format(len(drifted_df))
stacked_barplots(initial_df, drifted_df, parameters, filename)

# 3D plot initial
make_3d_plot(initial_df, parameters)
# 3D plot drifted
make_3d_plot(drifted_df, parameters)




# ===========
# NEW BARPLOT
# ===========


# XXX ...

results_dir = '../results/Study_Spacer_New/'
sample_size = 20


drifted_df_list = []
for f in os.listdir(results_dir):
    if not os.path.isdir(results_dir + f):
        continue
    initial_df, drifted_df, parameters = process_data(results_dir + '/' + f + '/', sample_size)
    drifted_df_list.append(drifted_df)

# Make Stacked Barplot
study_spacer_stacked_barplot(drifted_df_list, parameters, sample_size)

# 3D plot
merged_df = merge_dataframes(drifted_df_list)
make_3d_plot(merged_df, parameters)


# 2D plot


# =======================
# 2D plot
# =======================


Rseq_list = [df['Rseq1'].mean() + df['Rseq2'].mean() for df in drifted_df_list]
Rspacer_list = [df.loc[0,'Rspacer'] for df in drifted_df_list]
gamma = parameters['gamma']
G = parameters['G']
n = parameters['motif_n']
min_Rspacer = max(np.log2(gamma), np.log2(G/gamma))
max_Rspacer = np.log2(G)






min_Rspacer = max(np.log2(gamma), np.log2(G/gamma))
max_Rspacer = np.log2(G)
min_Rsequence = 0
max_Rsequence = n * np.log2(G/gamma)
margin = 0.5

plt.xlim((0, max_Rspacer + margin))
plt.ylim((0, max_Rsequence + margin))
# Vertical lines for Rspacer bounds
plt.vlines(min_Rspacer, ymin=0, ymax=max_Rsequence + margin, colors='grey', linestyles='dashed', label='Min Rspacer',)
plt.vlines(max_Rspacer, ymin=0, ymax=max_Rsequence + margin, colors='black', linestyles='dotted', label='Max Rspacer',)
# Datapoints
plt.scatter(Rspacer_list, Rseq_list)
plt.xlabel('Rspacer')
plt.ylabel('Rsequence')
# Theoretical phase space
x1, x2 = min_Rspacer, max_Rspacer
y1, y2 = np.log2(G**n / gamma) - x1, np.log2(G**n / gamma) - x2
plt.plot([x1, x2], [y1, y2], color='red', label='Theoretical')

plt.xticks(list(range(int(np.ceil(max_Rspacer))   + 1)))
plt.yticks(list(range(int(np.ceil(max_Rsequence)) + 1)))
plt.gca().set_aspect('equal')
plt.grid(alpha=0.5)
plt.legend(framealpha=1, loc='upper left')
plt.savefig('twoD_plot_Spacer_Study2.png', dpi=600)
plt.show()











gamma, G = 50, 4E6


max_Rsequence = n * np.log2(G/gamma)
margin = 0.5

plt.xlim((0, np.log2(G) + margin))
plt.ylim((0, max_Rsequence + margin))
# Vertical lines for Rspacer bounds
plt.vlines(np.log2(gamma), ymin=0, ymax=max_Rsequence + margin, colors='black', linestyles='solid', label='Min Rspacer',)
plt.vlines(np.log2(G),     ymin=0, ymax=max_Rsequence + margin, colors='black', linestyles='solid', alpha=0.3,)

plt.xlabel('Rspacer')
plt.ylabel('Rsequence')
# Theoretical phase space
x1, x2 = np.log2(gamma), np.log2(G)
y1, y2 = np.log2(G**n / gamma) - x1, np.log2(G**n / gamma) - x2
plt.plot([x1, x2], [y1, y2], color='red', label='Theoretical')
plt.hlines(y1, xmin=0, xmax=np.log2(G), color='black', alpha=0.3)
plt.hlines(y2, xmin=0, xmax=np.log2(G), color='black', alpha=0.3)
plt.gca().set_aspect('equal')
plt.grid(alpha=0.5)
#plt.legend(framealpha=1)
plt.show()








































