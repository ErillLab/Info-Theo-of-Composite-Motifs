

import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def get_gen(filename):
    return int(filename.split('gen_')[1].split('_')[0])

def parse_report(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['IC Correction'] + list(df.columns[1:])
    return df

def merge_dataframes(list_of_df, ic_correction='true_Rseq'):
    df = pd.concat(list_of_df, ignore_index=True)
    df = df[df.iloc[:,0]==ic_correction]
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
    y1 = np.array([max_possible_Rseq(motif_len), initial_df['Rseq1_targets'].mean(), drifted_df['Rseq1_targets'].mean()])
    y2 = np.array([max_possible_Rseq(motif_len), initial_df['Rseq2_targets'].mean(), drifted_df['Rseq2_targets'].mean()])
    y3 = np.array([max_possible_Rspacer(G), initial_df['Rspacer_targets'].mean(), drifted_df['Rspacer_targets'].mean()])
    
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
    
    xdata = df['Rseq1_targets']
    ydata = df['Rseq2_targets']
    zdata = df['Rspacer_targets']
    
    ax.scatter3D(xdata, ydata, zdata, c=df['Rtot'], cmap='jet')
    
    # Axes labels
    ax.set_xlabel('Rseq1_targets')
    ax.set_ylabel('Rseq2_targets')
    ax.set_zlabel('Rspacer_targets')
    
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

def sort_filenames_by_gen(reports_filenames):
    ''' Given `reports_filenames` (a list of strings), it returns a new list
    where the file names are sorted according to the generation number. '''
    generations = [get_gen(f) for f in reports_filenames]
    return [report for gen, report in sorted(zip(generations, reports_filenames))]

def process_data(resultsdir, ic_correction='true_Rseq', sample_size=None):
    
    # CLEANUP FOLDER (remove empty/incomplete subfolders)
    for f in os.listdir(resultsdir):
        # Skip files if present
        if not os.path.isdir(resultsdir + f):
            print('Not a folder:  ' + resultsdir + f)
            continue
        # Remove empty subfolders
        if len(os.listdir(resultsdir + f)) == 0:
            print('Removing empty subfolder:', f)
            os.rmdir(resultsdir + f)
        # Remove incomplete subfolders
        else:
            contains_latest_sol = False
            for fname in os.listdir(resultsdir + f):
                if fname.startswith('sol_latest_'):
                    contains_latest_sol = True
            if not contains_latest_sol:
                '''
                print('Removing incomplete subfolder:', f)
                shutil.rmtree(resultsdir + f)
                '''
                warnings.warn('Incomplete folder: ' + f)
    
    # Input folders
    folders = [fld for fld in os.listdir(resultsdir) if os.path.isdir(resultsdir + fld)]
    # print(folders)
    if not sample_size is None:
        folders = folders[:sample_size]
    
    # Generate Dataframes
    initial = []
    drifted = []
    all_ev = dict([])
    for folder in folders[:sample_size]:
        ev_df_list = []
        sol_df_list = []
        for fname in os.listdir(resultsdir + folder):
            
            # "ev" IC reports
            if fname.startswith('ev_'):
                # Choose the IC report
                if fname.endswith('.csv'):
                    ev_df_list.append(fname)
            # "sol" IC reports
            elif fname.startswith('sol_'):
                # Choose the IC report
                if fname.endswith('.csv'):
                    sol_df_list.append(fname)
        
        ev_df_list  = sort_filenames_by_gen(ev_df_list)
        sol_df_list = sort_filenames_by_gen(sol_df_list)
        
        # ic_reports_gen = [get_gen(f) for f in ic_reports]
        # sorted_ic_reports = [report for gen, report in sorted(zip(ic_reports_gen, ic_reports))]
        if len(sol_df_list) != 2:
            warnings.warn('There should be a "sol_first_*" and a "sol_latest_*".')
            continue
            ###raise ValueError('There should be a "sol_first_*" and a "sol_latest_*".')
        initial.append(parse_report(resultsdir + folder + '/' + sol_df_list[0]))
        drifted.append(parse_report(resultsdir + folder + '/' + sol_df_list[1]))
        
        all_ev[folder] = [parse_report(resultsdir + folder + '/' + f) for f in ev_df_list]
        
    initial_df = merge_dataframes(initial)
    drifted_df = merge_dataframes(drifted)
    
    # Read parameters that were used
    parameters = read_json_file(resultsdir + folder + '/parameters.json')
    
    return initial_df, drifted_df, all_ev, parameters

def study_spacer_stacked_barplot(results_dir, drifted_df_list, parameters, sample_size=None):
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
        Rseq1_vals.append(df['Rseq1_targets'].mean())
        Rseq2_vals.append(df['Rseq2_targets'].mean())
        Rspacer_vals.append(df['Rspacer_targets'].mean())
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
    plt.legend([r'$R_{frequency}$', r'$R_{spacer}$', r'$R_{sequence(1)}$', r'$R_{sequence(2)}$'])
    ###plt.title("Dyad information content")
    plt.ylim((0,maxIC))
    plt.yticks(list(plt.yticks()[0]) + [Rfreq, maxIC])
    plt.gca().get_yticklabels()[-2].set_color("red")
    plt.gca().get_yticklabels()[-1].set_color("red")
    
    xlabels = ['Maximum IC'] + [r'$R_{spacer}$ = ' + str(int(s)) + ' bits' for s in y1[1:]]
    plt.xticks(x, xlabels, rotation=45, ha='right')
    
    # Save Figure
    if sample_size:
        filename = 'Study_Spacer_Barplot_' + str(sample_size) + '.png'
    else:
        filename = 'Study_Spacer_Barplot_ALL.png'
    figure_filepath = results_dir + filename
    plt.savefig(figure_filepath, bbox_inches="tight", dpi=600)
    plt.close()
    print('Plot saved here: ' + figure_filepath)


# ================================================
# Plot for n=1 case, to reproduce (Schneider 2000)
# ================================================


def plot_Rsequence_Ev(results_path):
    '''
    Saves a line plot as a PNG, showing the evolution of Rsequence in runs
    where n=1. It can be used to reproduce Figure 2 from (Schneider 2000).
    '''
    parameters = read_json_file(results_path + '/parameters.json')
    update_period = parameters['update_period']
    
    Rseq_list = []
    gen_list = []
    gen = 0
    ic_report_path = '{}/ev_gen_{}_ic_report.csv'.format(results_path, gen)
    while os.path.exists(ic_report_path):
        df = parse_report(ic_report_path)
        df.index = df['IC Correction'].to_list()
        Rseq_list.append(df.loc['true_Rseq','Rseq_targets'])
        gen_list.append(gen)
        gen += update_period
        ic_report_path = '{}/ev_gen_{}_ic_report.csv'.format(results_path, gen)
    
    # Rfrequency
    Rfrequency = - np.log2(parameters['gamma'] / parameters['G'])
    
    # Save data as CSV table
    df = pd.DataFrame({'Generation': gen_list, 'Rsequence': Rseq_list})
    table_filepath = results_path +'/Rsequence_evolution_data.csv'
    df.to_csv(table_filepath, index=False)
    
    # Save line plot as PNG file
    plt.hlines(Rfrequency, 0, len(Rseq_list),color='red', linestyle='dashed',
               label=r'$R_{frequency}$', zorder=2)
    plt.plot(gen_list, Rseq_list, label=r'$R_{sequence}$', zorder=1)
    y_axis_lower_bound = min(-1, min(Rseq_list))
    y_axis_upper_bound = parameters['motif_len'] * 2
    plt.ylim((y_axis_lower_bound,y_axis_upper_bound))
    plt.xlabel('Generation')
    plt.ylabel('Information (bits)')
    plt.legend()
    figure_filepath = results_path + '/Rsequence_evolution.png'
    plt.savefig(figure_filepath, bbox_inches="tight", dpi=600)
    plt.close()
    print('Plot saved here: ' + figure_filepath)


# Reproduce (Schneider 2000)
results_path = '../results/20241018161731_Reproduce_Schneider'
plot_Rsequence_Ev(results_path)


# ===============
# Analyze results (old)
# ===============


results_dir = '../results/ML6_SH4/'
sample_size = None


initial_df, drifted_df, all_ev, parameters = process_data(results_dir, sample_size)


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
results_dir = '../../RESULTS/Study_Spacer_New/'

# Apply new code on renamed old files
results_dir = '../../RESULTS/Study_Spacer_New_Repr_Renamed3/'

sample_size = 20


drifted_df_list = []
for subfolder in os.listdir(results_dir):
    if not os.path.isdir(results_dir + subfolder):
        continue
    initial_df, drifted_df, all_ev, parameters = process_data(results_dir + subfolder + '/', sample_size)
    drifted_df_list.append(drifted_df)

# Stacked Barplot
# ---------------
study_spacer_stacked_barplot(results_dir, drifted_df_list, parameters, sample_size)

# 3D plot
# -------
merged_df = merge_dataframes(drifted_df_list)
make_3d_plot(merged_df, parameters)


# Print names of runs that were used
'''
results_dir = '../../RESULTS/Study_Spacer_New/'
sample_size = 20
for subfolder in os.listdir(results_dir):
    # Skip files if present
    if not os.path.isdir(results_dir + subfolder):
        continue
    resultsdir = results_dir + '/' + subfolder + '/'
    folders = [fld for fld in os.listdir(resultsdir) if os.path.isdir(resultsdir + fld)]
    print(folders[:sample_size])
'''

# 2D plot
# -------

Rseq_list = [df['Rseq1_targets'].mean() + df['Rseq2_targets'].mean() for df in drifted_df_list[:sample_size]]
Rspacer_list = [df.loc[0,'Rspacer_targets'] for df in drifted_df_list]
gamma = parameters['gamma']
G = parameters['G']
n = parameters['motif_n']
# OVERALL LOWER BOUND: the highest between the functional lower bound and the
# informational lower bound
min_Rspacer = max(np.log2(gamma), np.log2(G/gamma))
# Upper bound is always log2(G)
max_Rspacer = np.log2(G)


min_Rsequence = 0
max_Rsequence = n * np.log2(G/gamma)
margin = 0.5

plt.xlim((0, max_Rspacer + margin))
plt.ylim((0, max_Rsequence + margin))
# Vertical lines for Rspacer bounds
plt.vlines(min_Rspacer, ymin=0, ymax=max_Rsequence + margin, colors='grey', linestyles='dashed', label='Min Rspacer',)
plt.vlines(max_Rspacer, ymin=0, ymax=max_Rsequence + margin, colors='black', linestyles='dotted', label='Max Rspacer',)

# Theoretical solutions space
x1, x2 = min_Rspacer, max_Rspacer
y1, y2 = np.log2(G**n / gamma) - x1, np.log2(G**n / gamma) - x2
plt.plot([x1, x2], [y1, y2], color='red', label='Solution space')

# Datapoints
plt.scatter(Rspacer_list, Rseq_list, label='Evolved')
plt.xlabel(r'$R_{spacer}$ (bits)')
plt.ylabel(r'$R_{sequence}$ (bits)')

plt.xticks(list(range(int(np.ceil(max_Rspacer))   + 1)))
plt.yticks(list(range(int(np.ceil(max_Rsequence)) + 1)))
plt.gca().set_aspect('equal')
plt.grid(alpha=0.5)
#plt.legend(framealpha=1, loc='upper left')
plt.legend()
figure_filepath = results_dir + '2D_plot_with_sim_data.png'
plt.savefig(figure_filepath, bbox_inches='tight', dpi=600)
#plt.show()
plt.close()
print('Plot saved here: ' + figure_filepath)




# =============
# AUPRC_fitness (old)
# =============

results_dir = '../../RESULTS/AUPRC_fitness/'
results_dir = '../../RESULTS/For_New_Spring_Const_Evo/'

sample_size = None

# XXX
subfolder = 'Gauss_9sites'
subfolder = 'Gauss_16sites'
subfolder = 'Unif_16sites'
subfolder = 'three_vals_histogram'
subfolder = 'one_val_histogram'
subfolder = 'seven_vals_histogram'

###parameters = read_json_file(results_dir + subfolder + '/20241003155050_0' + '/parameters.json')

initial_df, drifted_df, all_ev, parameters = process_data(results_dir + subfolder + '/', sample_size)

# Make Stacked Barplot
stacked_barplots(initial_df, drifted_df, parameters, filename=subfolder+"_StackedBarPlot")

# Optimal spring constant
sp = parameters['spacers']
sp_mu, sp_sigma = np.mean(sp), np.std(sp)
opt_k = (0.019235/sp_sigma)**2

# Observed spring constants
sigmas = []
variances = []
k_values = []
for fldr in os.listdir(results_dir + subfolder):
    fldr_content = os.listdir(results_dir + subfolder + '/' + fldr)
    org_f = [f for f in fldr_content if (f.startswith('sol_latest_') and f.endswith('org.json'))]
    if len(org_f) > 1:
        raise ValueError('More than one final organism.')
    org = read_json_file(results_dir + subfolder + '/' + fldr + '/' + org_f[0])
    sigmas.append(org['sigma'])
    variances.append(org['sigma']**2)
    k_values.append((0.019235/org['sigma'])**2)

k_of_mean_sigma = (0.019235/np.mean(sigmas))**2
k_of_mean_var = (0.019235**2 / np.mean(variances))


# np.log10(k_values)
np.log10(opt_k)
np.log10(k_of_mean_sigma)
np.log10(k_of_mean_var)
np.log10(np.mean(k_values))
np.mean(np.log10(k_values))



# ev spring constant
# ------------------

def study_stiffness_evolution(subfolder):
    
    meta_ev_sigmas = dict()
    meta_ev_variances = dict()
    meta_ev_k_values = dict()
    
    for fldr in os.listdir(results_dir + subfolder):
        fldr_content = os.listdir(results_dir + subfolder + '/' + fldr)
        org_f = [f for f in fldr_content if (f.startswith('ev_') and f.endswith('org.json'))]
        org_f = sort_filenames_by_gen(org_f)
        
        ev_sigmas = []
        ev_variances = []
        ev_k_values = []
        for f in org_f:
            org = read_json_file(results_dir + subfolder + '/' + fldr + '/' + f)
            ev_sigmas.append(org['sigma'])
            ev_variances.append(org['sigma']**2)
            ev_k_values.append((0.019235/org['sigma'])**2)
        
        # meta_ev_sigmas.append(ev_sigmas)
        # meta_ev_variances.append(ev_variances)
        # meta_ev_k_values.append(ev_k_values)
        
        meta_ev_sigmas[fldr] = ev_sigmas
        meta_ev_variances[fldr] = ev_variances
        meta_ev_k_values[fldr] = ev_k_values
    
    n_gen = min([len(meta_ev_k_values[run]) for run in meta_ev_k_values])
    n_gen_max = max([len(meta_ev_k_values[run]) for run in meta_ev_k_values])
    
    # Make dataframes (Missing values at the end as NaNs, due to different lengths of the columns)
    meta_ev_sigmas    = pd.DataFrame(dict([(k, pd.Series(v)) for (k, v) in meta_ev_sigmas.items()]))
    meta_ev_variances = pd.DataFrame(dict([(k, pd.Series(v)) for (k, v) in meta_ev_variances.items()]))
    meta_ev_k_values  = pd.DataFrame(dict([(k, pd.Series(v)) for (k, v) in meta_ev_k_values.items()]))
    
    
    # Add column for the average
    meta_ev_sigmas['Average']    = meta_ev_sigmas.mean(axis=1)
    meta_ev_variances['Average'] = meta_ev_variances.mean(axis=1)
    meta_ev_k_values['Average']  = meta_ev_k_values.mean(axis=1)
    
    # Add column for k estimate
    meta_ev_sigmas['k of E[sigma]']  = [(0.019235/sigma)**2 for sigma in meta_ev_sigmas['Average']]
    meta_ev_variances['k of E[var]'] = [(0.019235**2/var) for var in meta_ev_variances['Average']]
    meta_ev_k_values['k of E[k]']    = meta_ev_k_values['Average']
    
    return meta_ev_sigmas, meta_ev_variances, meta_ev_k_values, n_gen, n_gen_max

meta_ev_sigmas, meta_ev_variances, meta_ev_k_values, n_gen, n_gen_max = study_stiffness_evolution(subfolder)





key = '20240506163533_9'  # Gauss 9 example
key = '20240507003158_15'  # Gauss 16 example


# For poster
key = '20240506163533_9'  # Gauss 9 example
plt.rcParams.update({'font.size': 14})
plt.plot(meta_ev_k_values[key], label=r'$\kappa$: spring constant of most fit organism')
plt.hlines(opt_k, xmin=0, xmax=len(meta_ev_k_values[key]),
           colors='red', linestyles='dashed', label=r'$\kappa_{opt}$: predicted optimal spring constant')
plt.xlabel('generation')
plt.ylabel('spring constant (N/m)')
plt.xlim((0,40))  # !!! < < < < < < < < < < < < < < < < < < < < < < < < < < < <
plt.ylim((0,0.001))
plt.legend()
plt.tight_layout()
plt.savefig('SPRING_CONSTANT_evolution0.png', dpi=300)



# For extra Figure in paper
key = '20240923161633_6'

plt.rcParams.update({'font.size': 14})
plt.plot(meta_ev_k_values[key], label=r'$\kappa$: spring constant of most fit organism')
plt.hlines(opt_k, xmin=0, xmax=len(meta_ev_k_values[key]),
           colors='red', linestyles='dashed', label=r'$\kappa_{opt}$: predicted optimal spring constant')
plt.xlabel('generation')
plt.ylabel('spring constant (N/m)')
plt.xlim((0,100))  # !!! < < < < < < < < < < < < < < < < < < < < < < < < < < < <
plt.ylim((0,0.001))
plt.legend()
plt.show()

plt.tight_layout()
plt.savefig('SPRING_CONSTANT_evolution0.png', dpi=300)











# Average across runs
# -------------------


# avg_sigma = []
# avg_var = []
# avg_k = []
# for i in range(n_gen):
#     avg_sigma.append(np.mean([run[i] for run in meta_ev_sigmas]))
#     avg_var.append(np.mean([run[i] for run in meta_ev_variances]))
#     avg_k.append(np.mean([run[i] for run in meta_ev_k_values]))

# avg1 = [(0.019235/sigma)**2 for sigma in avg_sigma]
# avg2 = [(0.019235**2/var) for var in avg_var]
# avg3 = avg_k[:]



# avg1 = [(0.019235/sigma)**2 for sigma in avg_sigma]
# avg2 = [(0.019235**2/var) for var in avg_var]
# avg3 = avg_k[:]


# Plot using log scale
plt.plot(np.log10(meta_ev_sigmas['k of E[sigma]']), label='from E[sigma]')
plt.plot(np.log10(meta_ev_variances['k of E[var]']), label='from E[Var]')
plt.plot(np.log10(meta_ev_k_values['k of E[k]']), label='from E[k]')
plt.hlines(np.log10(opt_k), xmin=0, xmax=len(meta_ev_sigmas['k of E[sigma]']), colors='red')
plt.ylabel('spring constant (N/m)')
plt.ylim((-5,0))
plt.legend()

# Plot using linear scale
plt.plot(meta_ev_sigmas['k of E[sigma]'], label='from E[sigma]')
plt.plot(meta_ev_variances['k of E[var]'], label='from E[Var]')
plt.plot(meta_ev_k_values['k of E[k]'], label='from E[k]')
plt.hlines(opt_k, xmin=0, xmax=len(meta_ev_sigmas['k of E[sigma]']), colors='red')
plt.ylabel('spring constant (N/m)')
#plt.ylim((0,0.005))
plt.legend()


cols = ['20240506163533_0', '20240506163533_1', '20240506163533_10',
       '20240506163533_11', '20240506163533_12', '20240506163533_13',
       '20240506163533_14', '20240506163533_15', '20240506163533_2',
       '20240506163533_3', '20240506163533_4', '20240506163533_5',
       '20240506163533_6', '20240506163533_7', '20240506163533_8',
       '20240506163533_9']

# cols = ['20240507003157_3', '20240507003157_7', '20240507003158_0',
#        '20240507003158_1', '20240507003158_12', '20240507003158_13',
#        '20240507003158_14', '20240507003158_15', '20240507003158_2',
#        '20240507003158_5', '20240507003158_6', '20240507003158_9',
#        '20240507003159_10', '20240507003159_4']

cols = list(meta_ev_sigmas.columns)[:-2]

k_estimates = []
iteration = []
for col in cols:
    ser = meta_ev_k_values[col].dropna()
    k_estimates += ser.to_list()
    iteration += ser.index.to_list()
new_dict = {'k estimates': k_estimates, 'iter': iteration}
new_df = pd.DataFrame(new_dict)

for interval in [10, 25, 50, 95, 100]:
    plot = sns.lineplot(data=new_df, x='iter', y='k estimates', estimator=np.median,
                        color="C0", errorbar=('ci', interval))
# plot.set(ylim=(0, max(new_df['k estimates'])))
plot.set(ylim=(0, 0.005))

plot.axhline(y=opt_k)


# # outlier removed for Gauss16
# selected_cols = ['20240507003157_3', '20240507003158_0',
#         '20240507003158_1', '20240507003158_12', '20240507003158_13',
#         '20240507003158_14', '20240507003158_15', '20240507003158_2',
#         '20240507003158_5', '20240507003158_6', '20240507003158_9',
#         '20240507003159_10', '20240507003159_4']
plt.ylim((0,0.005))
sns.lineplot(data=meta_ev_k_values.loc[:,cols], estimator="median", errorbar=("pi", 50))
sns.lineplot().axhline(y=opt_k)
plt.ylim((0,0.005))


# Final k values
final_k_vals = [run[n_gen-1] for run in meta_ev_k_values]
x = [1] * len(final_k_vals)
plt.scatter(x, np.log10(final_k_vals))
plt.hlines(np.log10(opt_k), xmin=0, xmax=2, colors='red')
plt.ylabel('spring constant (N/m)')
plt.ylim((-5,0))

# All k values
for i in range(n_gen):
    gen_k_vals = [run[i] for run in meta_ev_k_values]
    x = [i+1] * len(gen_k_vals)
    plt.scatter(x, np.log10(gen_k_vals), alpha=0.2)
plt.hlines(np.log10(opt_k), xmin=0, xmax=n_gen, colors='red')
plt.ylabel('spring constant (N/m)')
plt.ylim((-5,0))


# =============================================
# For paper Figure on spring constant evolution
# =============================================

results_dir = '../../RESULTS/For_New_Spring_Const_Evo/'
sample_size = None


subfolder_5 = 'Gauss_9sites'
subfolder_3 = 'three_vals_histogram'
subfolder_1 = 'one_val_histogram'
subfolder_7 = 'seven_vals_histogram'


def study_stiffness_evolution(subfolder):
    
    meta_ev_sigmas = dict()
    meta_ev_variances = dict()
    meta_ev_k_values = dict()
    
    for fldr in os.listdir(results_dir + subfolder):
        fldr_content = os.listdir(results_dir + subfolder + '/' + fldr)
        org_f = [f for f in fldr_content if (f.startswith('ev_') and f.endswith('org.json'))]
        org_f = sort_filenames_by_gen(org_f)
        
        ev_sigmas = []
        ev_variances = []
        ev_k_values = []
        for f in org_f:
            org = read_json_file(results_dir + subfolder + '/' + fldr + '/' + f)
            ev_sigmas.append(org['sigma'])
            ev_variances.append(org['sigma']**2)
            ev_k_values.append((0.019235/org['sigma'])**2)
        
        # meta_ev_sigmas.append(ev_sigmas)
        # meta_ev_variances.append(ev_variances)
        # meta_ev_k_values.append(ev_k_values)
        
        meta_ev_sigmas[fldr] = ev_sigmas
        meta_ev_variances[fldr] = ev_variances
        meta_ev_k_values[fldr] = ev_k_values
    
    n_gen = min([len(meta_ev_k_values[run]) for run in meta_ev_k_values])
    n_gen_max = max([len(meta_ev_k_values[run]) for run in meta_ev_k_values])
    
    # Make dataframes (Missing values at the end as NaNs, due to different lengths of the columns)
    meta_ev_sigmas    = pd.DataFrame(dict([(k, pd.Series(v)) for (k, v) in meta_ev_sigmas.items()]))
    meta_ev_variances = pd.DataFrame(dict([(k, pd.Series(v)) for (k, v) in meta_ev_variances.items()]))
    meta_ev_k_values  = pd.DataFrame(dict([(k, pd.Series(v)) for (k, v) in meta_ev_k_values.items()]))
    
    
    # Add column for the average
    meta_ev_sigmas['Average']    = meta_ev_sigmas.mean(axis=1)
    meta_ev_variances['Average'] = meta_ev_variances.mean(axis=1)
    meta_ev_k_values['Average']  = meta_ev_k_values.mean(axis=1)
    
    # Add column for k estimate
    meta_ev_sigmas['k of E[sigma]']  = [(0.019235/sigma)**2 for sigma in meta_ev_sigmas['Average']]
    meta_ev_variances['k of E[var]'] = [(0.019235**2/var) for var in meta_ev_variances['Average']]
    meta_ev_k_values['k of E[k]']    = meta_ev_k_values['Average']
    
    return meta_ev_sigmas, meta_ev_variances, meta_ev_k_values, n_gen, n_gen_max


def make_df_for_avg_spring_const_plot(meta_ev_sigmas, meta_ev_k_values, update_period):
    cols = list(meta_ev_sigmas.columns)[:-2]
    
    k_estimates = []
    iteration = []
    for col in cols:
        ser = meta_ev_k_values[col].dropna()
        k_estimates += ser.to_list()
        iters = ser.index.to_list()
        
        iteration += [it * update_period for it in iters]
    new_dict = {'k estimates': k_estimates, 'iter': iteration}
    new_df = pd.DataFrame(new_dict)
    return new_df

THRSH = 15

initial_df_7, drifted_df_57, all_ev_7, parameters_7 = process_data(results_dir + subfolder_7 + '/', sample_size)
meta_ev_sigmas_7, meta_ev_variances_7, meta_ev_k_values_7, n_gen_7, n_gen_max_7 = study_stiffness_evolution(subfolder_7)
meta_ev_sigmas_7 = meta_ev_sigmas_7.dropna(thresh=THRSH)
meta_ev_k_values_7 = meta_ev_k_values_7.dropna(thresh=THRSH)
n_iter_7 = len(meta_ev_k_values_7)
opt_k_7 = (0.019235/np.std(parameters_7['spacers']))**2
#new_df_7 = make_df_for_avg_spring_const_plot(meta_ev_sigmas_7, meta_ev_k_values_7)

# initial_df_5, drifted_df_5, all_ev_5, parameters_5 = process_data(results_dir + subfolder_5 + '/', sample_size)
# meta_ev_sigmas_5, meta_ev_variances_5, meta_ev_k_values_5, n_gen_5, n_gen_max_5 = study_stiffness_evolution(subfolder_5)
# opt_k_5 = (0.019235/np.std(parameters_5['spacers']))**2
# new_df_5 = make_df_for_avg_spring_const_plot(meta_ev_sigmas_5, meta_ev_k_values_5)

initial_df_3, drifted_df_3, all_ev_3, parameters_3 = process_data(results_dir + subfolder_3 + '/', sample_size)
meta_ev_sigmas_3, meta_ev_variances_3, meta_ev_k_values_3, n_gen_3, n_gen_max_3 = study_stiffness_evolution(subfolder_3)
meta_ev_sigmas_3 = meta_ev_sigmas_3.dropna(thresh=THRSH)
meta_ev_k_values_3 = meta_ev_k_values_3.dropna(thresh=THRSH)
n_iter_3 = len(meta_ev_k_values_3)
opt_k_3 = (0.019235/np.std(parameters_3['spacers']))**2
#new_df_3 = make_df_for_avg_spring_const_plot(meta_ev_sigmas_3, meta_ev_k_values_3)

# initial_df_1, drifted_df_1, all_ev_1, parameters_1 = process_data(results_dir + subfolder_1 + '/', sample_size)
# meta_ev_sigmas_1, meta_ev_variances_1, meta_ev_k_values_1, n_gen_1, n_gen_max_1 = study_stiffness_evolution(subfolder_1)
# opt_k_1 = (0.019235/np.std(parameters_1['spacers']))**2
# new_df_1 = make_df_for_avg_spring_const_plot(meta_ev_sigmas_1, meta_ev_k_values_1)






stop = min(n_iter_3, n_iter_7)

meta_ev_sigmas_7 = meta_ev_sigmas_7.iloc[:stop,:]
meta_ev_k_values_7 = meta_ev_k_values_7.iloc[:stop,:]
new_df_7 = make_df_for_avg_spring_const_plot(meta_ev_sigmas_7, meta_ev_k_values_7, parameters_7['update_period'])

meta_ev_sigmas_3 = meta_ev_sigmas_3.iloc[:stop,:]
meta_ev_k_values_3 = meta_ev_k_values_3.iloc[:stop,:]
new_df_3 = make_df_for_avg_spring_const_plot(meta_ev_sigmas_3, meta_ev_k_values_3, parameters_3['update_period'])


# intervals = [10, 25, 50, 95, 100]
# intervals = [25, 50, 75, 95]
# intervals = [50, 75, 100]
intervals = [50, 75]

yaxis_scale = 'log'
yaxis_scale = 'linear'

xaxis_lim = min(n_gen_max_3, n_gen_max_7)

# kappa_opt lines
# plot.axhline(y=opt_k_7, linestyle='dashed', color='C0', label=r'$\kappa_{opt}^{(1)}$', zorder=1)
# plot.axhline(y=opt_k_3, linestyle='dashed', color='C1', label=r'$\kappa_{opt}^{(2)}$', zorder=2)

# 7 val hist
#   * mid 50% of the data
plot = sns.lineplot(data=new_df_7, x='iter', y='k estimates', estimator=np.median,
             color="C0", errorbar=('ci', 50), label=r'Simulations using $D_{1}$', zorder=3)
# plot = sns.lineplot(data=new_df_7, x='iter', y='k estimates', estimator=np.median,
#              color="C0", errorbar=('ci', 50), label=r'$\kappa^{(1)}$')
#   * mid 75% of the data
plot = sns.lineplot(data=new_df_7, x='iter', y='k estimates', estimator=np.median,
             color="C0", errorbar=('ci', 75), zorder=4)
plt.yscale(yaxis_scale)
plot.axhline(y=opt_k_7, linestyle='dashed', color='C0', label=r'$\kappa_{opt}^{(1)}$', zorder=1)


# # 5 val hist
# for interval in intervals:
#     plot = sns.lineplot(data=new_df_5, x='iter', y='k estimates', estimator=np.median,
#                         color="C0", errorbar=('ci', interval))
# # plot.set(ylim=(0, max(new_df['k estimates'])))
# ###plot.set(ylim=(0, 0.005))
# plt.yscale(yaxis_scale)
# plot.axhline(y=opt_k_5, linestyle='dashed', color='C0')

# 3 val hist
#   * mid 50% of the data
plot = sns.lineplot(data=new_df_3, x='iter', y='k estimates', estimator=np.median,
             color="C1", errorbar=('ci', 50), label=r'Simulations using $D_{2}$', zorder=5)
# plot = sns.lineplot(data=new_df_3, x='iter', y='k estimates', estimator=np.median,
#              color="C1", errorbar=('ci', 50), label=r'$\kappa^{(2)}$')
#   * mid 75% of the data
plot = sns.lineplot(data=new_df_3, x='iter', y='k estimates', estimator=np.median,
             color="C1", errorbar=('ci', 75), zorder=6)
plt.yscale(yaxis_scale)
plot.axhline(y=opt_k_3, linestyle='dashed', color='C1', label=r'$\kappa_{opt}^{(2)}$', zorder=2)


# # 1 val hist
# for interval in intervals:
#     plot = sns.lineplot(data=new_df_1, x='iter', y='k estimates', estimator=np.median,
#                         color="C2", errorbar=('ci', interval))
# # plot.set(ylim=(0, max(new_df['k estimates'])))
# ###plot.set(ylim=(0, 0.005))
# plt.yscale(yaxis_scale)
# plot.axhline(y=opt_k_1, linestyle='dashed', color='C2')

plot.set(ylim=(-0.0005, 0.005))
plt.ylabel(r'$\kappa$ (N/m)')
plt.xlabel('Generation')
plt.legend()
figure_filepath = results_dir + 'Spring_Constant_Evo_Figure.png'
plt.savefig(figure_filepath, dpi=600)
plt.close()
print('Plot saved as: ' + figure_filepath)

# Now plot the histograms for the two spacer size distributions
# ---------------------------------------------------------

spacers_7 = parameters_7['spacers']
spacers_7_unique = list(set(spacers_7))
spacers_7_unique.sort()

spacers_3 = parameters_3['spacers']
spacers_3_unique = list(set(spacers_3))
spacers_3_unique.sort()

x_min = min(min(spacers_7_unique), min(spacers_3_unique))
x_max = max(max(spacers_7_unique), max(spacers_3_unique))
x_axis = list(range(x_min, x_max+1))

y_7 = []
y_3 = []
for spacer_size in x_axis:
    y_7.append(spacers_7.count(spacer_size))
    y_3.append(spacers_3.count(spacer_size))

y_max = max(max(y_7), max(y_3))

labels_fontsize = 42
ticks_fontsize = 28

# Plot 7 vals hist
plt.bar(x_axis, y_7, width=1, color='#1f77b4', edgecolor='black')
plt.title(r'$D_{1}$', fontsize=labels_fontsize)
plt.xlabel('Spacer size (bp)', fontsize=labels_fontsize)
plt.ylabel('# targets', fontsize=labels_fontsize)
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.ylim((0,y_max+0.5))
figure_filepath = results_dir + 'Hist_for_D1.png'
plt.savefig(figure_filepath, bbox_inches="tight", dpi=300)
plt.close()

# Plot 3 vals hist
plt.bar(x_axis, y_3, width=1, color='#ff7f0e', edgecolor='black')
plt.title(r'$D_{2}$', fontsize=labels_fontsize)
plt.xlabel('Spacer size (bp)', fontsize=labels_fontsize)
plt.ylabel('# targets', fontsize=labels_fontsize)
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.ylim((0,y_max+0.5))
figure_filepath = results_dir + 'Hist_for_D2.png'
plt.savefig(figure_filepath, bbox_inches="tight", dpi=300)
plt.close()



# ev information
# --------------

ev_runs = []
for key in all_ev.keys():
    run = all_ev[key]
    df = pd.concat(run, ignore_index=True)
    df = df.loc[df['IC Correction'] == 'true_Rseq']
    df = df.reset_index(drop=True)
    ev_runs.append(df)


subdir = 'Unif_16sites_Evo_Stack_Rconn'
for i in range(len(ev_runs)):
    df = ev_runs[i]
    df = df.iloc[:25,:]
    plt.stackplot(list(range(len(df))), df['Rseq1_targets'], df['Rseq2_targets'],
                  df['Rconnector'], baseline ='zero')
    plt.hlines(df['Rfrequency'], xmin=0, xmax=len(df), color='red')
    plt.ylim((0, max_possible_diad_IC(parameters['G'], parameters['motif_len'])))
    outfilepath = '../results/AUPRC_fitness/{}/Evo_Stack_Rconn_Run_{}.png'.format(subdir, i)
    plt.savefig(outfilepath)
    plt.close()

# ev Rspacer VS Rconnector

rst = drifted_df['Rspacer_targets']
rsh = drifted_df['Rspacer_hits']
rc = drifted_df['Rconnector']

plt.scatter(rst, rc)


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








































