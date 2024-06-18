

import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
                print('Removing incomplete subfolder:', f)
                shutil.rmtree(resultsdir + f)
    
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
            raise ValueError('There should be a "sol_first_*" and a "sol_latest_*".')
        initial.append(parse_report(resultsdir + folder + '/' + sol_df_list[0]))
        drifted.append(parse_report(resultsdir + folder + '/' + sol_df_list[1]))
        
        all_ev[folder] = [parse_report(resultsdir + folder + '/' + f) for f in ev_df_list]
        
    initial_df = merge_dataframes(initial)
    drifted_df = merge_dataframes(drifted)
    
    # Read parameters that were used
    parameters = read_json_file(resultsdir + folder + '/parameters.json')
    
    return initial_df, drifted_df, all_ev, parameters

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
sample_size = 20


drifted_df_list = []
for subfolder in os.listdir(results_dir):
    if not os.path.isdir(results_dir + subfolder):
        continue
    initial_df, drifted_df, all_ev, parameters = process_data(results_dir + '/' + subfolder + '/', sample_size)
    drifted_df_list.append(drifted_df)

# Make Stacked Barplot
study_spacer_stacked_barplot(drifted_df_list, parameters, sample_size)

# 3D plot
merged_df = merge_dataframes(drifted_df_list)
make_3d_plot(merged_df, parameters)



# =============
# AUPRC_fitness
# =============

results_dir = '../results/AUPRC_fitness/'
sample_size = None

# XXX
subfolder = 'Gauss_9sites'
subfolder = 'Gauss_16sites'
subfolder = 'Unif_16sites'

initial_df, drifted_df, all_ev, parameters = process_data(results_dir + subfolder + '/', sample_size)

# Make Stacked Barplot
stacked_barplots(initial_df, drifted_df, parameters, filename=subfolder+"_StackedBarPlot0000")

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

# Make dataframes (Missing values at the end as NaNs, due to different lengths of the columns)
meta_ev_sigmas    = pd.DataFrame(dict([(k, pd.Series(v)) for (k, v) in meta_ev_sigmas.items()]))
meta_ev_variances = pd.DataFrame(dict([(k, pd.Series(v)) for (k, v) in meta_ev_variances.items()]))
meta_ev_k_values  = pd.DataFrame(dict([(k, pd.Series(v)) for (k, v) in meta_ev_k_values.items()]))


# Add column for the average
meta_ev_sigmas['Average']    = meta_ev_sigmas.mean(axis=1)
meta_ev_variances['Average'] = meta_ev_variances.mean(axis=1)
meta_ev_k_values['Average']  = meta_ev_k_values.mean(axis=1)



key = '20240506163533_9'  # Gauss 9 example
key = '20240507003158_15'  # Gauss 16 example
plt.plot(meta_ev_k_values[key])
plt.hlines(opt_k, xmin=0, xmax=len(meta_ev_k_values[key]), colors='red')
plt.ylabel('spring constant (N/m)')




# Average across runs

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

# Add column for k estimate
meta_ev_sigmas['k of E[sigma]']  = [(0.019235/sigma)**2 for sigma in meta_ev_sigmas['Average']]
meta_ev_variances['k of E[var]'] = [(0.019235**2/var) for var in meta_ev_variances['Average']]
meta_ev_k_values['k of E[k]']    = meta_ev_k_values['Average']

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
#plt.ylim((-5,0))
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

k_estimates = []
iteration = []
for col in cols:
    ser = meta_ev_k_values[col].dropna()
    k_estimates += ser.to_list()
    iteration += ser.index.to_list()
new_dict = {'k estimates': k_estimates, 'iter': iteration}
new_df = pd.DataFrame(new_dict)

for interval in [10, 25, 50, 95]:
    plot = sns.lineplot(data=new_df, x='iter', y='k estimates', estimator=np.median,
                        color="C0", errorbar=('ci', interval))
#plot.set(ylim=(0, max(new_df['k estimates'])))
plot.axhline(y=opt_k)


# # outlier removed for Gauss16
# selected_cols = ['20240507003157_3', '20240507003158_0',
#         '20240507003158_1', '20240507003158_12', '20240507003158_13',
#         '20240507003158_14', '20240507003158_15', '20240507003158_2',
#         '20240507003158_5', '20240507003158_6', '20240507003158_9',
#         '20240507003159_10', '20240507003159_4']

sns.lineplot(data=meta_ev_k_values.loc[:,cols], estimator="median", errorbar=("pi", 50))
sns.lineplot().axhline(y=opt_k)



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








































