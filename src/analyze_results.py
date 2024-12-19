"""

Code to generate the figures for:
    Mascolo & Erill, "Information Theory of Composite Sequence Motifs:
    Mutational and Biophysical Determinants of Complex Molecular Recognition"

"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def unique_filepath(filepath):
    while os.path.exists(filepath):
        warnings.warn('file {} already exists. Appending "x" to ensure unique name'.format(filepath))
        parts = filepath.split('.')
        filepath = '.'.join(parts[:-1]) + 'x.' + parts[-1]
    return filepath

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

def max_possible_dyad_IC(G, motif_len):
    return  2 * max_possible_Rseq(motif_len) + max_possible_Rspacer(G)

def dyad_Rfreq(G, gamma):
    return np.log2(G**2/gamma)

def read_json_file(filename):
    with open(filename) as json_content:
        return json.load(json_content)

# def stacked_barplots(initial_df, drifted_df, parameters, filename='barplots.png'):
    
#     # Read parameters
#     G = parameters['G']
#     gamma = parameters['gamma']
#     motif_len = parameters['motif_len']
    
#     # Data for bars (average IC)
#     x = [1, 2, 3]
#     y1 = np.array([max_possible_Rseq(motif_len), initial_df['Rseq1_targets'].mean(), drifted_df['Rseq1_targets'].mean()])
#     y2 = np.array([max_possible_Rseq(motif_len), initial_df['Rseq2_targets'].mean(), drifted_df['Rseq2_targets'].mean()])
#     y3 = np.array([max_possible_Rspacer(G), initial_df['Rspacer_targets'].mean(), drifted_df['Rspacer_targets'].mean()])
    
#     # Maximum IC and predicted IC
#     maxIC = max_possible_dyad_IC(G, motif_len)
#     Rfreq = dyad_Rfreq(G, gamma)
    
#     # Plot horizontal line for Rfrequency
#     plt.hlines(Rfreq, xmin=min(x)-0.5, xmax=max(x)+0.5, linestyles='dashed', color='r')
    
#     # Plot bars with 95% Confidence Interval (CI)
#     # Note that 95% CI = 1.96 * SEM  [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3487226/]
#     initial_95ci = initial_df['Rtot'].sem() * 1.96
#     drifted_95ci = drifted_df['Rtot'].sem() * 1.96
#     plt.bar(x, y1)
#     plt.bar(x, y2, bottom=y1)
#     plt.bar(x, y3, bottom=y1+y2, yerr=(0, initial_95ci, drifted_95ci), capsize=4)
#     # Labels and legend
#     plt.ylabel("Information (bits)")
#     plt.legend(["Rfreq(dyad)", "Rseq1", "Rseq2", "Rspacer"])
#     plt.title("Dyad information content")
#     plt.ylim((0,maxIC))
#     plt.yticks(list(plt.yticks()[0]) + [Rfreq, maxIC])
#     plt.gca().get_yticklabels()[-2].set_color("red")
#     plt.gca().get_yticklabels()[-1].set_color("red")
#     xlabels = ['Maximum IC','First solution','After drift time']
#     plt.xticks(x, xlabels, rotation='vertical')
    
#     # Save Figure
#     filepath = results_dir + filename
#     plt.savefig(filepath, bbox_inches="tight", dpi=600)
#     plt.close()

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

def process_data(resultsdir, sample_size=None, ic_correction='true_Rseq'):
    
    # Check for empty/incomplete subfolders
    for f in os.listdir(resultsdir):
        # Skip files if present
        if not os.path.isdir(resultsdir + f):
            print('Skipped (not a folder):  ' + resultsdir + f)
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

def study_spacer_stacked_barplot(drifted_df_list, parameters, sample_size=None,
                                     results_dir='', spacer_weep=False, labelsfontsize=10):
    '''
    Saves a stacked barplot for the Spacer-Study as a PNG file.
      
      - `drifted_df_list`: list of pandas dataframes (with the 'drifted' values)
      - `parameters`: dictionary from the JSON parameters file used in the Spacer-Study
      - `sample_size`: (optional) string to set an informative file name
      - `results_dir`: (optional) string to specify a path where file should be saved
      - `spacer_weep`: (optional) string to set meaningful x-tick labels
      - `labelsfontsize`: (optional) int to set font size for labels
    
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
        # Note that 95% CI = 1.96 * SEM
        # see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3487226/
    
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
    maxIC = max_possible_dyad_IC(G, motif_len)
    Rfreq = dyad_Rfreq(G, gamma)
    
    # Make plot
    
    # Plot horizontal line for Rfrequency
    plt.hlines(Rfreq, xmin=min(x)-0.5, xmax=max(x)+0.5, linestyles='dashed', color='r')
    # Plot stacked bar
    plt.bar(x, y1)
    plt.bar(x, y2, bottom=y1)
    plt.bar(x, y3, bottom=y1+y2, yerr=([0] + Rtot_CI_vals), capsize=4)
    
    # Labels and legend
    plt.ylabel("Information (bits)", fontsize=labelsfontsize)
    plt.legend([r'$R_{frequency}$', r'$R_{spacer}$', r'$R_{sequence(1)}$', r'$R_{sequence(2)}$'],
               fontsize=labelsfontsize)
    plt.ylim((0,maxIC))
    yticks_list = list(plt.yticks()[0])
    yticks_list = yticks_list[:-1]  # Remove highest tick (it will be replaced by maxIC)
    plt.yticks(yticks_list + [Rfreq, maxIC], fontsize=8)
    plt.gca().get_yticklabels()[-2].set_color("red")
    plt.gca().get_yticklabels()[-1].set_color("red")
    
    # x-axis ticks
    if spacer_weep:
        xlabels = ['Maximum IC'] + [r'$R_{spacer}$ = ' + str(int(s)) + ' bits' for s in y1[1:]]
    else:
        xlabels = ['Maximum IC'] + ['Evolved IC']*len(drifted_df_list)
    plt.xticks(x, xlabels, rotation=45, ha='right', fontsize=labelsfontsize)
    
    # Save Figure
    if sample_size:
        filename = 'Study_Spacer_Barplot_' + str(sample_size) + '.png'
    else:
        filename = 'Study_Spacer_Barplot_ALL.png'
    figure_filepath = results_dir + filename
    figure_filepath = unique_filepath(figure_filepath)
    plt.savefig(figure_filepath, bbox_inches="tight", dpi=600)
    plt.close()
    print('Plot saved to: ' + figure_filepath)

def plot_Rsequence_Ev(results_path, labelsfontsize=10):
    '''
    Saves a line plot as a PNG, showing the evolution of Rsequence in runs
    where n=1. It can be used to reproduce Figure 2 from (Schneider 2000).
    '''
    parameters = read_json_file(results_path + '/parameters.json')
    update_period = parameters['update_period']
    Rfrequency = - np.log2(parameters['gamma'] / parameters['G'])
    
    # Rsequence over generations is stored as a CSV table
    table_filepath = results_path + '/Rsequence_evolution_data.csv'
    
    if os.path.exists(table_filepath):
        # Data is already prepared as a table.
        df = pd.read_csv(table_filepath)
        gen_list, Rseq_list = df['Generation'].to_list(), df['Rsequence'].to_list()
    
    else:
        # Data has not been processed yet. Make a table.
        gen_list = []
        Rseq_list = []
        gen = 0
        ic_report_path = '{}/ev_gen_{}_ic_report.csv'.format(results_path, gen)
        while os.path.exists(ic_report_path):
            df = parse_report(ic_report_path)
            df.index = df['IC Correction'].to_list()
            Rseq_list.append(df.loc['true_Rseq','Rseq_targets'])
            gen_list.append(gen)
            gen += update_period
            ic_report_path = '{}/ev_gen_{}_ic_report.csv'.format(results_path, gen)    
        # Save data as CSV table
        df = pd.DataFrame({'Generation': gen_list, 'Rsequence': Rseq_list})
        df.to_csv(table_filepath, index=False)
        print('CSV file saved here: ' + table_filepath)
    
    # Make plot
    plt.figure(figsize=(12, 4.8))
    # Horizontal line for Rfrequency
    plt.hlines(Rfrequency, 0, len(Rseq_list),color='red', linestyle='dashed',
               label=r'$R_{frequency}$', zorder=2)
    # Line plot for the evolving population
    plt.plot(gen_list, Rseq_list, label=r'$R_{sequence}$', zorder=1)
    # Legend
    plt.legend(fontsize=labelsfontsize)
    # x-axis
    pad = max(gen_list)/50
    plt.xlim((-pad, max(gen_list)+pad))
    plt.xlabel('Generation', fontsize=labelsfontsize)
    # y-axis
    max_ic = parameters['motif_len'] * 2
    y_axis_lower_bound = min(-1, min(Rseq_list))
    plt.ylim((y_axis_lower_bound,max_ic))
    plt.ylabel('Information (bits)', fontsize=labelsfontsize)
    # Set y-ticks
    yticks_list = list(plt.yticks()[0])
    yticks_list = yticks_list[:-1]  # Remove highest tick (it will be replaced by maxIC)
    plt.yticks(yticks_list + [Rfrequency, max_ic])
    plt.gca().get_yticklabels()[-2].set_color("red")
    plt.gca().get_yticklabels()[-1].set_color("red")
    plt.ylim((y_axis_lower_bound,max_ic))
    
    # Save line plot as PNG file
    figure_filepath = results_path + '/Rsequence_evolution.png'
    figure_filepath = unique_filepath(figure_filepath)
    plt.savefig(figure_filepath, bbox_inches="tight", dpi=600)
    plt.close()
    print('Plot saved to: ' + figure_filepath)

def map_dyads_to_2D_plot(drifted_df_list, parameters, sample_size=None, results_dir='', labelsfontsize=10):
    '''
    Makes the figure that maps evolved organisms to the 2D solution space
    (x-axis: Rspacer, y-axis: Rsequence)
    '''
    if sample_size is None:
        sample_size = len(drifted_df_list)
    
    # Parameters
    gamma = parameters['gamma']
    G = parameters['G']
    n = parameters['motif_n']
    
    # Set bounds
    #   * Rspacer bounds
    # OVERALL LOWER BOUND: the highest between the functional lower bound and
    # the informational lower bound
    min_Rspacer = max(np.log2(gamma), np.log2(G/gamma))
    # Upper bound is always log2(G)
    max_Rspacer = np.log2(G)
    #   * Rsequence bounds
    min_Rsequence = 0
    max_Rsequence = n * np.log2(G/gamma)
    
    # Plot structure
    margin = 0.5
    plt.xlim((min_Rsequence, max_Rspacer + margin))
    plt.ylim((min_Rsequence, max_Rsequence + margin))
    # Vertical lines for Rspacer bounds
    plt.vlines(min_Rspacer, ymin=0, ymax=max_Rsequence + margin, colors='grey', linestyles='dashed', label=r' Min $R_{spacer}$')
    plt.vlines(max_Rspacer, ymin=0, ymax=max_Rsequence + margin, colors='black', linestyles='dotted', label=r' Max $R_{spacer}$')
    
    # Theoretical solutions space
    x1, x2 = min_Rspacer, max_Rspacer
    y1, y2 = np.log2(G**n / gamma) - x1, np.log2(G**n / gamma) - x2
    plt.plot([x1, x2], [y1, y2], color='red', label='Solution space')
    
    # Datapoints
    Rseq_list = [df['Rseq1_targets'].mean() + df['Rseq2_targets'].mean() for df in drifted_df_list[:sample_size]]
    Rspacer_list = [df.loc[0,'Rspacer_targets'] for df in drifted_df_list]
    plt.scatter(Rspacer_list, Rseq_list, label='Evolved')
    plt.xlabel(r'$R_{spacer}$ (bits)', fontsize=labelsfontsize)
    plt.ylabel(r'$R_{sequence}$ (bits)', fontsize=labelsfontsize)
    
    # Grid and ticks
    plt.xticks(list(range(int(np.ceil(max_Rspacer))   + 1)))
    plt.yticks(list(range(int(np.ceil(max_Rsequence)) + 1)))
    plt.gca().set_aspect('equal')
    plt.grid(alpha=0.5)
    #plt.legend(framealpha=1, loc='upper left')
    plt.legend(fontsize=labelsfontsize)
    
    # Save plot
    figure_filepath = results_dir + '2D_plot_with_sim_data.png'
    figure_filepath = unique_filepath(figure_filepath)
    plt.savefig(figure_filepath, bbox_inches='tight', dpi=600)
    plt.close()
    print('Plot saved to: ' + figure_filepath)


# def make_df_for_avg_spring_const_plot_OLD(meta_ev_sigmas, meta_ev_k_values, update_period):
#     cols = list(meta_ev_sigmas.columns)[:-2]
    
#     k_estimates = []
#     iteration = []
#     for col in cols:
#         ser = meta_ev_k_values[col].dropna()
#         k_estimates += ser.to_list()
#         iters = ser.index.to_list()
        
#         iteration += [it * update_period for it in iters]
#     new_dict = {'k estimates': k_estimates, 'iter': iteration}
#     new_df = pd.DataFrame(new_dict)
#     return new_df


# def make_df_for_avg_spring_const_plot_NEW(meta_ev_k_values, update_period):
#     cols = list(meta_ev_k_values.columns)[:-2]
    
#     k_estimates = []
#     iteration = []
#     for col in cols:
#         ser = meta_ev_k_values[col].dropna()
#         k_estimates += ser.to_list()
#         iters = ser.index.to_list()
        
#         iteration += [it * update_period for it in iters]
#     return pd.DataFrame({'k estimates': k_estimates, 'iter': iteration})


def study_stiffness_evolution(path, max_n_runs=None):
    '''
    XXX Docstring here ...
    '''
    meta_ev_sigmas = dict()
    meta_ev_variances = dict()
    meta_ev_k_values = dict()
    
    # Collect the runs (replicas)
    if max_n_runs:
        runs = os.listdir(path)[:max_n_runs]
    else:
        runs = os.listdir(path)
    
    # For loop over all the runs (replicas)
    for run in runs:
        fldr_content = os.listdir(path + '/' + run)
        org_f = [f for f in fldr_content if (f.startswith('ev_') and f.endswith('org.json'))]
        org_f = sort_filenames_by_gen(org_f)
        
        ev_sigmas = []
        ev_variances = []
        ev_k_values = []
        for f in org_f:
            org = read_json_file(path + '/' + run + '/' + f)
            ev_sigmas.append(org['sigma'])
            ev_variances.append(org['sigma']**2)
            ev_k_values.append((0.019235/org['sigma'])**2)
        
        # Compile dictionaries
        meta_ev_sigmas[run] = ev_sigmas
        meta_ev_variances[run] = ev_variances
        meta_ev_k_values[run] = ev_k_values
    
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

def prepare_spring_const_data(path, thresh, max_n_runs=None):
    '''
    Prepares data for the `spring_const_plot` function.
    
    Input args
        path: path to the folder that contains the runs (replicas) subfolders.
        thresh: minimum number of (non-missing) values.
        max_n_runs: (optional) can be set to use only up to a certain number of replicas.
    
    Returns
        df: data (dataframe)
        opt_k: estimated (predicted) optimal value of kappa
        params: paramters that were used to generate the data (dictionary)
    '''
    
    # Read parameters
    for x in os.listdir(path):
        if os.path.isdir(path + '/' + x):
            params = read_json_file(path + '/' + x + '/parameters.json')
            break
    
    meta_ev_sigmas, meta_ev_variances, meta_ev_k_values, n_gen, n_gen_max = study_stiffness_evolution(path, max_n_runs)
    meta_ev_sigmas = meta_ev_sigmas.dropna(thresh=thresh)
    meta_ev_k_values = meta_ev_k_values.dropna(thresh=thresh)
    opt_k = (0.019235/np.std(params['spacers']))**2
    
    update_period = params['update_period']
    
    # Make dataframe
    # Dataframe has two columns
    k_estimates = []
    iteration = []
    # Compile 
    for col in list(meta_ev_k_values.columns)[:-2]:
        ser = meta_ev_k_values[col].dropna()
        k_estimates += ser.to_list()
        iters = ser.index.to_list()
        
        iteration += [it * update_period for it in iters]
    df = pd.DataFrame({'k estimates': k_estimates, 'iter': iteration})
    return df, opt_k, params

def spring_const_plot(df_1, df_2, opt_k_1, opt_k_2, results_dir='', tag=None):
    '''
    Makes the figure for the evolution of the spring constant.
    '''
    
    yaxis_scale = 'linear'
    
    # CASE 1
    #   (*) mid 50% of the data
    plot = sns.lineplot(data=df_1, x='iter', y='k estimates', estimator=np.median,
                 color="C0", errorbar=('ci', 50), label=r'Simulations using $D_{1}$', zorder=3)
    #   (*) mid 75% of the data
    plot = sns.lineplot(data=df_1, x='iter', y='k estimates', estimator=np.median,
                 color="C0", errorbar=('ci', 75), zorder=4)
    plt.yscale(yaxis_scale)
    plot.axhline(y=opt_k_1, linestyle='dashed', color='C0', label=r'$\kappa_{opt}^{(1)}$', zorder=1)
    
    # CASE 2
    #   (*) mid 50% of the data
    plot = sns.lineplot(data=df_2, x='iter', y='k estimates', estimator=np.median,
                 color="C1", errorbar=('ci', 50), label=r'Simulations using $D_{2}$', zorder=5)
    #   (*) mid 75% of the data
    plot = sns.lineplot(data=df_2, x='iter', y='k estimates', estimator=np.median,
                 color="C1", errorbar=('ci', 75), zorder=6)
    plt.yscale(yaxis_scale)
    plot.axhline(y=opt_k_2, linestyle='dashed', color='C1', label=r'$\kappa_{opt}^{(2)}$', zorder=2)
    
    # Labels and bounds
    plot.set(ylim=(-0.0005, 0.005))
    plt.ylabel(r'$\kappa$ (N/m)')
    plt.xlabel('Generation')
    plt.legend()
    
    # Save plot
    if tag:
        figure_filename = 'Spring_Constant_Evo_Figure_max{}runs.png'.format(tag)
    else:
        figure_filename = 'Spring_Constant_Evo_Figure.png'
    figure_filepath = results_dir + figure_filename
    figure_filepath = unique_filepath(figure_filepath)
    plt.savefig(figure_filepath, dpi=600)
    plt.close()
    print('Plot saved to: ' + figure_filepath)

def plot_D_histogram(x_axis, y, results_dir='', tag='', color='#1f77b4', y_max=None, labels_fontsize=42, ticks_fontsize=28):
    '''
    Function used to generate the little histograms for spacer size distribution.
    '''
    if y_max is None:
        y_max = max(y)
    
    if tag == '':
        title = 'D'
    else:
        title = r'$D_{}$'.format(tag)
    
    plt.bar(x_axis, y, width=1, color=color, edgecolor='black')
    plt.title(title, fontsize=labels_fontsize)
    plt.xlabel('Spacer size (bp)', fontsize=labels_fontsize)
    plt.ylabel('# targets', fontsize=labels_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.ylim((0,y_max+0.5))
    figure_filepath = results_dir + 'Hist_for_D{}.png'.format(tag)
    figure_filepath = unique_filepath(figure_filepath)
    plt.savefig(figure_filepath, bbox_inches="tight", dpi=300)
    plt.close()
    print('D histogram saved as: ' + figure_filepath)

def plot_D1_D2_histograms(params_1, params_2, results_dir=''):
    '''
    Generates the spacer size distribution histograms for the figure for the
    evolution of the spring constant.
    '''
    spacers_1 = params_1['spacers']
    spacers_1_unique = list(set(spacers_1))
    spacers_1_unique.sort()
    
    spacers_2 = params_2['spacers']
    spacers_2_unique = list(set(spacers_2))
    spacers_2_unique.sort()
    
    x_min = min(min(spacers_1_unique), min(spacers_2_unique))
    x_max = max(max(spacers_1_unique), max(spacers_2_unique))
    x_axis = list(range(x_min, x_max+1))
    
    y_1 = []
    y_2 = []
    for spacer_size in x_axis:
        y_1.append(spacers_1.count(spacer_size))
        y_2.append(spacers_2.count(spacer_size))
    
    y_max = max(max(y_1), max(y_2))
    
    color1 = '#1f77b4'
    color2 = '#ff7f0e'
    
    labels_fontsize = 42
    ticks_fontsize = 28
    
    plot_D_histogram(x_axis, y_1, results_dir, '1', color1, y_max, labels_fontsize, ticks_fontsize)
    plot_D_histogram(x_axis, y_2, results_dir, '2', color2, y_max, labels_fontsize, ticks_fontsize)

def plot_comp_exp(experiment_dirpath, variable_spcr='flex'):
    '''
    Makes the figure showing the results of the competition experiments.
    '''
    percentages = [0, 25, 50, 75, 100]
    fig, axs = plt.subplots(len(percentages), sharex=True, layout="constrained", figsize=(6, 7))
    fig.supxlabel('generation')
    fig.supylabel('frequency')
    #fig.suptitle('Indel size: {} bp'.format(INDEL_SIZE), fontsize=15)
    
    # the name of the experiment is the same as the name of the directory
    experiment = os.path.basename(os.path.normpath(experiment_dirpath))
    
    ###experiment_dirpath = '../results/' + experiment + '/'
    
    for i in range(len(percentages)):
        
        # Percentages for this subplot
        pc_indel = percentages[i]
        pc_subst = 100 - pc_indel
        
        # Read data for this subplot
        subplot_dirname = 'comp_exp_{}_{}pc/'.format(experiment, pc_indel)
        subplot_dirpath = os.path.join(experiment_dirpath, subplot_dirname)
        ###base_dir_path = experiment_dirpath + 'comp_exp_{}_{}pc/'.format(experiment, pc_indel)
        dataframes = []
        for run in os.listdir(subplot_dirpath):
            df = pd.read_csv(subplot_dirpath + run + '/competition_report.csv').loc[:,[variable_spcr,'stiff']]
            # Remove rows after fixation
            fixation_gen = None
            for row_idx in range(len(df)):
                if df.loc[row_idx,'stiff'] in [1,0]:
                    fixation_gen = row_idx
                    break
            if fixation_gen:
                df_cut = df.loc[:row_idx+1,:]
                dataframes.append(df_cut)
        
        # Plot lines in this subplot
        for j, df in enumerate(dataframes):
            # Avoid repeating labels in the legend
            if i==0 and j==0:
                stiff_label, flex_label = 'Non-variable spacer', 'Variable spacer'
            else:
                stiff_label, flex_label = '', ''
            axs[i].plot(df['stiff'], c='C1',label=stiff_label)
            axs[i].plot(df[variable_spcr], c='C0', linestyle='dashed', label=flex_label)
            axs[i].margins(x=0)
            axs[i].margins(y=0)
            axs[i].set_yticks((0,0.5,1))
            # axs[i].set_ylim((0,1))
            # axs[i].set_xlim((0,120))
            # if i == 2:
            #     axs[i].set_ylabel('frequency')
            # if i == 4:
            #     axs[i].set_xlabel('generation')
        axs[i].title.set_text('{}% substitutions, {}% indels'.format(pc_subst, pc_indel))
        #plt.legend()
    
    # plt.figlegend(bbox_to_anchor=(1.05,0.5))
    fig.legend(loc="outside upper right", ncol=1)
    #fig.tight_layout()
    figure_filename = 'full_competition_experiment_{}.png'.format(experiment)
    ### figure_filepath = experiment_dirpath + figure_filename
    figure_filepath = os.path.join(experiment_dirpath, figure_filename)
    figure_filepath = unique_filepath(figure_filepath)
    fig.savefig(figure_filepath, dpi=300)
    plt.close()
    print('Plot saved to: ' + figure_filepath)


# ================================================
# Plot for n=1 case, to reproduce (Schneider 2000)
# ================================================

# Reproduce (Schneider 2000)
results_path = '../results/Reproduce_Schneider_2000'
plot_Rsequence_Ev(results_path, labelsfontsize=14)


# ===========================================
# Plots for n=2 case, to study dyad evolution
# ===========================================

# results_dir = '../../RESULTS/Study_Spacer_New_Repr_Renamed3/'
results_dir = '../results/Test_Rsequence_plus_Rspacer/'
sample_size = 20

# Prepare data
drifted_df_list = []
for subfolder in os.listdir(results_dir):
    if not os.path.isdir(results_dir + subfolder):
        continue
    initial_df, drifted_df, all_ev, parameters = process_data(results_dir + subfolder + '/', sample_size)
    drifted_df_list.append(drifted_df)

# Stacked Barplot
study_spacer_stacked_barplot(drifted_df_list, parameters, sample_size, results_dir, spacer_weep=True, labelsfontsize=10)

# 2D plot
map_dyads_to_2D_plot(drifted_df_list, parameters, sample_size, results_dir, labelsfontsize=10)

# ====================
# Supplementary Figure
# ====================

# results_dir = '../../RESULTS/AUPRC_fitness/'
results_dir = '../results/Test_Rsequence_plus_Rspacer_Gaussian_Conn/'

# (A) PANEL
subfolder = 'Gauss_9sites'
sample_size = None

# Prepare data
initial_df, drifted_df, all_ev, parameters = process_data(results_dir + subfolder + '/', sample_size)

# Stacked Barplot
study_spacer_stacked_barplot([drifted_df], parameters, sample_size, results_dir + subfolder + '/')

# Plot spacer distribution as histogram
spacers = parameters['spacers']
spacers_unique = list(set(spacers))
spacers_unique.sort()
counts = [spacers.count(d) for d in spacers_unique]
plot_D_histogram(spacers_unique, counts, results_dir + subfolder + '/', y_max=6)

# (B) PANEL
subfolder = 'Gauss_16sites'
sample_size = None

# Prepare data
initial_df, drifted_df, all_ev, parameters = process_data(results_dir + subfolder + '/', sample_size)

# Stacked Barplot
study_spacer_stacked_barplot([drifted_df], parameters, sample_size, results_dir + subfolder + '/')

# Plot spacer distribution as histogram
spacers = parameters['spacers']
spacers_unique = list(set(spacers))
spacers_unique.sort()
counts = [spacers.count(d) for d in spacers_unique]
plot_D_histogram(spacers_unique, counts, results_dir + subfolder + '/', y_max=6)


# ===================================
# Figure on spring constant evolution
# ===================================

# results_dir = '../../RESULTS/For_New_Spring_Const_Evo/'
results_dir = '../results/Spring_Constant_Evo/'
sample_size = None

# Two experiments
exp_1 = 'seven_vals_histogram'
exp_2 = 'three_vals_histogram'

# MAX_N_RUNS = None
MAX_N_RUNS = 25

# THRSH: Missing value tolerance (some runs stopped earlier and don't have
# values for late generations)
THRSH = 10

df_1, opt_k_1, params_1 = prepare_spring_const_data(results_dir + exp_1, THRSH, MAX_N_RUNS)
df_2, opt_k_2, params_2 = prepare_spring_const_data(results_dir + exp_2, THRSH, MAX_N_RUNS)

# Trim dataframes
stop_gen = min(df_2['iter'].max(), df_1['iter'].max())
df_1 = df_1.loc[df_1['iter'] <= stop_gen]
df_2 = df_2.loc[df_2['iter'] <= stop_gen]

# Make line plots with shaded areas
spring_const_plot(df_1, df_2, opt_k_1, opt_k_2, results_dir, MAX_N_RUNS)

# Plot the histograms for the two spacer size distributions (D1 and D2)
plot_D1_D2_histograms(params_1, params_2, results_dir)


# ==================================
# Figure for competition experiments
# ==================================

# Figure for paper: Stiff VS Flexible
experiment_dirpath = '../results/Competition_Experiments/Stiff_VS_Flex_indel_1bp/'
plot_comp_exp(experiment_dirpath)

# Extra Figure (not in paper): Stiff VS Medium
experiment_dirpath = '../results/Competition_Experiments/Stiff_VS_Med_indel_1bp/'
plot_comp_exp(experiment_dirpath, variable_spcr='medium')
















