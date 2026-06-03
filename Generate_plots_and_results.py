import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import rdtools
import seaborn as sns
import datetime
import statsmodels.api as sm
from scipy.stats import linregress, bootstrap
from sklearn.linear_model import RANSACRegressor

matplotlib.rcParams['font.family'] = 'arial'

plt.rc('font', size=7)


def weighted_median(values, weights):
    """
    Compute the weighted median of a set of values.
    
    Parameters:
        values (array-like): Numeric values.
        weights (array-like): Corresponding weights (non-negative).
    
    Returns:
        float: Weighted median.
    """
    # Convert to numpy arrays
    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)

    # Validate inputs
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length.")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    if np.sum(weights) == 0:
        raise ValueError("Sum of weights must be greater than zero.")

    # Sort by values
    sorted_indices = np.argsort(values)
    values_sorted = values[sorted_indices]
    weights_sorted = weights[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = np.cumsum(weights_sorted)
    cutoff = np.sum(weights_sorted) / 2.0

    # Find the weighted median
    median_index = np.where(cumulative_weights >= cutoff)[0][0]
    return values_sorted[median_index]


def estimate_PLR_from_distribution(data, print_info=False, method='mean', uncertainty_weighted=False):
    # Find mean value, which is the most reliable estimate of real PLR
    # First, filter outliers with interquartile ranges
    Q1 = np.percentile(data['PLR'], 25, method = 'midpoint') 
    Q3 = np.percentile(data['PLR'], 75, method = 'midpoint')
    IQR = Q3 - Q1
    upper = np.where(data['PLR'] >= (Q3+1.5*IQR))
    lower = np.where(data['PLR'] <= (Q1-1.5*IQR))
    PLRs_wo_outl = data.copy()
    PLRs_wo_outl.drop(upper[0], inplace=True)
    PLRs_wo_outl.drop(lower[0], inplace=True)

    #print(len(PLRs_wo_outl))
    if method == 'mean':
        PLR_avge = PLRs_wo_outl.PLR.mean()

    elif method == 'median':
        if uncertainty_weighted:
            PLR_avge = weighted_median(PLRs_wo_outl.PLR, 1/PLRs_wo_outl.uncertainty)
        else:
            PLR_avge = PLRs_wo_outl.PLR.median()

    res = bootstrap((PLRs_wo_outl.PLR,), np.median, confidence_level=0.95)

        
    if print_info:
        print('Number of points without outliers:', len(PLRs_wo_outl))
        print('Average PLR:', PLR_avge)
        print('Normal standard error:', PLRs_wo_outl.PLR.std()/(np.sqrt(len(PLRs_wo_outl.PLR))))
        print('Normal 95% CI:', 1.96*PLRs_wo_outl.PLR.std()/(np.sqrt(len(PLRs_wo_outl.PLR))))
        print('Normal standard deviation:', PLRs_wo_outl.PLR.std())
        print('Bootstrap standard error:', res.standard_error)
        print('Bootstrap 95% CI:', (res.confidence_interval.high-res.confidence_interval.low)/2, res.confidence_interval)
    return PLR_avge, PLRs_wo_outl


def PLR_density_plot(file_path, xlims=None, bin_width = 0.075, ax=None, method_conditions=None):
    """
    
    """
    df = pd.read_csv(file_path)
    df['PLR'] = df.PLR*100
    if method_conditions is not None:
        for key, value in method_conditions.items():
            df = df[df[key] == value]

    data = df.copy()
    data.reset_index(inplace=True)
    print(len(data))
    print(f'Max: {max(data.PLR)}, min: {min(data.PLR)}')

    PLR_avge, PLRs_wo_outl = estimate_PLR_from_distribution(data, print_info=True)

    # Plot the figure
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(1063/300, 500/300), dpi=300)
    bin_count_2 = int((PLRs_wo_outl.PLR.max()-PLRs_wo_outl.PLR.min())/bin_width)

    try:
        ax.hist(PLRs_wo_outl.PLR, bins=bin_count_2)
    except ValueError:
        print('Try reducing the bin_width.')
        return 
    ax.vlines([PLR_avge],0,ax.get_ylim()[1], color='red', linewidth=1.2, label=r'$\overline{PLR}=$'+f'{PLR_avge:.2f} %/year')
    ax.legend()
    ax.set_xlabel('PLR (%/year)')
    ax.set_ylabel('Count (-)')

    if xlims is not None:
        ax.set_xlim(xlims)

    plt.tight_layout()

    if ax is None:
        plt.show()
    return ax


def PLR_density_all_in_same(file_paths, xlims=None, bin_width = 0.075, ax=None, method='median'):
    """
    
    """
    df = pd.read_csv(file_paths[0])
    df_post_filter = pd.read_csv(file_paths[1])

    df['PLR'] = df.PLR*100
    df = df[df.POA_l > 0]  # Nighttimes are filtered in all datasets, so POA LL of 0 and 5 are redundant
    df_post_filter['PLR'] = df_post_filter.PLR*100
    df_post_filter = df_post_filter[df_post_filter.POA_l > 0]

    df_no_outlier = df[df.PvsG == 0].copy() 
    df_outlier = df[df.PvsG == 1].copy()
    df_post_filter_no_outlier = df_post_filter[df_post_filter.PvsG == 0].copy()
    df_post_filter_outlier = df_post_filter[df_post_filter.PvsG == 1].copy()


    PLR_avges = []
    plr_data = pd.DataFrame()
    columns = ['No threshold', 'Threshold', 'Outlier + No threshold',  'Outlier + threshold']
    
    for d, c in zip([df, df_post_filter, df_outlier, df_post_filter_outlier], columns):
        d.reset_index(inplace=True)
        #print(len(d))
        #print(f'Max: {max(d.PLR)}, min: {min(d.PLR)}')
        PLR_avge, PLRs_wo_outl = estimate_PLR_from_distribution(d, print_info=True, method='median')
        PLR_avges.append(PLR_avge)
        plr_data[c] = PLRs_wo_outl.PLR
        print(c, f': {len(PLRs_wo_outl.PLR)} datapoints\n')

    # Plot the figure
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(1063/300, 500/300), dpi=300)
    #bin_count_2 = int((PLRs_wo_outl.PLR.max()-PLRs_wo_outl.PLR.min())/bin_width)
    sns.histplot(ax = ax,
             data=plr_data,
              bins=50,
              kde=True,
              fill=True)
    
    y_max = ax.get_ylim()[1]
    ax.vlines([PLR_avges[0]],0, y_max, color='#1f77b4', linestyle='dashed', linewidth=1.2, label=r'No threshold: $\overline{PLR}=$'+f'{PLR_avges[0]:.2f} %/year')
    ax.vlines([PLR_avges[1]],0, y_max, color='#ff7f0e', linestyle='dashed', linewidth=1.2, label=r'Threshold: $\overline{PLR}=$'+f'{PLR_avges[1]:.2f} %/year')
    ax.vlines([PLR_avges[2]],0, y_max, color='#2ca02c', linestyle='dashed', linewidth=1.2, label=r'Outlier + no threshold: $\overline{PLR}=$'+f'{PLR_avges[2]:.2f} %/year')
    ax.vlines([PLR_avges[3]],0, y_max, color='#d62728', linestyle='dashed', linewidth=1.2, label=r'Outlier + threshold: $\overline{PLR}=$'+f'{PLR_avges[3]:.2f} %/year')
    ax.legend()
    ax.set_xlabel('PLR (%/year)')
    ax.set_ylabel('Count (-)')

    if xlims is not None:
        ax.set_xlim(xlims)

    plt.tight_layout()

    if ax is None:
        plt.show()
    return ax


def PLR_density_all_in_same_no_bars(file_paths, xlims=None, bin_width = 0.075, ax=None, method='median', uncertainty_weighted=False):
    """
    
    """
    df = pd.read_csv(file_paths[0])
    df_post_filter = pd.read_csv(file_paths[1])

    df['PLR'] = df.PLR*100
    df = df[df.POA_l > 0]  # Nighttimes are filtered in all datasets, so POA LL of 0 and 5 are redundant
    df_post_filter['PLR'] = df_post_filter.PLR*100
    df_post_filter = df_post_filter[df_post_filter.POA_l > 0]

    df_no_outlier = df[df.PvsG == 0].copy() 
    df_outlier = df[df.PvsG == 1].copy()
    df_post_filter_no_outlier = df_post_filter[df_post_filter.PvsG == 0].copy()
    df_post_filter_outlier = df_post_filter[df_post_filter.PvsG == 1].copy()


    PLR_avges = []
    plr_data = pd.DataFrame()
    columns = ['No threshold', 'Threshold', 'Outlier + No threshold',  'Outlier + threshold']
    
    for d, c in zip([df, df_post_filter, df_outlier, df_post_filter_outlier], columns):
        d.reset_index(inplace=True)
        #print(len(d))
        #print(f'Max: {max(d.PLR)}, min: {min(d.PLR)}')
        PLR_avge, PLRs_wo_outl = estimate_PLR_from_distribution(d, print_info=True, method='median', uncertainty_weighted=uncertainty_weighted)
        PLR_avges.append(PLR_avge)
        plr_data[c] = PLRs_wo_outl.PLR
        print(c, f': {len(PLRs_wo_outl.PLR)} datapoints\n')

    # Plot the figure
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(1063/300, 500/300), dpi=300)
    #bin_count_2 = int((PLRs_wo_outl.PLR.max()-PLRs_wo_outl.PLR.min())/bin_width)

    sns.kdeplot(ax = ax,
                data=plr_data,
                fill=False,
                common_norm=False,
                linewidth=1
                )
    
    y_max = ax.get_ylim()[1]
    ax.vlines([PLR_avges[0]],0, y_max, color='#1f77b4', linestyle='dashed', linewidth=1.2, label=r'No threshold: $\overline{PLR}=$'+f'{PLR_avges[0]:.2f} %/year')
    ax.vlines([PLR_avges[1]],0, y_max, color='#ff7f0e', linestyle='dashed', linewidth=1.2, label=r'Threshold: $\overline{PLR}=$'+f'{PLR_avges[1]:.2f} %/year')
    ax.vlines([PLR_avges[2]],0, y_max, color='#2ca02c', linestyle='dashed', linewidth=1.2, label=r'Outlier + no threshold: $\overline{PLR}=$'+f'{PLR_avges[2]:.2f} %/year')
    ax.vlines([PLR_avges[3]],0, y_max, color='#d62728', linestyle='dashed', linewidth=1.2, label=r'Outlier + threshold: $\overline{PLR}=$'+f'{PLR_avges[3]:.2f} %/year')
    ax.legend()
    ax.set_xlabel('PLR (%/year)')
    ax.set_ylabel('Density (-)')

    if xlims is not None:
        ax.set_xlim(xlims)

    plt.tight_layout()

    if ax is None:
        plt.show()
    return ax


def read_datasets(folder, datetime_name='utctime'):
    """
    Inputs:
        folder: (str) Path to performance metric data folder (e.g., '.../helsinki_performance_metrics')
    Returs:
        dfs: a list of filtered performance metric datasets
        dfs_time_series_plot: a dictionary containing datasets for the performance metric times series plot
    """

    P_nofilt = pd.read_csv(os.path.join(folder, 'ME', 'POA0_P_ME.csv')) # Monthly power
    P_PvsPOA_POA50 = pd.read_csv(os.path.join(folder, 'ME', 'POA50_P_ME.csv'))
    P_PvsPOA_POA0 = pd.read_csv(os.path.join(folder, 'ME', 'POA0_P_ME.csv'))
    #P_PvsPOA_CS_POA50 = pd.read_csv(os.path.join(folder, 'ME', 'CS20_POA50_P_ME.csv'))
    insta_P_PvsPOA_POA50 = pd.read_csv(os.path.join(folder, 'insta', 'POA50_P_insta.csv'))
    insta_P_nofilt = pd.read_csv(os.path.join(folder, 'insta', 'POA0_P_insta.csv'))

    PR_nofilt = pd.read_csv(os.path.join(folder, 'ME', 'POA0_PR_ME.csv')) # Monthly performance ratio
    PR_PvsPOA_POA50 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA50_PR_ME.csv')) # Monthly filtered performance ratio
    PR_PvsPOA_POA0 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA0_PR_ME.csv'))
    #PR_PvsPOA_CS_POA50 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_CS20_POA50_PR_ME.csv'))

    PRt_PvsPOA_POA50 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA50_PRt_ME.csv')) # Monthly temperature corrected, filtered PR
    PRt_PvsPOA_POA0 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA0_PRt_ME.csv'))
    #PRt_PvsPOA_CS_POA50 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_CS20_POA50_PRt_ME.csv'))


    PI_nofilt = pd.read_csv(os.path.join(folder, 'ME', 'POA0_PI_ME.csv')) # Monthly PI
    PI_PvsPOA_POA50 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA50_PI_ME.csv')) # Monthly filtered PI
    PI_PvsPOA_POA0 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA0_PI_ME.csv'))
    #PI_PvsPOA_CS_POA50 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_CS20_POA50_PI_ME.csv'))
    insta_PI_PvsPOA_POA50 = pd.read_csv(os.path.join(folder, 'insta', 'PvsPOA_POA50_PI_insta.csv'))

    PVUSA_nofilt = pd.read_csv(os.path.join(folder, 'ME', 'POA0_PVUSA_ME.csv'))
    PVUSA_PvsPOA_POA50 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA50_PVUSA_ME.csv'))
    PVUSA_PvsPOA_POA0 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA0_PVUSA_ME.csv'))
    PVUSA_PvsPOA_POA500 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA500_PVUSA_ME.csv'))
    PVUSA_PvsPOA_POA200 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA200_PVUSA_ME.csv'))

    huld_nofilt = pd.read_csv(os.path.join(folder, 'ME', 'POA0_huld_ME.csv'))
    huld_PvsPOA_POA50 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA50_huld_ME.csv'))
    huld_PvsPOA_POA0 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_POA0_huld_ME.csv'))
    #huld_PvsPOA_CS_POA50 = pd.read_csv(os.path.join(folder, 'ME', 'PvsPOA_CS20_POA50_huld_ME.csv'))


    d_P_nofilt = pd.read_csv(os.path.join(folder, 'D', 'POA0_P_D.csv'), parse_dates=[datetime_name])
    d_P_filt = pd.read_csv(os.path.join(folder, 'D', 'PvsPOA_POA0_P_D.csv'), parse_dates=[datetime_name])
    d_PR = pd.read_csv(os.path.join(folder, 'D', 'PvsPOA_POA0_PR_D.csv'), parse_dates=[datetime_name])
    d_PRt = pd.read_csv(os.path.join(folder, 'D', 'PvsPOA_POA0_PRt_D.csv'), parse_dates=[datetime_name])
    d_PI = pd.read_csv(os.path.join(folder, 'D', 'PvsPOA_POA0_PI_D.csv'), parse_dates=[datetime_name])
    d_PVUSA = pd.read_csv(os.path.join(folder, 'D', 'PvsPOA_POA0_PVUSA_D.csv'), parse_dates=[datetime_name])
    d_huld = pd.read_csv(os.path.join(folder, 'D', 'PvsPOA_POA0_huld_D.csv'), parse_dates=[datetime_name])

    # Create a list to make it nicer to fix datetime-objects
    dfs = [huld_PvsPOA_POA0, PRt_PvsPOA_POA0, PVUSA_PvsPOA_POA500, PVUSA_PvsPOA_POA0, PI_PvsPOA_POA0, PR_PvsPOA_POA0, 
        P_PvsPOA_POA0, 
        #P_PvsPOA_CS_POA50, huld_PvsPOA_CS_POA50, 
        PVUSA_PvsPOA_POA200, 
        #PI_PvsPOA_CS_POA50, PR_PvsPOA_CS_POA50, PRt_PvsPOA_CS_POA50, 
        # insta_P_PvsPOA_POA50, insta_P_nofilt, insta_PI_PvsPOA_POA50, 
        P_nofilt, P_PvsPOA_POA50, PR_nofilt, PR_PvsPOA_POA50 , PRt_PvsPOA_POA50, PI_nofilt, PI_PvsPOA_POA50 , 
        PVUSA_nofilt, PVUSA_PvsPOA_POA50, huld_nofilt, huld_PvsPOA_POA50]
    for df in dfs:
        df[datetime_name] = pd.to_datetime(df[datetime_name])

    dfs_time_series_plot = {'d_P':d_P_filt, 'd_PR':d_PR, 'd_PRt':d_PRt,
                            'd_PI':d_PI, 'd_PVUSA':d_PVUSA, 'd_huld':d_huld,
                            'm_P':P_PvsPOA_POA0, 'm_PR':PR_PvsPOA_POA0, 'm_PRt':PRt_PvsPOA_POA0,
                            'm_PI':PI_PvsPOA_POA0, 'm_PVUSA':PVUSA_PvsPOA_POA0, 'm_huld':huld_PvsPOA_POA0}

    return dfs, dfs_time_series_plot


def plot_performance_time_series(folder, P_RATED, save_figure_path, datetime_name='utctime', date_range=None, save_figure=False, data=None, threshold=None):
    """
    Inputs:
        folder (str): Path to performance metric data folder (e.g., '.../helsinki_performance_metrics'). 
                      If None, input data has to be given
        data (2D array of pandas dataframes): performance time series to be plotted
    """
    fig, ax = plt.subplots(3,2, figsize=(1654/300,1654*1.2/300), dpi=700)

    alpha = 0.5 
    lwidth_bg = 1.5 # Linewidth of the plots behind the current line
    lwidth_top = 1.5
    markersize = 5
    labels = [r'$P_{MPP}/P_0$', 'huld', r'PVUSA/$P_0$', 'PR', 'PRt', 'PI']

    fontsize = 7


    if folder is not None:
        m_color1 = 'teal'
        d_color1 = 'darkturquoise'
        style = '.-.'
        markerstyle = '.'
    
        _, dataframes = read_datasets(folder, datetime_name)

        d_P = dataframes['d_P']
        d_PR = dataframes['d_PR'] 
        d_PRt = dataframes['d_PRt'] 
        d_PVUSA = dataframes['d_PVUSA'] 
        d_huld = dataframes['d_huld'] 
        d_PI = dataframes['d_PI'] 
        m_P = dataframes['m_P'] 
        m_PR = dataframes['m_PR']
        m_PRt = dataframes['m_PRt']
        m_PVUSA = dataframes['m_PVUSA']
        m_huld = dataframes['m_huld']
        m_PI = dataframes['m_PI']

        d_label = 'Daily point'
        m_label = 'Monthly trend'

        # x-location of added text elements
        start = d_P[datetime_name].iloc[0]
        end = d_P[datetime_name].iloc[-1]
        text_x = start+(end-start)/2

        ax[0,0].scatter(datetime.datetime(2019,1,1), -100, color='blue', label=d_label)
        for a in (ax[0,1], ax[1,0], ax[1,1], ax[2,0], ax[2,1]):
            a.scatter(datetime.datetime(2019,1,1), -100, color=d_color1, label=d_label)
            if threshold is not None:
                a.plot([start, end], [threshold, threshold], linestyle='dashed', color='red', alpha=0.3)

        ax[0,0].plot(d_P[datetime_name].values, d_P.P.values/P_RATED, markerstyle, alpha=0.2, color=d_color1, markersize=markersize)
        ax[0,0].plot(m_P[datetime_name].values, m_P.P.values/P_RATED, style, color=m_color1, linewidth=lwidth_top, label=m_label, markersize=markersize)
        ax[0,0].set_ylabel('Performance (-)')
        ax[0,0].set_xlabel('Datetime (year-month)')
        ax[0,0].set_ylim(-0.1, 0.8)
        ax[0,0].text(text_x, 0.7, r'$P_{MPP}/P_0$', fontsize=fontsize, ha="center")

        ax[0,1].plot(m_P[datetime_name].values, m_P.P.values/P_RATED, '.--', color='blue', alpha=alpha, linewidth=lwidth_bg, label=labels[0], markersize=markersize)
        ax[0,1].plot(d_PR[datetime_name].values, d_PR.PR.values, markerstyle, alpha=0.2, color=d_color1, markersize=markersize)
        ax[0,1].plot(m_PR[datetime_name].values, m_PR['PR'].values, style, color=m_color1, linewidth=lwidth_top, label=m_label, markersize=markersize)
        ax[0,1].set_ylabel('Performance (-)')
        ax[0,1].set_xlabel('Datetime (year-month)')
        ax[0,1].set_ylim(-0.1, 1.3)
        ax[0,1].text(text_x, 1.1, 'PR', fontsize=fontsize, ha="center")

        ax[1,0].plot(m_P[datetime_name].values, m_P.P.values/P_RATED, '.--', color='blue', alpha=alpha, linewidth=lwidth_bg, label=labels[0], markersize=markersize)
        ax[1,0].plot(d_PRt[datetime_name].values, d_PRt.PRt.values, markerstyle, alpha=0.2, color=d_color1, markersize=markersize)
        ax[1,0].plot(m_PRt[datetime_name].values, m_PRt.PRt.values, style, color=m_color1, linewidth=lwidth_top, label=m_label, markersize=markersize)
        ax[1,0].set_ylabel('Performance (-)')
        ax[1,0].set_xlabel('Datetime (year-month)')
        ax[1,0].set_ylim(-0.1, 1.3)
        ax[1,0].text(text_x, 1.1, 'PRt', fontsize=fontsize, ha="center")

        ax[1,1].plot(m_P[datetime_name].values, m_P.P.values/P_RATED, '.--', color='blue', alpha=alpha, linewidth=lwidth_bg, label=labels[0], markersize=markersize)
        ax[1,1].plot(d_huld[datetime_name].values, d_huld['huld'].values, markerstyle, alpha=0.2, color=d_color1, markersize=markersize)
        ax[1,1].plot(m_huld[datetime_name].values, m_huld['huld'].values, style, color=m_color1, linewidth=lwidth_top, label=m_label, markersize=markersize)
        ax[1,1].set_ylabel('Performance (-)')
        ax[1,1].set_xlabel('Datetime (year-month)')
        ax[1,1].set_ylim(-0.1, 1.3)
        ax[1,1].text(text_x, 1.1, 'Huld', fontsize=fontsize, ha="center")

        ax[2,0].plot(m_P[datetime_name].values, m_P.P.values/P_RATED, '.--', color='blue', alpha=alpha, linewidth=lwidth_bg, label=labels[0], markersize=markersize)
        ax[2,0].plot(d_PVUSA[datetime_name].values, d_PVUSA.PVUSA.values, markerstyle, alpha=0.2, color=d_color1, markersize=markersize)
        ax[2,0].plot(m_PVUSA[datetime_name].values, m_PVUSA.PVUSA.values, style, color=m_color1, linewidth=lwidth_top, label=m_label, markersize=markersize)
        ax[2,0].set_ylabel('Performance (-)')
        ax[2,0].set_xlabel('Datetime (year-month)')
        ax[2,0].set_ylim(-0.1, 1.3)
        ax[2,0].text(text_x, 1.1, 'PVUSA', fontsize=fontsize, ha="center")

        ax[2,1].plot(m_P[datetime_name].values, m_P.P.values/P_RATED, '.--', color='blue', alpha=alpha, linewidth=lwidth_bg, label=labels[0], markersize=markersize)
        ax[2,1].plot(d_PI[datetime_name].values, d_PI.PI.values, markerstyle, alpha=0.2, color=d_color1, markersize=markersize)
        ax[2,1].plot(m_PI[datetime_name].values, m_PI.PI.values, style, color=m_color1, linewidth=lwidth_top, label=m_label, markersize=markersize)
        ax[2,1].set_ylabel('Performance (-)')
        ax[2,1].set_xlabel('Datetime (year-month)')
        ax[2,1].set_ylim(-0.1, 1.3)
        ax[2,1].text(text_x, 1.1, 'PI', fontsize=fontsize, ha="center")



    if data is not None:
        systems = ['HEL', 'TKU', 'KUO', 'SOT-20', 'SOT-90']
        metrics = [r'$P/P_0$', 'PR', 'PRt', 'Huld', 'PVUSA', 'PI']
        # Loop through the strategies
        for d, a, metric in zip(data, [ax[0,0], ax[0,1], ax[1,0], ax[1,1], ax[2,0], ax[2,1]], metrics):
            # Loop through the systems
            for system_data, system_label in zip(d, systems):
                a.plot(system_data[datetime_name].values, system_data.iloc[:,-1].values, '.--', alpha=1, linewidth=1, markersize=1.5, label=system_label)
            a.set_ylabel('Performance (-)')
            a.set_xlabel('Datetime (year-month)')
            a.set_ylim(-0.05, 1.4)
            a.set_title(metric)
            if date_range is not None:
                a.set_xlim(date_range)

    handles, labels = ax[1,0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=5, borderaxespad=0.5, labelspacing=0.2, markerscale=0.5)
  
    for a in [ax[0,0], ax[0,1], ax[1,0], ax[1,1], ax[2,0], ax[2,1]]:
        a.tick_params(axis='x', labelrotation = 30)

    for a, letter in list(zip([ax[0,0], ax[0,1], ax[1,0], ax[1,1], ax[2,0], ax[2,1]], 
                            ['A', 'B', 'C', 'D', 'E', 'F'])):
        x0, xmax = a.set_xlim()
        y0, ymax = a.set_ylim()
        data_width = xmax - x0
        data_height = ymax - y0
        a.text(x0-(0.01*data_width), (y0 + 1.05*data_height), letter, weight='bold', fontsize=11, va='bottom', ha='right')
        a.grid(alpha=0.3)

    plt.tight_layout()

    if save_figure:
        plt.savefig(save_figure_path, bbox_inches='tight', dpi=700)
    plt.show()


def plot_plr_boxplots(file_path, save_figure_path, x='Metric', hue="Method", x_lims=(-0.5,3.5), save_figure=False, PLR_estimate=None, threshold_filter=0.8, ylims=None, ax=None, method_conditions=None):
    """
    Inputs:
        file_path: (str) path to PLR csv data
    """

    df = pd.read_csv(file_path)
    df['PLR'] = df.PLR*100
    df = df[df.POA_l > 0]  # Nighttimes are filtered in all datasets, so POA LL of 0 and 5 are redundant
    df = df.replace({'OLSLR':'LSLR'})
    df = df.replace({'CS': np.nan}, {'CS': 'no'})
    if method_conditions is not None:
        for key, value in method_conditions.items():
            df = df[df[key] == value]
    df = df.sort_values(by=['Method', 'Metric'])
    # Calcute the average PLR
    data = df
    data.reset_index(inplace=True)

    PLR_avge, PLRs_wo_outl = estimate_PLR_from_distribution(data, method='median', uncertainty_weighted=True)

    if PLR_estimate is not None:
        PLR_avge = PLR_estimate

    sns.set_theme(style="ticks", palette="muted", font_scale=0.8)

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    matplotlib.rcParams['font.family'] = 'arial'
    #sns.set(font_scale=1.2)
    #plt.figure(figsize=(160*px, 70*px), dpi=300)
    # Plot the figure
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(1063/300, 500/300), dpi=700)
    
    data2 = df[df.Metric.isin(['PR', 'PI', 'PRt', 'huld', 'PVUSA'])&
            df.Method.isin(['LSLR', 'RLR', 'YOY'])&
            df.POA_l.isin([5, 50, 100, 200, 500, 800])]

    ax = sns.boxplot(x=x, y="PLR", hue=hue, palette=["b", 'orange', 'green'], data=data2, ax=ax)
    medians = data2.groupby([x, hue])['PLR'].median()
    print(medians)
    sns.despine(offset=10)
    ax.hlines(PLR_avge, -1, (x_lims[1]+1), color='red', linestyle='-', 
            alpha=0.5, label=r'$\overline{}_{}=$'.format('{PLR}', '{'+str(threshold_filter)+'}')+f'{PLR_avge:.2f} %/a')
    ax.legend(ncols=2, bbox_to_anchor=(0,1), loc='lower left')
    ax.set_ylabel('PLR (%/a)')
    if ylims is not None:
        ax.set_ylim(ylims)
    #ax.set_ylim([-7.5,2.5])
    ax.set_xlim(x_lims)

    if save_figure:
        plt.savefig(save_figure_path, bbox_inches='tight', dpi=700)
    if ax is None:
        plt.show()

    return ax, PLR_avge


def plot_uncertainty_boxplots(file_path, save_figure_path, x='Metric', hue="Method", x_lims=(-0.5,3.5), save_figure=False, PLR_estimate=None, threshold_filter=0.8, ylims=None, ax=None, method_conditions=None):
    """
    Inputs:
        file_path: (str) path to PLR csv data
    """

    df = pd.read_csv(file_path)
    df['uncertainty'] = df.uncertainty*100
    df = df[df.POA_l > 0]  # Nighttimes are filtered in all datasets, so POA LL of 0 and 5 are redundant
    df = df.replace({'OLSLR':'LSLR'})
    df = df.replace({'CS': np.nan}, {'CS': 'no'})
    if method_conditions is not None:
        for key, value in method_conditions.items():
            df = df[df[key] == value]
    df = df.sort_values(by=['Method', 'Metric'])
    # Calcute the average PLR
    data = df
    data.reset_index(inplace=True)

    PLR_avge, PLRs_wo_outl = estimate_PLR_from_distribution(data, method='median')

    if PLR_estimate is not None:
        PLR_avge = PLR_estimate

    sns.set_theme(style="ticks", palette="muted", font_scale=0.8)

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    matplotlib.rcParams['font.family'] = 'arial'
    #sns.set(font_scale=1.2)
    #plt.figure(figsize=(160*px, 70*px), dpi=300)
    # Plot the figure
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(1063/300, 500/300), dpi=700)
    
    data2 = df[df.Metric.isin(['PR', 'PI', 'PRt', 'huld', 'PVUSA'])&
            df.Method.isin(['LSLR', 'RLR', 'YOY'])&
            df.POA_l.isin([5, 50, 100, 200, 500, 800])]

    ax = sns.boxplot(x=x, y="uncertainty", hue=hue, palette=["b", 'orange', 'green'], data=data2, ax=ax)
    medians = data2.groupby([x, hue])['uncertainty'].median()
    print(medians)
    sns.despine(offset=10)
    #ax.hlines(PLR_avge, -1, (x_lims[1]+1), color='red', linestyle='-', 
    #        alpha=0.5, label=r'$\overline{}_{}=$'.format('{PLR}', '{'+str(threshold_filter)+'}')+f'{PLR_avge:.2f} %/a')
    ax.get_legend().remove()
    ax.set_ylabel('uncertainty (%/a)')
    if ylims is not None:
        ax.set_ylim(ylims)
    #ax.set_ylim([-7.5,2.5])
    ax.set_xlim(x_lims)

    if save_figure:
        plt.savefig(save_figure_path, bbox_inches='tight', dpi=700)
    if ax is None:
        plt.show()

    return ax, PLRs_wo_outl.PLR.median()


def plot_effective_temp_coefficient(file_folder, P0, temperature_coefficient, save_figure_path, datetime_name='utctime', 
                                    irradiance_name='poa', fit='HUBER', save_figure=False, ylims=None, ax=None):
    """
    Inputs:
        file_folder: (str) path to a "filtered_datasets" folder.
        P0 (float): the nameplate rating in Wp
        temperature_coefficient (float): the nameplate temperature coefficient of power in %/degC
        fit (str): 'HUBER' or 'RANSAC'
        ax (plt ax object):
    """
    path = os.path.join(file_folder, 'PvsPOA_POA0.csv')
    df = pd.read_csv(path, parse_dates=[datetime_name])

    # Select only the first year of data
    data = df[df[datetime_name]<df.iloc[0][datetime_name]+datetime.timedelta(days=365)] 
    data = data.dropna()
    poas = [(-10,10), (95,105), (245,255), (395,405), (495,505), (595,605), (745,755)]
    poa = []
    rates = []

    if ax is None:
        fig, ax = plt.subplots(2,1,figsize=(1063/300, 1500/300), dpi=300)

    i = 0
    
    for lim in poas:

        x = data[(data[irradiance_name]>lim[0])&(data[irradiance_name]<lim[1])].cell_temp
        y = data[(data[irradiance_name]>lim[0])&(data[irradiance_name]<lim[1])].power/P0

        if len(x) == 0:
            x = np.array([-1, 0, 1])
            y = np.array([0, 0, 0])
        
        X = sm.add_constant(x)
        if fit == 'HUBER':
            rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
            res = rlm_model.fit()
            intercept = res.params[0]
            slope = res.params[1]
        elif fit == 'RANSAC':
            reg_model = RANSACRegressor(random_state=0).fit(X, y)
            slope = reg_model.estimator_.coef_[1]
            intercept = reg_model.estimator_.intercept_

        ax[0].scatter(x, y, s=1, alpha=1)
        ax[0].plot(np.array([-20,50]), intercept + slope*np.array([-20,50]), color='black', alpha=0.5)
        rates.append(slope*100)
        ax[0].set_xlim(-27, 57)
        ax[0].set_ylim(-0.03, 0.9)
        if i == 6:
            ax[0].text(x=min(x)-10, y=intercept, s=f'{slope*100:.3f}'+r'$\%/^{\circ}\mathrm{C}$')
        elif i == 0:
            ax[0].text(x=min(x)-5, y=intercept+0.02, s=f'{slope*100:.3f}'+r'$\%/^{\circ}\mathrm{C}$')
        elif i == 1:
            ax[0].text(x=min(x)-5, y=intercept+0.07, s=f'{slope*100:.3f}'+r'$\%/^{\circ}\mathrm{C}$')
        else:
            ax[0].text(x=min(x)-5, y=intercept+0.05, s=f'{slope*100:.3f}'+r'$\%/^{\circ}\mathrm{C}$')
        poa.append((lim[0] + lim[1])/2)
        i += 1
    #ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax[0].set_ylabel(r'$P/P_0$ (-)')
    ax[0].set_xlabel(r'Cell temperature ($\mathrm{^{\circ}C}$)')

    #poa = np.array([0, 100, 250, 400, 500, 600, 750])
    res = linregress(poa, rates)
    for i in range(len(poa)):
        ax[1].scatter(poa[i], rates[i], s=24, marker='^')
    ax[1].plot(np.array([0,1000]), res.intercept + res.slope * np.array([0,1000]), color='black', alpha=0.5)
    beta_1000 = res.intercept+res.slope*1000
    ax[1].scatter(1000, beta_1000, color='grey', s=50, marker='*')
    ax[1].text(700, temperature_coefficient, f'(1000,{beta_1000:.3f})')
    if ylims is not None:
        ax[1].set_ylim(ylims)
    ax[1].set_xlim(-20,1050)
    ax[1].set_xlabel(r'$G_\mathrm{rc}$ ($\mathrm{W/m^2}$)')
    ax[1].set_ylabel(r'$P/P_0/Tcell\ (\%/^{\circ}\mathrm{C})$')

    ax[1].scatter(-2, -2, s=12, marker='^', color='black', label='Measured value')
    ax[1].scatter(-2, -2, s=25, marker='*', color='black', label=r'Extrapolation to 1000 $\mathrm{W/m^2}$')
    ax[1].legend()

    for i in range(3):  # Loop 4 times to reset the color cycle
        ax[0].scatter(-2,2)
    ax[0].scatter(-2, -2, label=f'({poas[0][0]+10}'+r'$\pm$10) $\mathrm{W/m^2}$')
    for lim in poas[1:]:
        ax[0].scatter(-2, -2, label=f'({lim[0]+5}'+r'$\pm$5) $\mathrm{W/m^2}$')
    ax[0].legend(loc='lower left', bbox_to_anchor=(0, 1.05), ncol=3, handlelength=0, borderaxespad=0.5, labelspacing=0.2, markerscale=0.3)

    for a, letter in list(zip([ax[0], ax[1]], 
                            ['A', 'B'])):
        x0, xmax = a.set_xlim()
        y0, ymax = a.set_ylim()
        data_width = xmax - x0
        data_height = ymax - y0
        a.text(x0-(0.01*data_width), (y0 + 1.05*data_height), letter, weight='bold', fontsize=11, va='bottom', ha='right')

    plt.tight_layout()

    if save_figure:
        plt.savefig(save_figure_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_effective_temp_coefficient_all_in_same(file_folders, P0s, temp_coeffs, system_labels, save_figure_path, irradiance_names, datetime_name='utctime',
                                                fit='HUBER', save_figure=False, ylims=None, ax=None):
    """
    Inputs:
        file_folder: (str) path to a "filtered_datasets" folder.
        P0 (float): the nameplate rating in Wp
        temperature_coefficient (float): the nameplate temperature coefficient of power in %/degC
        ax (plt ax object):
    """

    poas = [(-10,10), (95,105), (245,255), (395,405), (495,505), (595,605), (745,755)]
    dfs = []

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(1063/300, 700/300), dpi=300)
    
    # Get the default color cycle
    color_cycle = plt.rcParams['axes.prop_cycle']

    # Extract the colors as a list
    colors = color_cycle.by_key()['color']

    color_index = 0
    for folder, P0, temp_coeff, irradiance_name, label in zip(file_folders, P0s, temp_coeffs, irradiance_names, system_labels):
        poa = []
        rates = []
        path = os.path.join(folder, 'PvsPOA_POA0.csv')
        df = pd.read_csv(path, parse_dates=[datetime_name])

        # Select only the first year of data
        data = df[df[datetime_name]<df.iloc[0][datetime_name]+datetime.timedelta(days=365)] 
        data = data.dropna()

        # Add to dataframe list
        #dfs.append(data)
        i = 0

        for lim in poas:
            # x is cell temperatures, y is normalized power
            x = data[(data[irradiance_name]>lim[0])&(data[irradiance_name]<lim[1])].cell_temp
            y = data[(data[irradiance_name]>lim[0])&(data[irradiance_name]<lim[1])].power/P0
            if len(x) == 0:
                x = np.array([-1, 0, 1])
                y = np.array([0, 0, 0])
            
            X = sm.add_constant(x)
            if fit == 'HUBER':
                rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
                res = rlm_model.fit()
                intercept = res.params[0]
                slope = res.params[1]
            elif fit == 'RANSAC':
                reg_model = RANSACRegressor(random_state=0).fit(X, y)
                intercept = reg_model.estimator_.intercept_
                slope = reg_model.estimator_.coef_[1]
                
            rates.append(slope*100)

            poa.append((lim[0] + lim[1])/2)
            i += 1
    
        res = linregress(poa, rates)
        
        ax.scatter(poa, rates, color=colors[color_index], s=12, label=label)
        ax.plot(np.array([0,1000]), res.intercept + res.slope * np.array([0,1000]), color=colors[color_index], alpha=0.5, linestyle='dashed')
        gamma_1000 = res.intercept+res.slope*1000
        ax.scatter(1000, gamma_1000, color=colors[color_index], s=50, marker='*', alpha=0.5)
        #ax.text(700, temperature_coefficient, f'(1000,{gamma_1000:.3f})')

        color_index += 1
    
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_xlim(-20,1050)
    ax.set_xlabel(r'$G_\mathrm{rc}$ ($\mathrm{W/m^2}$)')
    ax.set_ylabel(r'$P/P_0/Tcell\ (\%/^{\circ}\mathrm{C})$')

    ax.scatter(-2, -2, s=12, color='black', label='Measured value')
    ax.scatter(-2, -2, s=25, marker='*', color='black', label=r'Extrapolation to 1000 $\mathrm{W/m^2}$')
    ax.scatter(1000, -0.4, s=12, color='black', marker='^', label='Nameplate values')
    ax.legend(ncols=2)

    plt.tight_layout()

    if save_figure:
        plt.savefig(save_figure_path, dpi=300, bbox_inches='tight')
    plt.show()
