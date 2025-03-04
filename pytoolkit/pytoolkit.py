import pandas as pd
import numpy as np  
import os, psutil
import re
import inspect # print(inspect.getsource(func))  # kod zrodlowy funkcji 
from itertools import chain
#
import matplotlib.pyplot as plt
import seaborn as sns 
from pandas.plotting import parallel_coordinates
import plotly.express as px
from plotly.subplots import make_subplots
import plotnine as ggplot
#
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures 
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import TransformerMixin, BaseEstimator
import scipy.stats as st

from feature_engine.selection import SelectBySingleFeaturePerformance
#
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import jarque_bera 
from statsmodels.tsa.stattools import acf 
# qa
from sklearn.metrics import (  
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

## TODO 
# status + profnum
# sns.lmplot  
# class CFG:
#     data_folder = 'data/'
#     img_dim1 = 20
#     img_dim2 = 10
#     nepochs = 6
#     seed = 42
#     EPOCH = 300
#     bsize = 16
#     BATCH_SIZE = 1024

def cols_change(d):  
    d.columns = list(map(''.join, d.columns.values)) 
    return d

def g(obj):
    """Check object attributes, methods or parameters
    Args:
      obj  : py object
    
    Returns:
      A pandas dataframe.
       
    Example:
    --------
    from pytoolkit.pytoolkit import g
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    atr = g(clf)
    """
    # https://py4e.pl/html3/14-objects.php
    # https://py4e.pl/lessons/Objects
    
    ret = []
    for type in ['atrybuty', 'metody', 'parametry']:
        methods = []
        default = type
        if type == 'atrybuty':
            for attr in dir(obj): 
                try:
                    attr_check = callable(getattr(obj, attr))
                except:
                    print(f"An exception occurred {attr}")
                    attr_check = False
                # print(attr, attr_check)
                if not attr_check and not attr.startswith("__"):
                    methods.append(attr)      
        elif type == 'metody':
            for attr in dir(obj):
                try:
                    attr_check = callable(getattr(obj, attr))
                except:
                    print(f"An exception occurred {attr}")
                    attr_check = False
                if attr_check:
                    methods.append(attr)        
        elif type == 'parametry':
            # type = 'parametry'
            try:
                signature = inspect.signature(obj)
            except Exception as e:
                signature = type  
            if not isinstance(signature, str):
                default = []
                for param in signature.parameters.values():
                    methods.append(param.name)
                    default.append(param.default) 
        ret.append(pd.DataFrame({'name' : methods, 'default' : default, 'type' : type}))
         
    return pd.concat(ret)

def todf(data):
    """
    It converts almost any object to pandas dataframe. It supports: 1D/2D list, 1D/2D arrays, pandas series. If the object containts +2D it returns an error.
    Parameters:
    -----------
    data: data
    
    Returns:
    --------
    A pandas dataframe.

    Example:
    --------
    >> from numpy import array

    # Different case study:
    >> list_1d = [11, 12, 5, 2] 
    >> todf(list_1d)
    >> list_2d = [[11, 12, 5, 2], [15,24, 6,10], [10, 8, 12, 5], [12,15,8,6]]
    >> todf(list_2d)
    >> list_3d = [[[11, 12, 5, 2], [15,24, 6,10], [10, 8, 12, 5], [12,15,8,6]]]
    >> todf(list_3d)
    >> array_1d = array(list_1d)
    >> todf(array_1d)
    >> array_2d = array(list_2d)
    >> todf(array_2d)
    >> pd_df=pd.DataFrame({'v1':[11, 12, 5, 2], 'v2':[15,24, 6,10]}) # ok
    >> todf(pd_df)
    >> pd_series=pd_df.v1
    """
    if isinstance(data, list):
        data=np.array(data)

    if(len(data.shape))>2:
        raise Exception("I live in flattland! (can't handle objects with more than 2 dimensions)") 

    if isinstance(data, pd.Series):
        data2=pd.DataFrame({data.name: data})
    elif isinstance(data, np.ndarray):
        if(data.shape==1):
            data2=pd.DataFrame({'var': data}).convert_dtypes()
        else:
            data2=pd.DataFrame(data).convert_dtypes()
    else: 
        data2=data
        
    return data2

def status(data):
    """
    For each variable it returns: Quantity and percentage of zeros (q_zeros and p_zeros respectevly). Same metrics for NA values (q_NA/p_na), and infinite values (q_inf/p_inf). Last two columns indicates data type and quantity of unique values.
    status can be used for EDA or in a data flow to spot errors or take actions based on the result.
    
    Parameters:
    -----------
    data: It can be a dataframe or a single column, 1D or 2D numpy array. It uses the todf() function.
    
    Returns:
    --------
    A pandas dataframe containing the status metrics for each input variable.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> # dataframe as input
    >> status(iris)
    >> # single columns:
    >> status(iris['species'])
    """
    data2=todf(data)

    # total de rows
    tot_rows=len(data2)
    
    # total de nan
    d2=data2.isnull().sum().reset_index()
    d2.columns=['variable', 'q_nan']
    
    # percentage of nan
    d2[['p_nan']]=d2[['q_nan']]/tot_rows
    
    # num of zeros
    d2['q_zeros']=(data2==0).sum().values

    # perc of zeros
    d2['p_zeros']=d2[['q_zeros']]/tot_rows

    # total unique values
    d2['unique']=data2.nunique().values
    
    # get data types per column
    d2['type']=[str(x) for x in data2.dtypes.values]
    
    # mode
    # id_cols = d2[d2['unique'] == tot_rows]['variable'] 
    # d2 = d2.merge(data2.drop(columns=id_cols, errors='ignore').mode().T.rename(columns={0:'D'}),left_on='variable', right_index=True )
    d_freq = {} 
    for c in data2.columns:
        stat_freq = _freq_tbl_logic(data2[c], c) 
        if len(stat_freq) >= 1:
            stat_freq = stat_freq.loc[0,:]
            d_freq[c] = {'p_D': stat_freq.iloc[2], 'D': stat_freq.iloc[0]}
        else:
            d_freq[c] = {'p_D': np.float64(1.0), 'D': np.nan}
    d2 = d2.merge(pd.DataFrame(d_freq).T.assign(p_D = lambda x:round(x['p_D'].astype('float'),2)),left_on='variable', right_index=True )

    return d2

def unique(data):
    """
    Create a summary DataFrame with the count of unique values and 
    a comma-separated string of unique values for each column in the input DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame to summarize.

    Returns:
        pd.DataFrame: A DataFrame with the columns:
            - 'variable': The column name from the input DataFrame.
            - 'n': The count of unique values in that column.
            - 'uniq': A comma-separated string of unique values sorted in ascending order.
            
    Example:
        data = pd.DataFrame({
            'A': [1, 2, 2, 1],
            'B': ['x', 'y', 'x', 'y'],
            'C': [5.5, 6.6, 5.5, 6.6]
        })
        result = unique(data)
        print(result)
        
        Output:
            variable  n           uniq
        0        A  2           1, 2
        1        B  2           x, y
        2        C  2      5.5, 6.6
    """
    df_uniq = (data.melt().drop_duplicates().dropna().groupby('variable', as_index=False)
    .agg(n = ('value', 'nunique'), uniq = ('value', lambda x:','.join(map(str, sorted(x.unique())))) )
    .sort_values('n'))
    return df_uniq

def num_vars(data, exclude_var=None):
    """
    Returns the numeric variable names. Useful to use with pipelines or any other method in which we need to keep numeric variables. It `exclude_var` can be a list with the variable names to skip in the result. Useful when we want to skip the target variable (i.e. in a data transformation).
    It's also available for categorical variables in the function `cat_vars()`
    Parameters:
    -----------
    data: pandas dataframe
    exclude_var: list of variable names to exclude from the result
    
    Returns:
    --------
    A list with all the numeric variable names.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> num_vars(iris)
    """
    num_v = data.select_dtypes(include=['number', 'int64', 'float64']).columns
    if exclude_var is not None: 
        num_v=num_v.drop(exclude_var)
    return num_v

def cat_vars(data, exclude_var=None):
    """
    Returns the categoric variable names. Useful to use with pipelines or any other method in which we need to keep categorical variables. It `exclude_var` can be a list with the variable names to skip in the result. Useful when we want to skip the target variable (i.e. in a data transformation). It will include all `object`, `category` and `string` variables.
    It's also available for numeric variables in the function `num_vars()`
    
    Parameters:
    -----------
    data: pandas dataframe
    exclude_var: list of variable names to exclude from the result
    
    Returns:
    --------
    A list with all the categoric variable names.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> cat_vars(iris)
    """
    cat_v = data.select_dtypes(include=['object','category', 'string', 'bool']).columns
    if exclude_var is not None: 
        cat_v=cat_v.drop(exclude_var)
    return cat_v

def profiling_num(data):
    """
    Get a metric table with many indicators for all numerical variables, automatically skipping the non-numerical variables. Current metrics are: mean, std_dev: standard deviation, all the p_XX: percentile at XX number, skewness, kurtosis, iqr: inter quartile range, variation_coef: the ratio of sd/mean, range_98 is the limit for which the 98% of fall, range_80 similar to range_98 but with 80%. All NA values will be skipped from calculations.

    Parameters:
    -----------
    data: pandas  series/dataframe, numpy 1D/2D array
    
    Returns:
    --------
    A dataframe in which each row is an input variable, and each column an statistic.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> profiling_num(iris)
    """
    
    # handling different inputs to dataframe
    data=todf(data)
    
    # explicit keep the num vars
    d=data[num_vars(data)]
    
    des1=pd.DataFrame({'mean':d.mean().transpose(), 
                       'std_dev':d.std().transpose(),
                       'sum':d.sum().transpose()})
    # variation_coef
    des1['cv']=des1['std_dev']/des1['mean']
    
    d_quant=d.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).transpose().add_prefix('p_')
    
    des2=des1.join(d_quant, how='outer')
    
    des_final=des2.copy()
    
    des_final['variable'] = des_final.index
    
    des_final=des_final.reset_index(drop=True)
    des_final = des_final.merge(d.skew().reset_index().rename(columns={0:'skew'}),left_on='variable', right_on='index' )

    des_final=des_final[['variable', 'mean', 'std_dev', 'sum', 'cv', 'skew', 'p_0.0', 'p_0.01', 'p_0.05', 'p_0.25', 'p_0.5', 'p_0.75', 'p_0.95', 'p_0.99', 'p_1.0']]
    des_final = des_final.rename(columns = {'p_0.0' : 'min', 'p_1.0' : 'max'})

    cols = [col for col in des_final.columns if col != 'sum'] + ['sum']

    return des_final[cols].round(2)


def feat_cor(data, method='pearson'):
    """
    Calcuate the correlations among all numeric features. Non-numeric are excluded since it uses the `corr` pandas function.
    It's useful to quickly extract those correlated input features and the correlation between the input and the target variable.
    
    Parameters:
    -----------
    data: pandas data containing the variables to calculate the correlation
    method: `pearson` as default, same as `corr` function in pandas. 
    Returns:
    --------
    A pandas dataframe containing pairwaise correlation, R and R2 statistcs

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> feat_cor(iris)
    >> data_corr = feat_cor(num_data).loc[:, ['v1', 'v2', 'R']].pivot(index = 'v1',columns= 'v2', values = 'R').fillna(0)
    """
    data2=todf(data).select_dtypes('number')
    
    d_cor=data2.corr(method)

    d_cor2=d_cor.reset_index() # generates index as column

    d_long=d_cor2.melt(id_vars='index') # to long format, each row 1 var

    d_long.columns=['v1', 'v2', 'R']
    
    d_long[['R2']]=d_long[['R']]**2
    
    d_long2=d_long.query("v1 != v2") # don't need the auto-correlation

    return(d_long2)

def feat_cor_heatmap(data, target= None, figsize=(12, 11)):
    """
    Generate a heatmap to visualize the correlation between features.

    This function calculates the correlation coefficients between pairs of features
    using the `feat_cor` function, then creates a heatmap to display the correlation matrix.

    Parameters:
    -----------
    data_corr : pd.DataFrame
        A DataFrame containing the data for which correlations are to be computed.
        
    figsize : tuple, optional
        The size of the figure to be displayed. Default is (12, 11).

    Returns:
    --------
    None
        The function displays a heatmap of the correlation coefficients.

    Example:
    ---------
    >>> feat_cor_heatmap(data)
    """
    data_corr = feat_cor(data).loc[:, ['v1', 'v2', 'R']].pivot(index='v1', columns='v2', values='R').fillna(0)
    if target is not None:
        data_corr = data_corr.sort_values(target)
    plt.figure(figsize=figsize)
    sns.heatmap(data_corr, annot=True, fmt=".2f", annot_kws={"size": 5})
    # sns.heatmap(data_corr, vmax=1., vmin=-1., annot=True, linewidths=.8, cmap="YlGnBu")
    plt.show()

def feat_cor_dot(data_corr, target, width=6, height=5):
    """
    Create a dot plot that shows the correlation between a given variable and other variables.

    Args:
        data_corr (pd.DataFrame): The dataframe containing correlations. Expected columns are 'v1', 'v2', and 'R'.
        target (str): The variable for which correlations will be plotted.
        width (float, optional): Width of the plot in inches (default is 6 inches).
        height (float, optional): Height of the plot in inches (default is 3 inches).
    
    Returns:
        ggplot object: A dot plot showing correlations.
    """
    # Subset the data
    df_subset = data_corr[data_corr['v1'] == target]

    # Add a new column "Correlation" based on the value of 'R'
    df_subset['Correlation'] = np.where(df_subset['R'] >= 0, "Positive", "Negative")

    # Reorder the 'v2' column based on the absolute value of 'R'
    df_subset = df_subset.sort_values('R', ascending=True)

    # Create the plot
    # p = (
    #     ggplot.ggplot(df_subset, ggplot.aes(x='R', y='v2', group='v2')) +
    #     ggplot.geom_point(ggplot.aes(color='Correlation'), size=2) +
    #     ggplot.geom_segment(ggplot.aes(xend=0, yend='v2', color='Correlation'), size=1) +
    #     ggplot.geom_vline(xintercept=0, color='#1F77B4', size=1) +
    #     ggplot.expand_limits(x=(-1, 1)) +
    #     ggplot.scale_color_manual(values={"Positive": "#2C3E50", "Negative": "#E31A1C"}) +
    #     ggplot.theme_bw() +
    #     ggplot.ggtitle(f'Correlation with {target}') +
    #     ggplot.theme( 
    #         plot_title=ggplot.element_text(size=6),   
    #         axis_title_x=ggplot.element_text(size=6),   
    #         axis_title_y=ggplot.element_text(size=6),   
    #         axis_text_x=ggplot.element_text(size=6),   
    #         axis_text_y=ggplot.element_text(size=6),  
    #     )
    # )

    # # Adjust plot size in inches
    # p = p + ggplot.theme(figure_size=(width, height))
    color_mapping = {"Positive": "#2C3E50", "Negative": "#E31A1C"}

    p = px.scatter(
        df_subset.round(2),
        x='R',
        y='v2',
        color='Correlation',
        color_discrete_map=color_mapping,
        title=f'Correlation with {target}',
        labels={'R': 'R', 'v2': 'v2'} )
    for _, row in df_subset.iterrows():
        p.add_shape(
            type="line",
            x0=0,
            y0=row['v2'],
            x1=row['R'],
            y1=row['v2'],
            line=dict(color=color_mapping[row['Correlation']], width=1)
        )
        p.update_xaxes(range=[-1, 1])
        p.update_layout(
            xaxis_title="R",
            yaxis_title="v2",
            title=dict(font=dict(size=12)),
            xaxis=dict(title=dict(font=dict(size=10)), tickfont=dict(size=8)),
            yaxis=dict(title=dict(font=dict(size=10)), tickfont=dict(size=8)),
            template="plotly_white",
            width=width*100,   
            height=height*100  
        )
 
    return p

def plot_pca_features(data, col_pca, col_features, squish_lwr=0.01, squish_upr=0.99, engine='ggplot', ncol=3, hover_cols=None):
    """
    Creates a scatter plot of PCA features and scales the feature values for coloring, 
    with optional hover functionality for Plotly.

    Args:
    data: pd.DataFrame
        The dataframe with PCA and feature columns.
    col_pca: list of str
        The PCA component columns to be plotted on x and y axes.
    col_features: list of str
        The feature columns to be used for coloring the points.
    squish_lwr: float
        Lower bound for squishing the feature values to a range.
    squish_upr: float
        Upper bound for squishing the feature values to a range.
    engine: str
        Plotting engine to be used ('ggplot' for plotnine, 'plotly' for plotly express).
    ncol: int
        Number of columns for faceted plots.
    hover_cols: list of str, optional
        Additional columns to display when hovering over points (Plotly only).

    Example:
    >>> my.plot_pca_features(pcadf, ['pc1', 'pc2'], ['var_pca_master_Dim1'], engine='plotly', hover_cols=['RANK_T', 'TIER_T'])

    Returns:
    plot: ggplot or plotly figure
        The plot object (either from plotnine or plotly).
    """
    # Pivot the features and scale the values
    data_long = data.melt(id_vars=col_pca+hover_cols, value_vars=col_features, var_name='variable', value_name='value')

    # Scale the values (grouped by 'variable')
    data_long['value'] = data_long.groupby('variable')['value'].transform(
        lambda x: StandardScaler().fit_transform(x.values.reshape(-1, 1)).flatten()
    )
    
    # Clip the values to the 1st and 99th percentiles within each group
    data_long['value'] = data_long.groupby('variable')['value'].transform(
        lambda x: kit_squishToRange(x, squish_lwr, squish_upr)
    )
  
    if engine == 'ggplot':
        # Create the plot with plotnine (ggplot2 style)
        p = (
            ggplot.ggplot(data_long, ggplot.aes(x=col_pca[0], y=col_pca[1], color='value')) +
            ggplot.geom_point(size=3) +
            ggplot.scale_color_gradient(low="green", high="red") +
            ggplot.theme_bw() +
            ggplot.facet_wrap('~variable', ncol=ncol)
        )
    elif engine == 'plotly':
        # Include hover columns if provided
        hover_data = hover_cols if hover_cols else []

        # Create the plot with plotly
        p = px.scatter(
            data_long,
            x=col_pca[0],
            y=col_pca[1],
            color='value',
            facet_col='variable',
            facet_col_wrap=ncol,
            hover_data=hover_data,  # Add hover columns
            title="PCA Feature Plot"
        )
        # Adjust layout for plotly
        p.update_layout(
            xaxis_title=col_pca[0],
            yaxis_title=col_pca[1],
            coloraxis_colorbar=dict(title="Value")
        )
    else:
        raise ValueError(f"Engine '{engine}' is not supported. Use 'ggplot' or 'plotly'.")

    return p

def feat_cor_pca(data_corr, n_cluster = 3, n_dim = 2, target = None):
    """
    Perform PCA on the correlation data, cluster the results, and visualize the clusters in a scatter plot.

    Parameters:
    ----------
    data_corr : pd.DataFrame
        A DataFrame containing correlation data. If the diagonal is not all ones, 
        it will be processed to calculate feature correlations.
    
    n_cluster : int, optional
        The number of clusters to form using KMeans. Default is 3.
    
    n_dim : int, optional
        The number of dimensions to use for PCA. Default is 2.
    
    target : str, optional
        The variable to highlight in the plot. If specified, it will be used to color the points.

    Returns:
    -------
    None
        Displays a scatter plot of the PCA results with clusters.

    Notes:
    -----
    - The function checks if the diagonal of the correlation matrix is all ones. 
      If not, it computes feature correlations using a helper function `my.feat_cor`.
    - PCA is performed to reduce the dimensionality of the data.
    - KMeans clustering is applied to group the data based on PCA results.
    - A scatter plot is generated using Plotly, highlighting the clusters and target variable.
    """
    if all(np.diag(data_corr) == 1) is not True :
        if all(data_corr.columns.isin(['v1', 'v2', 'R', 'R2'])) is True:
            data_corr = data_corr.pivot(index = 'v1',columns= 'v2', values = 'R').fillna(1)
        else:
            data_corr = feat_cor(data_corr).loc[:, ['v1', 'v2', 'R']].pivot(index = 'v1',columns= 'v2', values = 'R').fillna(1)
    # dimension reduction
    pca = PCA(n_components=3)
    data_pca = pd.DataFrame(pca.fit_transform(data_corr), columns=['PC1','PC2','PC3']).assign(var = data_corr.index)
    # clustering
    ml_km = KMeans(n_cluster).fit(data_pca.iloc[:,:n_dim])

    data_pca['cl'] = ml_km.predict(data_pca.iloc[:,:n_dim])

    var_explained = np.round(np.cumsum(pca.explained_variance_ratio_)[1]*100,2)

    data_pca['col'] = (data_pca['var'] == target).astype(int).astype(str) + data_pca['cl'].astype(int).astype(str)

    # Create the plot
    fig = px.scatter(
        data_pca,
        x='PC1',
        y='PC2',
        color='col',
        text='var',
        labels={'col': 'cl', 'PC1': 'Dim_1', 'PC2': 'Dim_2'},
        title=f'var. ex. 2D: {var_explained} %'
    )

    # Add horizontal and vertical lines
    fig.add_hline(y=0, line=dict(color='black', width=0.5))
    fig.add_vline(x=0, line=dict(color='black', width=0.5))

    # Add labels using Plotly's update_traces
    fig.update_traces(textposition='top center', textfont=dict(size=12))
 
    return fig

def _freq_tbl_logic(var, name):
    """
    For internal use. Related to `freq_tbl`.

    Parameters:
    -----------
    var: pandas series
    name: column name (string)
    
    Returns:
    --------
    Dataframe with the metrics

    Example:
    --------

    """
    cnt=var.value_counts()
    df_res=pd.DataFrame({'frequency': cnt, 'percentage': round(cnt/len(var),3)})
    df_res.reset_index(drop=True)
    
    df_res[name] = df_res.index
    
    df_res=df_res.reset_index(drop=True)
    
    df_res['cumulative_perc'] = round(df_res.percentage.cumsum()/df_res.percentage.sum(),3)
    
    df_res=df_res[[name, 'frequency', 'percentage', 'cumulative_perc']]
    
    return df_res
 
def freq_tbl(data):
    """
    Frequency table for categorical variables. It retrieves the frequency, perrcentage and cummulative percentage for each categorical variables (excluding the numerical ones).

    Parameters:
    -----------
    data: pandas series/dataframe, numpy 1D/2D array
    
    Returns:
    --------
    If a single variable is passed, then it returns the table with the results (useful to be used in a processes and take actions based on the result.).
    If it contains more than one varible, it will print in the console the result for all the categorical variables (based on cat_vars). 

    Example:
    --------
    > import seaborn as sns
    > tips=sns.load_dataset('tips')
    > freq_tbl(tips)
    """
    data=todf(data)
    
    cat_v=cat_vars(data)
    if(len(cat_v)==0):
        cat_v=num_vars(data)
        # return('No categorical variables to analyze.')
    
    if(len(cat_v)>1):
        for col in cat_v:
            print(_freq_tbl_logic(data[col], name=col))
            print('\n----------------------------------------------------------------\n')
    else:
        # if only 1 column, then return the table for that variable
        col=cat_v[0]
        return _freq_tbl_logic(data[col], name=col)

def coord_plot(data, group_var): 
    """
    Coordinate plot analysis for clustering models. Also returns the original and the normalized (min-max) variable table. Useful to extract the main features for each cluster according to the variable means.
    Parameters:
    -----------
    data : Pandas DataFrame containing the variables to analyze the mean across each cluster
    group_var : String indicating the clustering variable name
    Returns:
    --------
    A tuple containing two data frames. The first contains the mean for each category across each value of the group_var. The other data set is      similar but it is min-max normalized, range [0-1].
    It also shows the coordinate or parallel plot.
    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    # If species is the cluster variable:
    >> coord_plot(iris, 'species')
    """
    # 1- group by cluster, get the means
    x_grp=data.groupby(group_var).mean()
    x_grp[group_var] = x_grp.index 
    x_grp=x_grp.reset_index(drop=True)
    x_grp # data with the original variables

    # 2- normalizing the data min-max
    x_grp_no_tgt=x_grp.drop(group_var, axis=1)

    mm_scaler = MinMaxScaler()
    mm_scaler.fit(x_grp_no_tgt)
    x_grp_mm=mm_scaler.transform(x_grp_no_tgt)

    # 3- convert to df
    df_grp_mm=pd.DataFrame(x_grp_mm, columns=x_grp_no_tgt.columns)

    df_grp_mm[group_var]=x_grp[group_var] # variables escaladas

    # 4- plot
    parallel_coordinates(df_grp_mm, group_var, colormap=plt.get_cmap("Dark2"))
    plt.xticks(rotation=90)

    return [x_grp, df_grp_mm]


## data wrangling
def mutate_if_numeric(data: pd.DataFrame, func=np.log1p) -> pd.DataFrame:
    """
    Apply a given function to all numeric columns in the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame with various column types.
        func (callable): A function to apply to all numeric columns (default is np.log1p).

    Returns:
        pd.DataFrame: A new DataFrame with the function applied to numeric columns.
    """
    # Select only numeric columns and apply the function
    data_copy = data.copy()
    numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
    data_copy[numeric_cols] = data_copy[numeric_cols].apply(func)
    
    return data_copy

def mutate_if_factor(data: pd.DataFrame, func=lambda x: x.astype('float')) -> pd.DataFrame:
    """
    Apply a given function to all categorical or object-type columns in the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame with various column types.
        func (callable): A function to apply to all categorical columns.

    Returns:
        pd.DataFrame: A new DataFrame with the function applied to categorical columns.
    """
    # Select only columns that are categorical (e.g., object, category, bool)
    data_copy = data.copy()
    factor_cols = data_copy.select_dtypes(include=['object', 'category', 'bool']).columns
    
    # Apply the function to these columns, without copying the entire DataFrame
    data_copy[factor_cols] = data_copy[factor_cols].apply(func)
    
    return data_copy

def mutate_at(data: pd.DataFrame, cols: list, func=np.log1p) -> pd.DataFrame:
    """
    Apply a given function to specific columns in a DataFrame.

    This function creates a copy of the input DataFrame and applies the given transformation function
    to the specified columns.

    Args:
        data (pd.DataFrame): The input DataFrame.
        cols (list): A list of column names (as strings) to which the function should be applied.
        func (callable): The function to apply to the specified columns. Default is np.log1p (natural logarithm of 1 + x).

    Returns:
        pd.DataFrame: A new DataFrame with the transformation applied to the specified columns.
    
    Example:
        df_transformed = mutate_at(df, cols=['col1', 'col2'], func=np.log)
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    data_copy = data.copy()
    
    # Apply the function to the specified columns
    data_copy[cols] = data_copy[cols].apply(func)
    
    return data_copy


## toolkit
def kit_encoder_ordinal_fast(x): return pd.factorize(x, sort=True)[0]
def kit_list_unpack(x): return list(chain(*x))
    # my.freq_tbl(my.kit_list_unpack([v.split() for v in features_to_inspect_interact] ) )
def kit_cat_indices(data, columns):
    return sorted([data.columns.get_loc(col) for col in columns])

def kit_squishToRange(series, lower_percentile=0.01, upper_percentile=0.99):
    """
    Clips the values in a series to the specified lower and upper percentiles.
    """
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    return series.clip(lower=lower_bound, upper=upper_bound)

def kit_calc_range(lst):
    return max(lst) - min(lst)

def kit_whiskers(feature: pd.Series):
    """
        Calculate and print the lower and upper whiskers (1.5 * IQR rule) for detecting outliers 
        in a given numerical feature using the Interquartile Range (IQR).

        Args:
            feature (pd.Series): A pandas Series representing the numerical data for which whiskers are calculated.
            
        Prints:
            Left whisker: The lower bound for detecting outliers (Q1 - 1.5 * IQR).
            Right whisker: The upper bound for detecting outliers (Q3 + 1.5 * IQR).
            
        Example:
            feature = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            calc_whiskers(feature)
            
            Output:
                Left whisker: -5.5
                Right whisker: 15.5
        """
    Q1 = np.percentile(feature, 25)
    Q3 = np.percentile(feature, 75)
    IQR = Q3 - Q1
    outl_l = Q1 - 1.5 * IQR
    outl_r = Q3 + 1.5 * IQR
    print(f'Left whisker: {outl_l}\nRight whisker: {outl_r}')
    return outl_l, outl_r

def kit_binarize_3way(cols, anchor_value = 0):
    # sklearn.preprocessing.Binarizer
    # train['col'].apply(kit_binarize_3way)
    return np.where(cols > anchor_value, 1, np.where(cols < anchor_value, -1, 0))

def kit_log_shift(data: pd.Series, shift: float = 6) -> pd.Series:
    """
    Apply a logarithmic transformation with a shift to a specified column in a DataFrame.
    np.log2(col + offset_log)
    """    
    return np.log2(data + shift)

def kit_log_shift_reverse(data: pd.Series, offset_log: float = 6) -> pd.Series:
    """
    Reverse the logarithmic transformation with an offset.
    """ 
    return np.exp2(data) - offset_log

def kit_cat_reorder(df, cat_column, feature_column, fun=np.median, ascending=True):
    """
    Reorder categorical column in a dataframe by sorting it along another feature column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the categorical and feature columns.
    cat_column : str
        The name of the categorical column to reorder.
    feature_column : str
        The name of the feature column by which the categorical column will be ordered.
    fun : callable
        A function to summarize the feature column for each category (default is np.median).
    ascending : bool
        If True, order the categorical column in ascending order of the summarized feature column.
    
    Returns
    -------
    pd.DataFrame
        A new dataframe with the categorical column reordered.
    """
    df_copy = df.copy()  # Create a copy of the input dataframe
    if len(df_copy[cat_column]) != len(df_copy[feature_column]):
        raise ValueError(
            "Lengths are not equal. len(cat_column) is {} and len(feature_column) is {}.".format(
                len(df_copy[cat_column]), len(df_copy[feature_column])
            )
        )
    # Summarize the feature column for each category in the categorical column
    summary = (df_copy[feature_column]
               .groupby(df_copy[cat_column])
               .apply(fun)
               .sort_values(ascending=ascending)
               )
    # Reorder the categorical column
    reordered_cats = pd.Categorical(df_copy[cat_column], categories=summary.index)
    df_copy[cat_column] = reordered_cats

    return df_copy

def kit_fillna_regex(data, regex, value = 0, **kwargs):
    # fillna_regex(leads_tags_df, regex = "^tag_")
    for col in data.columns:
        if re.match(pattern = regex, string = col): 
            data[col] = data[col].fillna(value = value, **kwargs)
    return data    

def kit_cat_rarelabel(data, cat_column, categories_to_keep, replace_with = 'Other'):
    # kit_cat_rarelabel(leads_tags_df, 'country_code', countries_to_keep)
    data[cat_column] = data[cat_column].apply(lambda x: x if x in categories_to_keep else replace_with)
    return data[cat_column]

def kit_remove_const(data):
    """
    Removes columns with only one unique value from a copy of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame without constant columns.
    """
    # Create a copy of the DataFrame
    df_copy = data.copy()

    # Identify columns with only one unique value
    constant_columns = [col for col in df_copy.columns if df_copy[col].nunique() == 1]
    
    # Drop the constant columns from the copy
    return df_copy.drop(columns=constant_columns)
## plot
def draw_prop_q(x, by=0.05, plot=True):
    """
    Funkcja oblicza kwantyle od 0 do 1 co 'by' (np. 0.05) na podstawie
    danych x (np. pd.Series, np.array, list) i, opcjonalnie, rysuje
    wykres tych kwantyli.

    Parametry:
    ----------
    x : array-like
        Dane, z których obliczane są kwantyle (np. list, np.array, pd.Series).
    by : float, domyślnie 0.05
        Rozmiar kroku między kolejnymi kwantylami (0 do 1).
    plot : bool, domyślnie True
        Czy wyświetlić wykres po obliczeniu kwantyli.

    Zwraca:
    ----------
    pd.Series
        Kwantyle w postaci serii, gdzie indeks to wartości kwantyli,
        a wartości to obliczone kwantyle.
    """
 
    s = pd.Series(x).dropna()
 
    q_values = np.arange(0, 1 + by, by)
    q_values[q_values > 1] = 1   
 
    quantiles = s.quantile(q_values)

    #  
    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(quantiles.index, quantiles.values, marker='o', linestyle='-')
        plt.title(' ')
        plt.xlabel('Percentile')
        plt.ylabel('Wartość')
        plt.grid(True)
        plt.show()

    return quantiles.values

def draw_boxplot_num(data, y, title="Box Plot"):

    boxplot = (
        ggplot.ggplot(data, ggplot.aes(y=y)) + 
        ggplot.geom_boxplot() +                  
        ggplot.labs(title=title) +
        ggplot.coord_flip()                  
    )
    
    return boxplot

def draw_boxplot_cat(data, by, y, qn=None, title="Box Plot", width=6, height=3, engine='ggplot'):
    """
    Draw a boxplot for a categorical 'by' column and a continuous 'y' column,
    treating 'by' as categorical only within the function without altering the original data.

    Args:
        data (pd.DataFrame): Input DataFrame with various column types.
        by (str): The column to group by (categorical).
        y (str): The continuous variable for the box plot.
        qn (int, optional): Number of quantiles or bins to cut the 'by' column into.
        title (str, optional): The title of the plot (default is 'Box Plot').
        width (int, optional): The width of the figure (default is 6).
        height (int, optional): The height of the figure (default is 3).
        engine (str, optional): The engine to use for plotting ('ggplot' or 'plotly'). Default is 'ggplot'.

    Returns:
        plot: The generated box plot (plotnine ggplot or plotly express).
    """
    data_ = data.copy()

    # Convert 'by' to categorical if it's not already, or apply quantile/bin splitting if necessary
    if data_[by].dtype not in ['object', 'category', 'string', 'bool']:
        if qn is not None:
            try:
                data_[by] = pd.qcut(data_[by], q=qn)
            except:
                data_[by] = pd.cut(data_[by], bins=qn)
        data_[by] = data_[by].astype('object')

    if engine == 'ggplot':
        # Create the boxplot with plotnine (ggplot)
        boxplot = (
            ggplot.ggplot(data_) + 
            ggplot.geom_boxplot(ggplot.aes(x=by, y=y)) +                      
            ggplot.labs(title=title) + 
            ggplot.theme(figure_size=(width, height),
               plot_title=ggplot.element_text(size=5),  
                axis_title_x=ggplot.element_text(size=6),  
                axis_title_y=ggplot.element_text(size=6), 
                axis_text_x=ggplot.element_text(size=5),  
                axis_text_y=ggplot.element_text(size=5))  # Set figure size for ggplot
        )
    
    elif engine == 'plotly':
        # Create the boxplot with plotly express
        boxplot = px.box(data_, x=by, y=y, title=title, width=width*100, height=height*100)
    
    else:
        raise ValueError("Invalid engine! Please choose either 'ggplot' or 'plotly'.")
    
    return boxplot

def draw_boxplot_all(data, ncol = 3):
    # Reshape the DataFrame from wide to long format
    data_long = pd.melt(data[num_vars(data)])

    # Create the boxplot
    p = (
        ggplot.ggplot(data_long, ggplot.aes(x='variable', y='value')) +
        ggplot.geom_boxplot() +
        ggplot.facet_wrap('variable', scales='free', ncol=ncol) +
        ggplot.theme_bw()
    )

    return p

def draw_cross_plot(data, x, target, qn_x=10, qn_target = 5 ):
    """
    Create a combined plot with a stacked bar plot and a grouped bar count plot.
    If x or y is not categorical and `discretize=True`, they will be binned into quantile-based buckets.

    Args:
        data (pd.DataFrame): Input DataFrame.
        x (str): The column name for the x-axis.
        target (str): The column name for the categorical grouping (color argument).
        qn (int, optional): Number of quantile-based buckets for discretization. Defaults to 10.
        discretize (bool, optional): Whether to discretize non-categorical x or y. Defaults to False.

    Returns:
        plotly.graph_objs._figure.Figure: Combined figure with the two plots in one row.
    """
    data_ = data.copy()

    def discretize_column(data, column, qn=None):
        """
        Discretize a column using pd.qcut or pd.cut, and format intervals to strings.

        Args:
            data (pd.DataFrame): Input DataFrame.
            column (str): Name of the column to discretize.
            qn (int): Number of bins or quantiles.

        Returns:
            pd.Series: Discretized column with formatted interval labels as strings.
        """
        if data[column].dtype not in ['object', 'category', 'string', 'bool']:
            try:
                # Use pd.qcut
                binned = pd.qcut(data[column], q=qn, precision=2)
            except ValueError:
                # Fallback to pd.cut
                binned = pd.cut(data[column], bins=qn, precision=2)

            # Format intervals as strings
            return binned.apply(lambda x: f"({x.left:.2f}, {x.right:.2f}]" if not pd.isna(x) else "NaN").astype('str')
        return data[column]
    
    if data_[x].dtype not in ['object', 'category', 'string', 'bool']:
        data_[x] = discretize_column(data_, x, qn=qn_x)
    if data_[target].dtype not in ['object', 'category', 'string', 'bool'] and data_[target].nunique() > 2:
        data_[target] = discretize_column(data_, target, qn=qn_target)
 
    # Grouping the data
    grouped_avg_score = (
        data_.groupby([x, target], as_index=False)
        .size()
        .rename(columns={'size': 'n'})
        .assign( p=lambda g: (g['n'] / g.groupby(x)['n'].transform('sum')).round(2) )
    ) 
    # Stacked bar plot
    fig1 = px.bar(
        grouped_avg_score, 
        x=x, 
        y='p', 
        color=target, 
        text='p', 
        title="Stacked Bar Plot"
    )
    fig1.update_traces(textposition='inside', texttemplate='%{text:.3f}')

    # Grouped bar count plot
    fig2 = px.bar(
        grouped_avg_score,
        x=x,
        y='n',
        color=target,
        barmode='group',
        title='Grouped Bar Count Plot',
        labels={'Count': 'Count'}
    )

    # Combine plots in subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(" ", " "),
        shared_xaxes=True
    )
    
    # Add traces from both figures
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        trace.showlegend = False  # Avoid duplicate legends
        fig.add_trace(trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        height=400,
        width=500,
        showlegend=True,
        yaxis=dict(range=[0, 1])  # Set y-axis for percentages
    )

    return fig

def draw_distr(feature, title='', bins='auto', width=16, height=9, check=False):
    """
    Draws a distribution plot (histogram and boxplot) for the given feature.
    Optionally generates additional plots for log1p and sqrt transformations.

    Args:
        feature (array-like): The data to plot.
        title (str, optional): Title for the plot. Defaults to an empty string.
        bins (str or int, optional): Number of bins or binning strategy for the histogram. Defaults to 'auto'.
        width (int, optional): The width of the figure in inches. Defaults to 16.
        height (int, optional): The height of the figure in inches. Defaults to 9.
        check (bool, optional): If True, generates plots for log1p and sqrt transformations. Defaults to False.

    Returns:
        None: The function displays the plot.
    """
    # Handle the case where feature contains negative or zero values for log1p and sqrt
    feature_log1p = np.log1p(feature) if np.all(feature >= 0) else None
    feature_sqrt = np.sqrt(feature) if np.all(feature >= 0) else None

    if check:
        # Create subplots for original, log1p, and sqrt transformations
        fig, axes = plt.subplots(2, 3, figsize=(width, height))
        transformations = [
            (feature, 'Original'),
            (feature_log1p, 'Log1p'),
            (feature_sqrt, 'Sqrt')
        ]
        
        for idx, (data, name) in enumerate(transformations):
            if data is not None:
                sns.histplot(x=data, bins=bins, ax=axes[0, idx]) 
                sns.boxplot(x=data, ax=axes[1, idx])
                axes[1, idx].set_xlabel(name)
            else:
                axes[0, idx].set_title(f'{name} not applicable')
                axes[0, idx].axis('off')
                axes[1, idx].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        # Create subplots for original data only
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height))
        sns.histplot(x=feature, bins=bins, ax=ax1)
        ax1.set_title(f'Distribution of {title}')
        sns.boxplot(x=feature, ax=ax2)
        ax2.set_xlabel(' ')
        plt.show()


def draw_pairplot(data, cols, target, engine = 'ggplot'):
    # Long format to plot pairwise relationships
    # df_long = pd.melt(data, id_vars=[target], value_vars=cols, var_name='Feature', value_name='Value')
    
    # Pairwise plot with color by target
    if engine == 'ggplot':
        plot = (
            ggplot.ggplot(data, ggplot.aes(x=cols[0], y=cols[1], color=target)) +
            ggplot.geom_point(alpha=0.6) +
            ggplot.geom_smooth(method='lm', color='blue', se=False) +
            # ggplot.facet_wrap('~ Feature', scales='free', ncol=ncol) +  # Facet wrap for each feature
            # ggplot.labs(title=f'Pairplot for {cols} and {target}') +
            ggplot.theme_bw() + 
            ggplot.theme(legend_position='right')  
        )
    elif engine == 'plotly':
        plot = px.scatter(
            data, 
            x=cols[0], 
            y=cols[1], 
            color=target,  # Color points by target
            title=f'Pairplot for {cols[0]} vs {cols[1]} by {target}',
            trendline="ols",  # Add linear regression line (optional)
            labels={cols[0]: cols[0], cols[1]: cols[1], 'color': target},
            hover_data=[target]  # Show target value on hover
        )

        # Update layout for better visualization
        plot.update_layout(
            legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="left", x=1),
            template="plotly_white"
        )
    else:
        raise ValueError("Invalid engine! Please choose either 'ggplot' or 'plotly'.")
    
    return plot


def draw_scatter(data: pd.DataFrame, x: str, y: str, by: str = None, kind='line', title: str = "Scatter Plot", engine='ggplot', width=10, height=6):
    """
    Create a scatter plot using plotnine or plotly for two features from the given DataFrame.

    Args:
        data (pd.DataFrame): The input data containing the features to be plotted.
        x (str): The column name for the x-axis feature.
        y (str): The column name for the y-axis feature.
        by (str, optional): The column name for grouping the data points by color (default is None).
        kind (str, optional): The type of plot, if 'line', adds a line between the points (default is 'line').
        title (str, optional): Title of the plot (default is "Scatter Plot").
        engine (str, optional): The plotting engine to use ('ggplot' or 'plotly', default is 'ggplot').
        width (int, optional): The width of the plot in inches (default is 10).
        height (int, optional): The height of the plot in inches (default is 6).

    Returns:
        plot: The scatter plot generated using the specified engine (plotnine ggplot or plotly).

    Example:
        draw_scatter(data=df, x="age", y="salary", title="Age vs Salary", engine="plotly", width=12, height=8)
    """
    data_ = data.copy()

    # Ensure 'by' is categorical if provided
    if by:
        if data_[by].dtype not in ['object', 'category', 'string', 'bool']:
            if data_[by].nunique() <= 10:
                data_[by] = data_[by].astype('object')

    # Use plotnine (ggplot) engine
    if engine == 'ggplot':
        if by:
            p = (
                ggplot.ggplot(data_, ggplot.aes(x=x, y=y, color=by)) +
                ggplot.geom_point() +
                ggplot.labs(title=title, x=x, y=y) +
                ggplot.theme(figure_size=(width, height))  # Set the figure size
            )
        else:
            p = (
                ggplot.ggplot(data_, ggplot.aes(x=x, y=y)) +
                ggplot.geom_point(color="blue") +
                ggplot.labs(title=title, x=x, y=y) +
                ggplot.theme(figure_size=(width, height))  # Set the figure size
            )
    
    # Use plotly engine
    elif engine == 'plotly':
        if by:
            p = px.scatter(data_, x=x, y=y, color=by, title=title, width=width*100, height=height*100)
        else:
            p = px.scatter(data_, x=x, y=y, title=title, width=width*100, height=height*100)
        
        # If kind is 'line', include lines between points
        if kind == 'line':
            p = p.update_traces(mode='lines+markers')
    
    return p



def draw_barplot_cat(data, x, y=None, by=None, kind=None, qn=None, title="Custom Bar Plot", label_percent=False, ncol=3, reverse_axis=False, width=5, height=4):
    """
    Create a custom bar plot using plotnine (ggplot in Python).
    
    Args:
        data (pd.DataFrame): The data to plot.
        x (str): The column to use for the x-axis.
        y (str, optional): The column to fill by (for grouped bar plots). Defaults to None.
        by (str, optional): The column to facet by. Defaults to None.
        kind (str, optional): The type of bar plot ('dodge' for grouped bar plots). Defaults to None.
        title (str, optional): The title of the plot. Defaults to "Custom Bar Plot".
        label_percent (bool, optional): Whether to show percentage labels on the bars. Defaults to False.
        ncol (int, optional): Number of columns for facets. Defaults to 3.
        reverse_axis (bool, optional): If True, reverses x and y axes. Defaults to False.
        width (int, optional): The width of the figure in inches. Defaults to 10.
        height (int, optional): The height of the figure in inches. Defaults to 6.
        
    Returns:
        ggplot or plotly: The plot object.
    """
    
    def prop_per_x(x, count):
        """
        Compute the proportion of the counts for each value of x
        """
        df = pd.DataFrame({"x": x, "count": count})
        prop = df["count"] / df.groupby("x")["count"].transform("sum")
        return prop

    if reverse_axis:
        x, y = y, x
    data_ = data.copy()

    # Convert `y` to categorical if it's not already
    if y is not None:
        if data_[y].dtype not in ['object', 'category', 'string', 'bool']:
            data_[y] = data_[y].astype('object')
    if qn is not None and data_[x].dtype not in ['object', 'category', 'string', 'bool']:
        try:
            data_[x] = pd.qcut(round(data_[x], 3), q=qn).astype('object')
        except:
            print('qcut fail')
            data_[x] = pd.cut(round(data_[x], 3), bins=qn).astype('object')

    # Basic plot without dodging
    if kind is None:
        plot = (
            ggplot.ggplot(data_, ggplot.aes(x=x)) +
            ggplot.geom_bar() +
            ggplot.labs(title=title, x=x) +
            ggplot.theme(figure_size=(width, height))  # Set figure size
        )
    
    # Grouped bar plot (dodge)
    elif kind == 'dodge':
        plot = (
            ggplot.ggplot(data_, ggplot.aes(x=x, fill=y)) +
            ggplot.geom_bar(position=ggplot.position_dodge()) +
            ggplot.theme(figure_size=(width, height),  # Set figure size
                         dpi=80,
                         axis_text_x=ggplot.element_text(rotation=45, hjust=1)) +
            ggplot.labs(title=title, y='Count')
        )
        if label_percent:
            plot = (plot +
                ggplot.geom_text(
                    ggplot.aes(
                        label=ggplot.after_stat("prop_per_x(x, count) * 100"),
                        y=ggplot.stage(after_stat="count", after_scale="y + 0.25"),
                    ),
                    stat="count",
                    position=ggplot.position_dodge2(width=0.9),
                    format_string="{:.1f}%",
                    size=9,
                )
            )
    
    # Stacked bar plot (plotly only)
    elif kind == 'stacked':
        data_grouped = data_.groupby([x, y] + ([by] if by else []), as_index=False).size()
        data_grouped[x] = data_grouped[x].astype('str')
        data_grouped['proportion'] = data_grouped.groupby([x] + ([by] if by else []))['size'].transform(lambda x: round(x / x.sum() * 100, 2))
        data_grouped[y] = data_grouped[y].astype('object')
        
        if by:
            # Create subplots with facet wrap if 'by' is provided
            if data_[by].dtype in [ 'category' ]: 
                unique_vals = sorted(data_[by].unique())
            else:
                unique_vals = data_[by].unique()
            nrows = -(-len(unique_vals) // ncol)  # Calculate number of rows based on columns
            plot = make_subplots(rows=nrows, cols=ncol, subplot_titles=[f"{by} = {val}" for val in unique_vals])

            for i, val in enumerate(unique_vals):
                facet_data = data_grouped[data_grouped[by] == val]
                facet_plot = px.bar(
                    facet_data,
                    x=x,
                    y='proportion',
                    color=y,
                    text='proportion',
                    labels={'proportion': 'Proportion (%)'}
                )

                for trace in facet_plot.data:
                    plot.add_trace(trace, row=(i // ncol) + 1, col=(i % ncol) + 1)

            plot.update_layout(
                title=title,
                width=width * 100,
                height=height * 100,
                barmode='stack',
                yaxis=dict(range=[0, 100]),
                xaxis_title=x,
                yaxis_title='Proportion (%)',
            )

            plot.update_traces(texttemplate='%{text:.2f}%', textposition='inside')
            plot.update_layout(showlegend=False)
        else:
            # Standard stacked bar plot without facets
            plot = px.bar(
                data_grouped,
                x=x,
                y='proportion',
                color=y,
                title=title,
                labels={'proportion': 'Proportion (%)'},
                text='proportion',
                width=width * 100,
                height=height * 100
            )

            plot.update_layout(
                barmode='stack',
                yaxis=dict(range=[0, 100]),
                xaxis_title=x,
                yaxis_title='Proportion (%)',
            )

            plot.update_traces(texttemplate='%{text:.2f}%', textposition='inside')

    # Add facet wrap if `by` is provided
    if by is not None and kind != 'stacked':
        plot += ggplot.facet_wrap(facets=by, ncol=ncol)

    return plot


def draw_count_plot(data, x, y, kind = 'count', title=' ', engine='plotly', width=800, height=600):
    """
    Draws a count plot (bar plot) to visualize the relationship between a categorical variable (x) 
    and a numerical variable (y). The plot can be generated using either `ggplot` or `plotly`.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe containing the data to plot.
    x : str
        The name of the categorical column to be shown on the x-axis.
    y : str
        The name of the numerical column to be shown on the y-axis.
    kind : str
        raw or count
    title : str, optional
        The title of the plot. Default is an empty string.
    engine : str, optional
        The plotting engine to use. Can be 'ggplot' or 'plotly'. Default is 'plotly'.

    Returns
    -------
    plot
        The generated plot based on the specified engine.
        
        - For `plotly`, returns a plotly.graph_objects.Figure object.
        - For `ggplot`, returns a plotnine.ggplot object.
    
    Notes
    -----
    - For `ggplot`, the categorical variable `x` is reordered using `kit_cat_reorder` based on the `y` values.
    - For `plotly`, the data is sorted by `y` values in descending order, and the x-axis labels are rotated.
    
    Examples
    --------
    >>> # Using plotly
    >>> plot = draw_count_plot(df_descr, 'variable', 'p_nan', title='Plotly Bar Plot', engine='plotly')
    >>> plot.show()

    >>> # Using ggplot
    >>> plot = draw_count_plot(df_descr, 'variable', 'p_nan', title='GGPlot Bar Plot', engine='ggplot')
    >>> print(plot)
    """
    
    if engine == 'ggplot':
        plot = (
            ggplot.ggplot(data.pipe(kit_cat_reorder, x, y), ggplot.aes(x=x, y=y)) +
            ggplot.geom_bar(stat='identity', fill='skyblue', color='black') +
            ggplot.labs(title=title, x='Variable', y='Percentage of NaN') +
            ggplot.theme(axis_text_x=ggplot.element_text(rotation=45, hjust=1),
                         figure_size=(width / 100, height / 100)) +  # Scaling for ggplot figure size
            ggplot.coord_flip()
        )
    
    elif engine == 'plotly':
        if kind == 'count':
            plot = px.bar(
                data.sort_values(by=y, ascending=False),
                x=x,
                y=y,
                title=title,
                width=width,   
                height=height   
            ) 
            plot.update_xaxes(tickangle=45)
        elif kind == 'raw':
            plot = px.histogram(
                data,
                x=x,
                color=y,
                title=title, 
                barmode='group'
            )
    
    return plot

 
def draw_heatmap_crosstab(data, x, y, value=None, aggfunc='size', normalize=False, title='Heatmap', width=600, height=400, zmax=None, zmin=None, round = 3):
    """
    Draws a heatmap plot from a crosstab computed from the raw DataFrame.

    Parameters:
        data (pd.DataFrame): The input data.
        x (str): Column name to use as index for the crosstab (rows).
        y (str): Column name to use as columns for the crosstab.
        value (str, optional): Column to aggregate; if None, counts occurrences.
        aggfunc (str or function): Aggregation function to use in crosstab (e.g., 'size', 'sum', 'mean').
        normalize (bool or str): Normalizes the crosstab. Options are False, True, or 'index'/'columns'.
        title (str): Title of the plot.
        width (int): Width of the plot.
        height (int): Height of the plot.
        zmax (float, optional): Maximum value for the color scale.
        zmin (float, optional): Minimum value for the color scale.
    Examples
    --------
    >>> #  
    >>> my.draw_heatmap_crosstab(adult, x='income',  y='marital.status', aggfunc='size') 
    >>> my.draw_heatmap_crosstab(adult, x='income',  y='marital.status', aggfunc='size', normalize='columns') 
    >>> my.draw_heatmap_crosstab(adult, x='income',  y='marital.status', value='age', aggfunc='mean')
 
    Returns:
        plotly.graph_objects.Figure: The heatmap plot.
    """

    # Validate input data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if x not in data.columns or y not in data.columns:
        raise ValueError(f"Columns '{x}' and '{y}' must exist in the DataFrame.")

    # Calculate the crosstab with specified aggregation function
    if value:
        crosstab_data = pd.crosstab(index=data[x], columns=data[y], values=data[value], aggfunc=aggfunc, normalize=normalize)
    else:
        crosstab_data = pd.crosstab(index=data[x], columns=data[y], normalize=normalize)

    # Handle zmin and zmax for color scaling
    zmax = zmax if zmax is not None else crosstab_data.values.max()
    zmin = zmin if zmin is not None else crosstab_data.values.min()

    # Create the heatmap plot
    plot = px.imshow(
        crosstab_data.round(round),
        labels=dict(x=y, y=x, color="Value" if not normalize else "Proportion"),
        title=title,
        width=width,
        height=height,
        zmin=zmin,
        zmax=zmax,
        text_auto=True
    )

    return plot

def draw_histogram(data, x=None, title="Histogram", bins=10):
    """
    Create a histogram for the specified feature from the given data.

    This function can handle input data in the form of a pandas Series, 
    pandas DataFrame, or a numpy ndarray. It generates a histogram using 
    the specified number of bins.

    Args:
        data (Union[pd.Series, pd.DataFrame, np.ndarray]): 
            The input data. Can be a pandas Series, a pandas DataFrame, 
            or a numpy ndarray.
        x (str, optional): 
            The name of the feature to plot if data is a DataFrame. 
            If data is a numpy ndarray, this will be used as the column name.
            Default is None.
        title (str, optional): 
            The title of the histogram. Default is "Histogram".
        bins (int, optional): 
            The number of bins for the histogram. Default is 10.

    Returns:
        ggplot.ggplot: 
            A plotnine ggplot object representing the histogram.

    Raises:
        ValueError: 
            If the input data is not a pandas Series, DataFrame, or 
            numpy ndarray.
    
    Example:
        # Using a pandas Series
        series_data = pd.Series(np.random.randn(100))
        plot1 = draw_histogram(series_data)

        # Using a pandas DataFrame
        df_data = pd.DataFrame({'feature1': np.random.randn(100)})
        plot2 = draw_histogram(df_data, feature='feature1')

        # Using a numpy ndarray
        ndarray_data = np.random.randn(100)
        plot3 = draw_histogram(ndarray_data, feature='feature1')  # feature name will be 'feature1'
    """
    # Check if data is a numpy ndarray
    if isinstance(data, np.ndarray):
        plot_data = pd.DataFrame(data, columns=['x'])  # Convert ndarray to DataFrame
        feature_name = 'x'
    # Check if data is a pandas Series
    elif isinstance(data, pd.Series):
        feature_name = data.name
        plot_data = data.to_frame()
    # If data is a DataFrame, use the feature column
    elif isinstance(data, pd.DataFrame) and x:
        feature_name = x
        plot_data = data[[x]]
    else:
        raise ValueError("Invalid input: data must be a pandas Series, DataFrame with a feature specified, or a numpy ndarray.")

    plot = (
        ggplot.ggplot(plot_data, ggplot.aes(x=feature_name)) +
        ggplot.geom_histogram(bins=bins, fill="skyblue", color="black") +
        ggplot.labs(title=title, x=feature_name, y="Count")
    )
    return plot

def draw_histogram_all(data, ncol=3, width=12, height=8):
    """
    Draws histograms for all numerical features in the DataFrame.
    
    Args:
        data (pd.DataFrame): The input DataFrame.
        ncol (int, optional): Number of columns in the facet wrap. Default is 3.
        width (int, optional): Width of the figure in inches. Default is 12.
        height (int, optional): Height of the figure in inches. Default is 8.

    Returns:
        plotnine.ggplot: The generated ggplot histogram with facets.
    """
    
    # Reshaping the data into long format for multiple histograms
    data_long = pd.melt(data[num_vars(data)])
 
    # Creating the histogram with facets and specified figure size
    p = (
        ggplot.ggplot(data_long, ggplot.aes(x='value')) +
        ggplot.geom_histogram(bins=30) +
        ggplot.facet_wrap('variable', scales='free', ncol=ncol) +
        ggplot.theme_bw() +
        ggplot.theme(figure_size=(width, height),
               plot_title=ggplot.element_text(size=5),  
                axis_title_x=ggplot.element_text(size=6),  
                axis_title_y=ggplot.element_text(size=6), 
                axis_text_x=ggplot.element_text(size=5),  
                axis_text_y=ggplot.element_text(size=5))  # Setting the figure size
    )

    return p


def draw_barplot_all(data, x=None, ncol=3, width=12, height=8):
    """
    Draws bar plots for all categorical features in the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        features (list, optional): A list of categorical columns to plot. If None, selects all categorical columns.
        ncol (int, optional): Number of columns in the facet wrap. Default is 3.
        width (int, optional): Width of the figure in inches. Default is 12.
        height (int, optional): Height of the figure in inches. Default is 8.

    Returns:
        plotnine.ggplot: The generated ggplot bar plot with facets.
    """
    if x is None:
        x = cat_vars(data)
        
    # Reshaping the data into long format
    data_long = pd.melt(data[x]) 
 
    # Creating the bar plot with facets and specified figure size
    p = (
        ggplot.ggplot(data_long, ggplot.aes(x='value')) +
        ggplot.geom_bar() +
        ggplot.facet_wrap('variable', scales='free', ncol=ncol) +
        ggplot.theme_bw() +
        ggplot.theme(figure_size=(width, height),
               plot_title=ggplot.element_text(size=5),  
                axis_title_x=ggplot.element_text(size=6),  
                axis_title_y=ggplot.element_text(size=6), 
                axis_text_x=ggplot.element_text(size=5),  
                axis_text_y=ggplot.element_text(size=5))  # Setting the figure size
    )

    return p


def draw_density(data, x, by=None, title="Density", alpha=0.5, engine='ggplot', width=10, height=6):
    """
    Draw a density plot using either ggplot or plotly.

    Args:
        data (pd.DataFrame): Input data containing the feature to be plotted.
        x (str): The column name of the feature to plot.
        by (str, optional): The column name to color the density plot by. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Density".
        alpha (float, optional): Opacity for the density fill. Defaults to 0.5.
        engine (str, optional): The plotting engine to use ('ggplot' or 'plotly'). Defaults to 'ggplot'.
        width (int, optional): The width of the figure in inches (default is 10).
        height (int, optional): The height of the figure in inches (default is 6).

    Returns:
        plot: The density plot generated using the specified engine (ggplot or plotly).
    """
    data_ = data.copy()

    # Use ggplot engine
    if engine == 'ggplot':
        if by is None:
            p = (ggplot.ggplot(data_, ggplot.aes(x=x)) +
                 ggplot.geom_density(alpha=alpha) +
                 ggplot.theme(figure_size=(width, height)) +  # Set the figure size
                 ggplot.labs(title=title))
        else:
            if data_[by].dtype not in ['object', 'category', 'string', 'bool']:
                data_[by] = data_[by].astype('object')
            p = (ggplot.ggplot(data_, ggplot.aes(x=x, fill=by)) +
                 ggplot.geom_density(alpha=alpha) +
                 ggplot.theme(figure_size=(width, height)) +  # Set the figure size
                 ggplot.labs(title=title))
        return p

    # Use plotly engine
    elif engine == 'plotly':
        p = px.histogram(
            data_,
            x=x,
            color=by,
            histnorm='density',
            title=title,
            opacity=alpha,
            width=width * 100,  # Convert inches to pixels for plotly
            height=height * 100  # Convert inches to pixels for plotly
        )
        p.update_layout(showlegend=True, title=title)
        return p

    else:
        raise ValueError("Invalid engine! Please choose either 'ggplot' or 'plotly'.")

def draw_density_all(data, by = None, ncol=3):
    # Reshape the DataFrame from wide to long format 
    if by is not None:
        data[by] = data[by].astype('object')
        data_long = pd.melt(data[ list(num_vars(data)) + [by] ], id_vars=by)
    else:
        data_long = pd.melt(data[ list(num_vars(data))], id_vars=by)

    # Create the density plot
    p = (
        ggplot.ggplot(data_long, ggplot.aes(x='value', fill = by if by is not None else 'variable' )) +
        ggplot.geom_density(alpha=0.5) +  
        ggplot.facet_wrap('variable', scales='free', ncol=ncol) +
        ggplot.theme_bw() + 
        ggplot.theme(legend_position='bottom')
    )

    return p

## ############################################### plot ml

def draw_pca_explained_variance(pca_object):
    # https://plotly.com/python/pca-visualization/#pca-analysis-in-dash
    exp_var_cumul = np.cumsum(pca_object.explained_variance_ratio_)

    p = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"} )
    
    return p

def importance_lm(ml, colnames, top_n=None):
    """
    Plots feature importance based on the absolute value of coefficients for linear models 
    and returns the importance dataframe.

    Parameters:
    -----------
    ml : fitted linear model
        The linear model (e.g., LinearRegression, LogisticRegression) that has the attribute `coef_` after fitting.
    
    colnames : list or array-like
        A list or array containing the names of the features used in the model.
    
    top_n : int, optional
        Number of top features to display in the plot. If None, all features will be displayed.

    Returns:
    --------
    importance : pandas.DataFrame
        A dataframe containing the feature names and their corresponding absolute coefficient values, 
        sorted in descending order.

    Example:
    --------
    >>> model = LogisticRegression()
    >>> model.fit(X_train, y_train)
    >>> importance_lm(model, X_train.columns, top_n=10)
    """
    # Create dataframe with absolute values of coefficients and feature names
    importance = pd.DataFrame({
        'abs_weight': np.abs(ml.coef_),
        'feature': colnames
    })

    # Sort by absolute weight and get top_n if specified
    importance = importance.sort_values(by='abs_weight', ascending=False)[:top_n]

    # Plot the top features based on the absolute values of coefficients
    plt.figure(figsize=(8, 5))
    sns.barplot(y='feature', x='abs_weight', data=importance, orient='h')

    # Set plot title and display
    plt.title(f'Top {top_n if top_n is not None else ""} important features')
    plt.show()

    return importance


def importance_tree(ml, colnames, top_n=None):
    """
    Plots feature importance for tree-based models and returns the importance dataframe.

    Parameters:
    -----------
    ml : fitted model
        The tree-based machine learning model (e.g., RandomForestClassifier, GradientBoostingClassifier, etc.)
        that has the attribute `feature_importances_` after fitting.
    
    colnames : list or array-like
        A list or array containing the names of the features used in the model.
    
    top_n : int, optional
        Number of top features to display in the plot. If None, all features will be displayed.

    Returns:
    --------
    importance : pandas.DataFrame
        A dataframe containing the feature names and their corresponding importance scores, sorted in descending order.

    Example:
    --------
    >>> model = RandomForestClassifier()
    >>> model.fit(X_train, y_train)
    >>> importance_tree(model, X_train.columns, top_n=10)
    """
    importance = pd.DataFrame({
        'imp': ml.feature_importances_,
        'feature': colnames
    })

    # Sort features by importance and get top_n if specified
    importance = importance.sort_values(by='imp', ascending=False)[:top_n]

    # Plot the feature importances
    plt.figure(figsize=(8, 5))
    sns.barplot(y='feature', x='imp', data=importance, orient='h')

    # Set plot title and display
    plt.title(f'Top {top_n if top_n is not None else ""} important features')
    plt.show()

    return importance

def tree_explain(data, target='score', model=None, type_ = 'clf', top_n=None, **kwargs):
    """
    Tworzy model względem zmiennej target i zwraca wykres z istotnością zmiennych.
    tree_explain(ds[col_cat + ['target']], target = 'target', max_depth = 22)
    Args:
        data (pd.DataFrame): Dane wejściowe zawierające cechy i target.
        target (str): Nazwa zmiennej celu.
        model: Model drzewa decyzyjnego (np. xgb.XGBRegressor).

    Returns:
        plt.Figure: Wykres istotności zmiennych.
    """
    if model is None and type_ == 'clf' : 
        from xgboost import XGBClassifier
        model = XGBClassifier(**kwargs, enable_categorical = True)
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor(**kwargs, enable_categorical = True)

    # Podział na cechy (X) i zmienną celu (y)
    X = data.drop(columns=[target])
    y = data[target]

    # Trening modelu
    model.fit(X, y)
  
    # Tworzenie DataFrame z istotnością zmiennych 
    importance_df = importance_tree(model, X.columns, top_n)

    return importance_df

def draw_ml_residuals(X_val: pd.DataFrame, y_val: pd.core.series.Series, ml):
    """
    Draws residual plots to analyze model errors using Plotnine with facet_wrap.

    Args:
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): True target values for validation.
        ml: Trained linear regression model (must have a predict method).
    
    Returns:
        None: Displays the residual plots.
    """
    # Predict the values and calculate residuals
    pred = ml.predict(X_val)
    errors = y_val - pred

    # Create a DataFrame for easier plotting
    plot_data = pd.DataFrame({
        'Index': range(len(errors)),
        'Errors': errors,
        'Predicted': pred
    })

    # Plot 1: Distribution of errors with Index on x-axis
    plot1 = (
        ggplot.ggplot(plot_data, ggplot.aes(x='Index', y='Errors')) +
        ggplot.geom_point(color="blue") +
        ggplot.labs(title="Distribution of Errors", x="Index", y="Error")
    )

    # Plot 2: Relationship between Predicted values and Errors (residual analysis)
    plot2 = (
        ggplot.ggplot(plot_data, ggplot.aes(x='Predicted', y='Errors')) +
        ggplot.geom_point(color="red") +
        ggplot.labs(title="Predicted Value vs Error (Residual Analysis)", x="Predicted Value", y="Error")
    ) 
    return plot1, plot2

def draw_ml_learning_curve(model, title, X, y, ylim=None, cv=None, scoring = 'roc_auc', n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    # draw_ml_learning_curve(ml_rf, "Learning Curves (Logistic Regression)", X, y, ylim=(0.7, 1.02), cv=cv, n_jobs=4)
    # Calculate learning curve data
    train_sizes, train_scores, test_scores = learning_curve(model, X, y,scoring=scoring, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    # Compute means and standard deviations
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Create a dataframe to hold data for plotting
    data = pd.DataFrame({
        'train_size': np.concatenate([train_sizes, train_sizes]),
        'score_mean': np.concatenate([train_scores_mean, test_scores_mean]),
        'score_std': np.concatenate([train_scores_std, test_scores_std]),
        'dataset': ['Train'] * len(train_sizes) + ['Validation'] * len(train_sizes)
    })
    
    # Plot the learning curve
    p = (
        ggplot.ggplot(data, ggplot.aes(x='train_size', y='score_mean', color='dataset')) +
        ggplot.geom_line(size=1.5) +
        ggplot.geom_ribbon(ggplot.aes(ymin='score_mean - score_std', ymax='score_mean + score_std', fill='dataset'), alpha=0.2) +
        ggplot.labs(title=title, x='Training examples', y='Score') +
        ggplot.theme_bw() +
        ggplot.scale_x_continuous(breaks=train_sizes)  # Add breaks for training sizes
    )
    
    return p 

def draw_hparam_gridsearch(data, cols_target = 'mean_test_score', cols_param = None, ncol = 3):

    if cols_param is None:
        cols_param = data.filter(regex='param').drop(columns=['params'], errors='ignore').columns
    
    # Filter out columns with only one unique value
    # cols_param = [c for c in cols_param if data[c].nunique() > 1]

    # Bin continuous variables with >=8 unique values into quartiles
    for c in cols_param:
        if data[c].nunique() >= 8 and pd.api.types.is_numeric_dtype(data[c]):
            data[c] = pd.qcut(data[c], q=4)

    # Melt  
    df_melted = data.melt(id_vars=[cols_target], 
                          value_vars=cols_param,
                          var_name='parameter', 
                          value_name='value')
    p = (ggplot.ggplot(df_melted, ggplot.aes(x='value', y=cols_target)) +
        ggplot.geom_boxplot() +
        ggplot.facet_wrap('parameter', scales='free', ncol=ncol) +
        ggplot.theme_bw())
    return p

## FEATENG

def estimator_avg_robust(x, round_ = 2): return round((np.median(x) + np.mean(x))/2,round_)

def filterVarImp(X, y, est=DecisionTreeClassifier(random_state=123), metric='roc_auc', thresh=0.51, cv=5):
    """
    est=DecisionTreeRegressor(random_state=123), metric='neg_root_mean_squared_error'
    est=LogisticRegression(), metric='roc_auc'

    # roc_auc < 0.53
    # 
    Filters and ranks features based on their importance using a specified estimator and evaluation metric.

    This function applies `SelectBySingleFeaturePerformance` to assess feature importance by evaluating
    the performance of individual features against the target variable. Features with performance above
    the defined threshold are selected and their importance scores are ranked and returned.

    Parameters:
    ----------
    X : pd.DataFrame
        The input feature matrix. Each column represents a feature, and each row represents an observation.
    
    y : pd.Series or array-like
        The target variable associated with the input features.

    est : estimator object, default=DecisionTreeClassifier(random_state=123)
        The estimator object used for evaluating feature performance. Should follow the scikit-learn API.
        Example: DecisionTreeClassifier, RandomForestClassifier, etc.

    metric : str, default='roc_auc'
        The scoring metric used to evaluate feature performance. Should be one of the valid scoring metrics 
        supported by scikit-learn, such as 'accuracy', 'roc_auc', 'f1', etc.

    thresh : float, default=0.51
        The threshold value used to filter features based on their performance. Only features with a performance
        score higher than the threshold are selected.

    cv : int, default=5
        The number of cross-validation folds used to assess feature performance.

    Returns:
    -------
    res : dict
        A dictionary with two keys:
        - 'imp': A dictionary of features and their corresponding performance scores, sorted in ascending order
                 and rounded to three decimal places.
        - 'select': A list of selected feature names (with 'missingindicator_' removed if present).

    Example:
    --------
    >>> result = filterVarImp(X, y, est=DecisionTreeClassifier(), metric='accuracy', thresh=0.6)
    >>> print(result['imp'])    # Print feature importance scores
    >>> print(result['select']) # Print selected features

    """
    sel = SelectBySingleFeaturePerformance(estimator=est, scoring=metric, cv=cv, threshold=thresh)
    sel.fit(X, y)  
    d_imp = round(pd.Series(dict(sorted(sel.feature_performance_.items(), key=lambda item: item[1]))),3) 
    
    res = { 
        'imp': d_imp,
        'select': [v.replace('missingindicator_', '') for v in sel.transform(X).columns.to_list()]
    }
    
    return res

def feat_interactions(X, degree=2, interaction_only=True, include_bias=False):
    """
    Generates interaction features for a given DataFrame.
    
    Parameters:
    - X (pd.DataFrame): Input DataFrame with original features.
    - degree (int): Degree of interactions to generate. Default is 2.
    - interaction_only (bool): If True, only interaction terms are produced. Default is True.
    - include_bias (bool): If True, includes the bias column (all ones). Default is False.
    
    Returns:
    - pd.DataFrame: DataFrame with original and interaction features.
    """
    
    # Create PolynomialFeatures with specified parameters
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    
    # Generate interaction features
    X_interactions = poly.fit_transform(X)
    
    # Create feature names for interaction terms
    interaction_feature_names = poly.get_feature_names_out(input_features=X.columns)
    
    # Return a DataFrame with interaction features
    X_interactions_df = pd.DataFrame(X_interactions, columns=interaction_feature_names, index=X.index)
    X_interactions_df = X_interactions_df.T.drop_duplicates().T

    return X_interactions_df

def feat_ngram(data, n, select = '_', replace = 'var_', nmin=5):
    txt_vec = data.columns[data.columns.str.contains(select)].tolist()  
    txt_vec = [name.replace(replace, '') for name in txt_vec]   
    txt_vec = [name.replace('_', ' ') for name in txt_vec]

    n_grams = []
    for phrase in txt_vec:
            tokens = phrase.split()
            # Generate n-grams manually
            for i in range(len(tokens) - n + 1):
                n_grams.append(' '.join(tokens[i:i + n]))
    return freq_tbl(n_grams).query("frequency >= @nmin").rename(columns = {0:'var'}).assign(var=lambda x:x['var'].str.replace(' ', '_'))

class FeatureRegexSelector(BaseEstimator, TransformerMixin):
    #  ('fselect',       my.FeatureRegexSelector(exclude = True, cols_regex='NONE')),
    def __init__(self, exclude = True,  cols_regex = None):
        self.cols_regex = cols_regex
        self.exclude = exclude
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        if self.exclude:
            col_names = X.columns
            r = re.compile( ".*" + self.cols_regex )
            col_filter = list(filter(r.match, col_names)) 
            col_filtered = np.setdiff1d(col_names, col_filter) 
            # print(len(col_filtered))
            return X.loc[:,col_filtered]
        else:
            return X.filter(regex=self.cols_regex)


## Statistical tests 

def test_cor_significance(data: pd.DataFrame, 
                         x : list,
                         y: str,
                         method: str ,
                         alpha: float = 0.05 
                           ):
    p_values = []
    for cols in x:
        if method == 'pearson':
          p_value = round(st.pearsonr(data[cols], data[y])[1], 3)
        else:
          p_value = round(st.spearmanr(data[cols], data[y])[1], 3)

        p_values.append(p_value)
        print(f'{cols} <-> {y} correlation: p-value = {p_value}')

        if p_value > alpha:
            print('This correlation coefficient is NOT significant!\n')
        else:
          print('This correlation coefficient is significant\n') 
    return

def test_ttest_ind_significance(sample1, sample2, alpha=0.05):
    """
    test_ttest_significance(data[data['yr']==0]['cnt'], data[data['yr']==1]['cnt'])
    Perform an independent t-test on two samples and interpret the p-value.
    This is a test for the null hypothesis that 2 independent samples have identical average.
    Test t-Studenta dla dwóch niezależnych prób
    Probki moga miec rozny rozmiar.

    Parameters:
    - sample1: array-like, first sample data
    - sample2: array-like, second sample data
    - alpha: significance level for the test (default is 0.05)

    Returns:
    - t_stat: t-statistic of the test
    - p_value: p-value of the test
    - interpretation: string interpretation of the p-value
    """
    # Calculate the t-statistic and p-value
    t_stat, p_value = st.ttest_ind(sample1, sample2)

    # Interpretation of the p-value
    if p_value < alpha:
        interpretation = "Reject the null hypothesis: There is a significant difference between the two groups."
    else:
        interpretation = "Fail to reject the null hypothesis: No significant difference between the two groups."

    return t_stat, p_value, interpretation

def test_ttest_rel_significance(sample1, sample2, alpha=0.05):
    """
    Perform a paired t-test on two samples and interpret the p-value.
    This is a test for the null hypothesis that two related samples have identical average.
    Test t-Studenta dla prób zależnych, powiązane próbki (np. przed i po eksperymencie/badaniach/leczeniu).
    Probki musza miec ten sam rozmiar.
    
    Parameters:
    - sample1: array-like, first sample data
    - sample2: array-like, second sample data
    - alpha: significance level for the test (default is 0.05)

    Returns:
    - t_stat: t-statistic of the test
    - p_value: p-value of the test
    - interpretation: string interpretation of the p-value
    """
    # Calculate the t-statistic and p-value
    t_stat, p_value = st.ttest_rel(sample1, sample2)

    # Interpretation of the p-value
    if p_value < alpha:
        interpretation = "Reject the null hypothesis: There is a significant difference between the two groups."
    else:
        interpretation = "Fail to reject the null hypothesis: No significant difference between the two groups."

    return t_stat, p_value, interpretation

def test_chi2_significance(data: pd.DataFrame, 
                         x : str,
                         y: str, 
                         alpha: float = 0.05 
                           ):
    stat, p, dof, expected  = st.chi2_contingency(pd.crosstab(data[x], data[y])) 
 
    print('p-value:', round(p, 5))
    if p <= alpha:
        v = '+++'
        s = 'dependent'
        print('reject H0 | there is a association between the two groups')
    else:
        v = '---'
        s = 'independent'
        print('fail to reject H0 | there is no association between the two groups')

    return [np.round(p,2), v, s]

def test_kruskal_significance(*groups, data=None, x=None, y=None, alpha=0.05 ):
    """
    Perform Kruskal-Wallis H-test to determine if there are statistically significant 
    differences between the distributions of two or more independent samples.
 
    Args:
        *groups: Variable number of independent sample groups (e.g., group_1, group_2, ...).
        data (DataFrame, optional): DataFrame containing the data for grouping.
        x (str, optional): Column name for grouping (categorical variable).
        y (str, optional): Column name with the values to compare across groups (numerical variable).
        alpha (float, optional): Significance level to test the hypothesis (default is 0.05).

    Returns:
        stat (float): Test statistic from the Kruskal-Wallis test.
        p (float): P-value from the test.
    
    Example usage:
        1. Direct group input:
            test_kruskal_significance(group_1, group_2, group_3)
        
        2. From DataFrame:
            test_kruskal_significance(data=df, x='group_col', y='value_col')
    """   
    if groups:
        stat, p = st.kruskal(*groups)
    
    # Option 2: If data, x, and y are passed
    elif data is not None and x is not None and y is not None:
        # Create groups from the DataFrame based on unique values in column x
        data_ = data[[x, y]].dropna().copy()
        groups = [data_[data_[x] == val][y] for val in data_[x].dropna().unique()]
        stat, p = st.kruskal(*groups)
    
    else:
        raise ValueError("Either provide *groups or data, x, and y.")
    print(f'stat=%.3f, p=%.3f' % (stat, p))    
    if p > alpha:
        print('The distributions are likely the same (fail to reject H0).')
    else:
        print('The distributions are significantly different (reject H0).')

    return stat, p
 
def test_normality(data: pd.DataFrame, x: str):
    """
    Perform a normality test on a specified column of a DataFrame using the Anderson-Darling test.
    # Test Andersona-Darlinga            Dobrze działa zarówno dla małych, jak i średnich próbek, szczególnie gdy wartości skrajne mają znaczenie
    # Test Shapiro-Wilka                 Bardzo czuły test dla małych próbek (n < 50).
    # Test Kolmogorova-Smirnova (K-S)    Stosowany do porównania rozkładów dla dowolnych rozkładów, nie tylko normalnych.
    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data to be tested for normality.
    x : str
        The name of the column in the DataFrame to be tested.

    Returns:
    -------
    Figure
        A histogram and Q-Q plot of the specified column's data, illustrating its distribution.
    """
    # Anderson-Darling test
    result = st.anderson(data[x])
    print('Anderson-Darling Test Statistic: %.3f' % result.statistic)

    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print('+++%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        else:
            print('---%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
    
    # Plotting
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data[x], kde=True)
    plt.title('Histogram of {}'.format(x))
    plt.xlabel(x)
    plt.ylabel('Frequency')

    # Q-Q plot
    plt.subplot(1, 2, 2)
    st.probplot(data[x], dist="norm", plot=plt)
    plt.title('Q-Q Plot of {}'.format(x))
    
    plt.tight_layout()
    plt.show()
    
    return plt

# Quality
def qa_clf(model, name, y_test, X_test):     
    y_pred      = model.predict(X_test) 
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    specificity = recall_score(y_test, y_pred, pos_label=0)
    pr_auc = average_precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    # KS statistic 
    positive_scores = y_pred_prob[y_test == 1]
    negative_scores = y_pred_prob[y_test == 0] 
    ks_statistic, _ = st.ks_2samp(positive_scores, negative_scores)
    #
    kppa = cohen_kappa_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    # conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred), #prawd_rl>punkt_odciecia
             columns = ['predicted negatives', 'predicted positives'], 
             index = ['actual negatives', 'actual positives'])

    report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])  # Adjust target_names as needed
    print(conf_matrix)
    print(report)

    return pd.DataFrame({
        'name' : name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'kappa':kppa,
        'f1': f1,
        'ks': ks_statistic,
        'pr_auc' : pr_auc ,
        'auc' : auc
        # 'gini' : 2 * auc - 1
      
    }, index=[name]).round(3)

# ts
def test_dickeyF(data, alpha=0.05):
    print("==== Augmented DickeyFuller (Null hypothesis - The process is non-stationary): ")

    result = adfuller(data, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

    if result[1] < alpha:
        print("The process is stationary.\n")
    else:
        print("The process is non-stationary.\n")

def test_kpss(data, alpha=0.05):
    print('==== Kwiatkowski PhillipsSchmidtShin (KPSS) test (Null hypothesis - The process is stationary):')

    kpsstest = kpss(data, regression='c')
    print("KPSS Statistic = " + str(kpsstest[0]))
    print( "p-value = " +str(kpsstest[1]))

    if kpsstest[1] < alpha:
        print("The process is non-stationary.\n")
    else:
        print("The process is stationary.\n")

def test_Ljung_Box(data, nlags=12):
    # Verify whether time series/residuals are white noise 
    print('=== H0: The residuals are independently distributed, no autocorrelation ===') 
    print('H1 : there is serial correlation')
    # Compute ACF and p-values
    _, _, _, pval = acf(data, nlags=nlags, qstat=True, alpha=0.05)

    # Identify lags where the null hypothesis is rejected
    rejected_lags = np.where(pval <= 0.05)[0]

    # Output results
    if len(rejected_lags) > 0:
        print(f"Null hypothesis is rejected for the following lags (indicating autocorrelation): {rejected_lags}")
        print("This suggests that the residuals are not independently distributed and may contain autocorrelation.")
    else:
        print("Null hypothesis is not rejected for all lags (indicating no autocorrelation).")
        print("This suggests that the residuals are independently distributed and can be considered white noise.")

def test_jarq_bera(data, alpha = 0.05):
    print('==== The Jarque-Bera test of normality (Null hypothesis - The distribution is normal):')
    jarque_beratest = jarque_bera(data)

    sns.histplot(x=data, bins=20) ; plt.show()

    print("JB Statistic = " + str(jarque_beratest[0]))
    print( "p-value = " +str(jarque_beratest[1]))

    if jarque_beratest[1]< 0.05:
        print("The distribution is non-normal.")
    else:
        print("The distribution is normal.")


# import gc
# gc.collect()
def reduce_mem_usage(df, verbose=True):
    """ iterate through all the columns of a dataframe and modify the data type to reduce memory usage.     
    source https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 
           https://github.com/PacktPublishing/The-Kaggle-Book/blob/main/chapter_07/reduce_mem_usage.py
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def reduce_mem_usage_pl(df):
        """ Reduce memory usage by polars dataframe {df} with name {name} by changing its data types. 
            source https://www.kaggle.com/code/yuuniekiri/fork-of-home-credit-catboost-inference
        """
        print(f"Memory usage of dataframe is {round(df.estimated_size('mb'), 2)} MB")
        
        Numeric_Int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
        Numeric_Float_types = [pl.Float32, pl.Float64]    
        
        for col in df.columns:
            if col == 'case_id': 
                continue
            try:
                col_type = df[col].dtype
                
                if col_type == pl.Categorical:
                    continue
                    
                c_min = df[col].min()
                c_max = df[col].max()
                
                if col_type in Numeric_Int_types:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df = df.with_columns(df[col].cast(pl.Int8))
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df = df.with_columns(df[col].cast(pl.Int16))
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df = df.with_columns(df[col].cast(pl.Int32))
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df = df.with_columns(df[col].cast(pl.Int64))
                
                elif col_type in Numeric_Float_types:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df = df.with_columns(df[col].cast(pl.Float32))
                    else:
                        pass
                # elif col_type == pl.Utf8:
                #     df = df.with_columns(df[col].cast(pl.Categorical))
                else:
                    pass
            except:
                pass
        print(f"Memory usage of dataframe became {round(df.estimated_size('mb'), 2)} MB")
        return df

def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 

def get_memory_usage_df(df):
    """
    Returns the memory size of a DataFrame in megabytes.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - Memory size in MB
    """
    memory_size = df.memory_usage(deep=True).sum()
    memory_in_mb = memory_size / (1024 ** 2)
    return f"Memory size of DataFrame: {memory_in_mb:.2f} MB"

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

# def seed_everything(seed,  tensorflow_init=True,  pytorch_init=True):
#     """
#     source https://github.com/PacktPublishing/The-Kaggle-Book/blob/main/chapter_07/seed_everything.py
#     Seeds basic parameters for reproducibility of results
#     """
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     if tensorflow_init is True:
#         tf.random.set_seed(seed)
#     if pytorch_init is True:
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False  
