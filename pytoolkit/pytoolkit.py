import pandas as pd
import numpy as np  
import os, psutil

import matplotlib.pyplot as plt
import seaborn as sns 
from pandas.plotting import parallel_coordinates
import plotly.express as px
import plotnine as ggplot

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.stats as st

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import jarque_bera 
from statsmodels.tsa.stattools import acf 

import inspect
# print(inspect.getsource(func))  # kod zrodlowy funkcji 

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
    
    return(d2)

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
    df_uniq = (data.melt().drop_duplicates().groupby('variable', as_index=False)
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
                   'std_dev':d.std().transpose()})
    # variation_coef
    des1['cv']=des1['std_dev']/des1['mean']
    
    d_quant=d.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).transpose().add_prefix('p_')
    
    des2=des1.join(d_quant, how='outer')
    
    des_final=des2.copy()
    
    des_final['variable'] = des_final.index
    
    des_final=des_final.reset_index(drop=True)
    des_final = des_final.merge(d.skew().reset_index().rename(columns={0:'skew'}),left_on='variable', right_on='index' )

    des_final=des_final[['variable', 'mean', 'std_dev','cv', 'skew', 'p_0.01', 'p_0.05', 'p_0.25', 'p_0.5', 'p_0.75', 'p_0.95', 'p_0.99']]
     
    return des_final.round(2)

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

def feat_cor_dot(data_corr, xvar):
    # Subset the data
    df_subset = data_corr[data_corr['v1'] == xvar] 

    # Add a new column "Correlation" based on the value of 'r'
    df_subset['Correlation'] = np.where(df_subset['R'] >= 0, "Positive", "Negative")

    # Reorder the 'y' column based on the absolute value of 'r'
    df_subset = df_subset.sort_values('R', ascending=True)

    # Create the plot
    p = (
        ggplot.ggplot(df_subset, ggplot.aes(x='R', y='v2', group='v2')) +
        ggplot.geom_point(ggplot.aes(color='Correlation'), size=2) +
        ggplot.geom_segment(ggplot.aes(xend=0, yend='v2', color='Correlation'), size=1) +
        ggplot.geom_vline(xintercept=0, color='#1F77B4', size=1) +
        ggplot.expand_limits(x=(-1, 1)) +
        ggplot.scale_color_manual(values={"Positive": "#2C3E50", "Negative": "#E31A1C"}) +
        ggplot.theme_bw() +
        ggplot.ggtitle(f'Cor {xvar}') 
    )

    return p

def plot_pca_features(data, col_pca, col_features, squish_lwr=0.01, squish_upr=0.99, engine='ggplot', ncol=3):
    """
    Creates a scatter plot of PCA features and scales the feature values for coloring.

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
    
    Returns:
    plot: ggplot or plotly figure
        The plot object (either from plotnine or plotly).
    """
    # Pivot the features and scale the values
    data_long = data.melt(id_vars=col_pca, value_vars=col_features, var_name='variable', value_name='value')

    # Scale the values (grouped by 'variable')
    data_long['value'] = data_long.groupby('variable')['value'].transform(lambda x: StandardScaler().fit_transform(x.values.reshape(-1, 1)).flatten())
    
    # Clip the values to the 1st and 99th percentiles within each group
    data_long['value'] = data_long.groupby('variable')['value'].transform(lambda x: kit_squishToRange(x, 0.01, 0.99))
  
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
        # Create the plot with plotly
        p = px.scatter(
            data_long,
            x=col_pca[0],
            y=col_pca[1],
            color='value',
            facet_col='variable',
            facet_col_wrap=ncol,
            # color_continuous_scale=px.colors.sequential.RdYlGn[::-1],  # Red to green
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
    if all(np.diag(data_corr) == 1) is not True:
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

    # Show the plot
    fig.show()  

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
    df_res=pd.DataFrame({'frequency': var.value_counts(), 'percentage': var.value_counts()/len(var)})
    df_res.reset_index(drop=True)
    
    df_res[name] = df_res.index
    
    df_res=df_res.reset_index(drop=True)
    
    df_res['cumulative_perc'] = df_res.percentage.cumsum()/df_res.percentage.sum()
    
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
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].apply(func)
    
    return data

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
    factor_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns
    
    # Apply the function to these columns, without copying the entire DataFrame
    data[factor_cols] = data[factor_cols].apply(func)
    
    return data

## toolkit
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
  print(f'Left whisker: {Q1 - 1.5 * IQR}\nRight whisker: {Q3 + 1.5 * IQR}')

def kit_binarize_3way(cols, anchor_value = 0):
    # sklearn.preprocessing.Binarizer
    # train['col'].apply(kit_binarize_3way)
    return np.where(cols > anchor_value, 1, np.where(cols < anchor_value, -1, 0))

## plot
def draw_boxplot_num(data, y, title="Box Plot"):

    boxplot = (
        ggplot.ggplot(data, ggplot.aes(y=y)) +  # Specify x and y variables
        ggplot.geom_boxplot() +                      # Add the box plot geometry
        ggplot.labs(title=title) +
        ggplot.coord_flip()                  
    )
    
    return boxplot

def draw_boxplot_cat(data, by, y, qn = None, title="Box Plot"):
    """
    Draw a boxplot for categorical 'by' column and a continuous 'y' column,
    treating 'by' as categorical only within the function without altering the original data.

    Args:
        data (pd.DataFrame): Input DataFrame with various column types.
        by (str): The column to group by (categorical).
        y (str): The continuous variable for the box plot.
        title (str): The title of the plot (default is 'Box Plot').

    Returns:
        plotnine.ggplot: Box plot visualizing the relationship between 'by' and 'y'.
    """
    data_ = data.copy()
    if data_[by].dtype not in ['object','category', 'string', 'bool']:
        if qn is not None:
            data_[by] = pd.qcut(data_[by], q=qn)
        data_[by] = data_[by].astype('object')
 
    boxplot = (
        ggplot.ggplot(data_) + 
        ggplot.geom_boxplot(ggplot.aes(x=by, y=y)) +                      
        ggplot.labs(title=title)              
    )
    
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


def draw_scatter(data: pd.DataFrame, feature_x: str, feature_y: str, by: str = None, title: str = "Scatter Plot", engine = 'ggplot'):
    """
    Create a scatter plot using plotnine for two features from the given DataFrame.

    Args:
        data (pd.DataFrame): The input data containing the features to be plotted.
        feature_x (str): The column name for the x-axis feature.
        feature_y (str): The column name for the y-axis feature.
        title (str): Title of the plot (default is "Scatter Plot").

    Returns:
        plot (ggplot): The scatter plot generated using plotnine.

    Example:
        plot_scatter(data=df, feature_x="age", feature_y="salary", title="Age vs Salary")
    """
    data_ = data.copy()
    if by:
        if data_[by].dtype not in ['object','category', 'string', 'bool']:
            if data_[by].nunique() <= 10:
                data_[by] = data_[by].astype('object')
    if engine == 'ggplot':
        if by:
            p = (
                ggplot.ggplot(data_, ggplot.aes(x=feature_x, y=feature_y, color=by)) +
                ggplot.geom_point() +
                ggplot.labs(title=title, x=feature_x, y=feature_y)
            )
        else:
            p = (
                ggplot.ggplot(data_, ggplot.aes(x=feature_x, y=feature_y)) +
                ggplot.geom_point(color="blue") +
                ggplot.labs(title=title, x=feature_x, y=feature_y)
            )
    elif engine == 'plotly':
        if by:
            p = px.scatter(data_, x=feature_x, y=feature_y, color=by, title=title)
        else:
            p = px.scatter(data_, x=feature_x, y=feature_y, title=title)
        p.show()
    return p

def draw_barplot_cat(data, x, y=None, by=None, type=None, title="Custom Bar Plot", label_percent=False, ncol = 3):
    """
    Create a custom bar plot using plotnine (ggplot in Python).
    
    Args:
        data (pd.DataFrame): The data to plot.
        x (str): The column to use for the x-axis.
        y (str, optional): The column to fill by (for grouped bar plots). Defaults to None.
        by (str, optional): The column to facet by. Defaults to None.
        type (str, optional): The type of bar plot ('dodge' for grouped bar plots). Defaults to None.
        title (str, optional): The title of the plot. Defaults to "Custom Bar Plot".
        label_percent (bool, optional): Whether to show percentage labels on the bars. Defaults to False.
        
    Returns:
        ggplot: The plot object.
    """

    def prop_per_x(x, count):
        """
        Compute the proportion of the counts for each value of x
        """
        df = pd.DataFrame({"x": x, "count": count})
        prop = df["count"] / df.groupby("x")["count"].transform("sum")
        return prop

    data_ = data.copy()

    # Convert `y` to categorical if it's not already
    if y is not None:
        if data_[y].dtype not in ['object', 'category', 'string', 'bool']:
            data_[y] = data_[y].astype('object')

    # Basic plot without dodging
    if type is None:
        plot = (
            ggplot.ggplot(data_, ggplot.aes(x=x)) +
            ggplot.geom_bar() +
            ggplot.labs(title=title, x=x)
        )
    
    # Grouped bar plot (dodge)
    elif type == 'dodge':
        plot = (
            ggplot.ggplot(data_, ggplot.aes(x=x, fill=y)) +
            ggplot.geom_bar(position=ggplot.position_dodge()) +
            ggplot.theme(figure_size=(10, 4),
                         dpi=80,
                         axis_text_x=ggplot.element_text(rotation=45, hjust=1)) +
            ggplot.labs(x=x, y='Count')
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

    # Add facet wrap if `by` is provided
    if by is not None:
        plot += ggplot.facet_wrap(facets=by, ncol=ncol)

    return plot

def draw_histogram(data, feature=None, title="Histogram", bins = 10):
    # sns.histplot(data['emp.var.rate'])
    if isinstance(data, pd.Series):
        feature_name = data.name
        plot_data = data.to_frame()
    # If data is a DataFrame, use the feature column
    elif isinstance(data, pd.DataFrame) and feature:
        feature_name = feature
        plot_data = data[[feature]]
    else:
        raise ValueError("Invalid input: data must be a pandas Series or DataFrame with a feature specified.")

    plot = (
        ggplot.ggplot(plot_data, ggplot.aes(x=feature_name)) +
        ggplot.geom_histogram(bins=bins, fill="skyblue", color="black") +
        ggplot.labs(title=title, x=feature_name, y="Count")
    )
    return plot

def draw_histogram_all(data, ncol = 3): 
    data_long = pd.melt(data[num_vars(data)])

    # Create the plot
    p = (
        ggplot.ggplot(data_long, ggplot.aes(x='value')) +
        ggplot.geom_histogram(bins=30) +
        ggplot.facet_wrap('variable', scales='free', ncol=ncol) +
        ggplot.theme_bw()
    )

    return p


def draw_density(data, feature, by = None, title="Density", alpha = 0.5):
    # sns.distplot(data['emp.var.rate'])
    data_ = data.copy()
    if by is None:
       plot = (
            ggplot.ggplot(data_, ggplot.aes(x=feature )) +
            ggplot.geom_density( alpha = alpha)  )
    else:
        if data_[by].dtype not in ['object','category', 'string', 'bool']:
            data_[by] = data_[by].astype('object')
        plot = (
            ggplot.ggplot(data_, ggplot.aes(x=feature, fill = by )) +
            ggplot.geom_density( alpha = alpha)  )
 
    return plot

def draw_density_all(data, ncol=3):
    # Reshape the DataFrame from wide to long format
    data_long = pd.melt(data[num_vars(data)])

    # Create the density plot
    p = (
        ggplot.ggplot(data_long, ggplot.aes(x='value', fill='variable')) +
        ggplot.geom_density(alpha=0.5) +  # Use alpha for transparency
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
    plt.title(f'Top {'' if top_n is None else top_n} important features')
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
    plt.title(f'Top {'' if top_n is None else top_n} important features')
    plt.show()

    return importance

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

def draw_ml_learning_curve(model, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    # draw_ml_learning_curve(ml_rf, "Learning Curves (Logistic Regression)", X, y, ylim=(0.7, 1.02), cv=cv, n_jobs=4)
    # Calculate learning curve data
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
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
    Użyj ttest_ind, gdy masz niezależne próbki (np. dwie różne grupy).
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
    Użyj ttest_rel, gdy masz powiązane próbki (np. przed i po pomiarach).
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
        groups = [data[data[x] == val][y] for val in data[x].unique()]
        stat, p = st.kruskal(*groups)
    
    else:
        raise ValueError("Either provide *groups or data, x, and y.")
    print(f'stat=%.3f, p=%.3f' % (stat, p))    
    if p > alpha:
        print('The distributions are likely the same (fail to reject H0).')
    else:
        print('The distributions are significantly different (reject H0).')

    return stat, p

def test_normality(data: pd.DataFrame, x : str):
    """
    Perform a normality test on a specified column of a DataFrame using the Anderson-Darling test.

    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data to be tested for normality.
    x : str
        The name of the column in the DataFrame to be tested.

    Returns:
    -------
    p : matplotlib.figure.Figure
        A histogram plot of the specified column's data, illustrating its distribution.

    Prints:
    -------
    Displays the test statistic and critical values for different significance levels, 
    indicating whether the null hypothesis (data is normally distributed) is rejected or not.
    """
    # test_normality(data.assign(n = np.random.normal(loc=50, scale=10, size=len(data)) ), 'n')
    result = st.anderson(data[x])
    print('Statistic: %.3f' % result.statistic)

    for i in range(len(result.critical_values)):
      sl, cv = result.significance_level[i], result.critical_values[i]
      if result.statistic < result.critical_values[i]:
        print('+++%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
      else:
        print('---%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
 
    p = draw_histogram(data, x)
    return p

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

def test_Ljung_Box(data):
  print('===H0: The residuals are independently distributed===')
  print('ACF')
  _, _, _, pval = acf(data, nlags=12, qstat=True, alpha=0.05)
  
  print('Null hypothesis is rejected for lags:', np.where(pval<=0.05))

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

def reduce_mem_usage(df, verbose=True):
    """
    source https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 
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