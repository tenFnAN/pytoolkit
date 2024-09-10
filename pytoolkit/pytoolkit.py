import pandas as pd
import numpy as np  
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import inspect

def cols_change(d):  d.columns = list(map(''.join, d.columns.values)) ; return d

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
    num_v = data.select_dtypes(include=['int64', 'float64']).columns
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
    cat_v = data.select_dtypes(include=['object','category', 'string']).columns
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

    des1['variation_coef']=des1['std_dev']/des1['mean']
    
    d_quant=d.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).transpose().add_prefix('p_')
    
    des2=des1.join(d_quant, how='outer')
    
    des_final=des2.copy()
    
    des_final['variable'] = des_final.index
    
    des_final=des_final.reset_index(drop=True)
    
    des_final=des_final[['variable', 'mean', 'std_dev','variation_coef', 'p_0.01', 'p_0.05', 'p_0.25', 'p_0.5', 'p_0.75', 'p_0.95', 'p_0.99']]
    
    return des_final

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
    """
    data2=todf(data)
    
    d_cor=data2.corr(method)

    d_cor2=d_cor.reset_index() # generates index as column

    d_long=d_cor2.melt(id_vars='index') # to long format, each row 1 var

    d_long.columns=['v1', 'v2', 'R']
    
    d_long[['R2']]=d_long[['R']]**2
    
    d_long2=d_long.query("v1 != v2") # don't need the auto-correlation

    return(d_long2)



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
        return('No categorical variables to analyze.')
    
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

