import pandas as pd
import numpy as np  

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
    from pytoolkit import g
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



