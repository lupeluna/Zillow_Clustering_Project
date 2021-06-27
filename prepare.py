import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



def handle_missing_values(df, prop_required_column = .7, prop_required_row = .7):
    ''' 
        take in a dataframe and a proportion for columns and rows
        return dataframe with columns and rows not meeting proportions dropped
    '''
    col_thresh = int(round(prop_required_column*df.shape[0],0)) # calc column threshold
    
    df.dropna(axis=1, thresh=col_thresh, inplace=True) # drop columns with non-nulls less than threshold
    
    row_thresh = int(round(prop_required_row*df.shape[1],0))  # calc row threshhold
    
    df.dropna(axis=0, thresh=row_thresh, inplace=True) # drop columns with non-nulls less than threshold
    
    return df




def split_continuous(df):
    """
    Takes in a df
    Returns train, validate, and test DataFrames
    """
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # Take a look at your split datasets

    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")
    return train, validate, test





# After clusters / before modeling

def train_validate_test(df, target):
    """
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    """
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test





# def detect_outliers(df, k, col_list):
#     ''' get upper and lower bound for list of columns in a dataframe 
#         if desired return that dataframe with the outliers removed
#     '''
    
#     odf = pd.DataFrame()
    
#     for col in col_list:

#         q1, q2, q3 = df[f'{col}'].quantile([.25, .5, .75])  # get quartiles
        
#         iqr = q3 - q1   # calculate interquartile range
        
#         upper_bound = q3 + k * iqr   # get upper bound
#         lower_bound = q1 - k * iqr   # get lower bound
        
#         # print each col and upper and lower bound for each column
#         print(f"{col}: Median = {q2} lower_bound = {lower_bound} upper_bound = {upper_bound}")

#         # return dataframe of outliers
#         odf = odf.append(df[(df[f'{col}'] < lower_bound) | (df[f'{col}'] > upper_bound)])
            
#     return odf






def create_features(df):
    df['age'] = 2017 - df.yearbuilt
    
    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100

    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560
    
    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet

    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet


#     # update datatypes of binned values to be float
#     df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',
#                     'structure_dollar_sqft_bin': 'float64', 'lot_dollar_sqft_bin': 'float64'})

    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = round((df.bathroomcnt/df.bedroomcnt), 2)

    return df



# def get_upper_outliers(s, k):
#     '''
#     Given a series and a cutoff value, k, returns the upper outliers for the
#     series.

#     The values returned will be either 0 (if the point is not an outlier), or a
#     number that indicates how far away from the upper bound the observation is.
#     '''
#     q1, q3 = s.quantile([.25, .75])
#     iqr = q3 - q1
#     upper_bound = q3 + k * iqr
#     return s.apply(lambda x: max([x - upper_bound, 0]))

# def add_upper_outlier_columns(df, k):
#     '''
#     Add a column with the suffix _outliers for all the numeric columns
#     in the given dataframe.
#     '''
#     # outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k)
#     #                 for col in df.select_dtypes('number')}
#     # return df.assign(**outlier_cols)

#     for col in df.select_dtypes('number'):
#         df[col + '_outliers'] = get_upper_outliers(df[col], k)

#     return df

# add_upper_outlier_columns(df, k=1.5)

# df.head()




# def split_zillow(df, target):
#     '''
#     this function takes in the zillow dataframe
#     splits into train, validate and test subsets
#     then splits for X (features) and y (target)
#     '''
#     # split df into 20% test, 80% train_validate
#     train_validate, test = train_test_split(df, test_size=0.2, random_state=1234)
#     # split train_validate into 30% validate, 70% train
#     train, validate = train_test_split(train_validate, test_size=0.3, random_state=1234)
#     # Split with X and y
#     X_train = train.drop(columns=[target])
#     y_train = train[target]
#     X_validate = validate.drop(columns=[target])
#     y_validate = validate[target]
#     X_test = test.drop(columns=[target])
#     y_test = test[target]
#     return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    df_dummies = df_dummies.drop(columns = ['regionidcounty'])
    return df_dummies