import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns






def get_numeric_X_cols(X_train, object_cols):
    """
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects.
    """
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]

    return numeric_cols


def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )

    return X_train_scaled, X_validate_scaled, X_test_scaled


def get_object_cols(df):
    """
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names.
    """
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()

    return object_cols






def get_heatmap(col_list, target,  color = 'mako'):
    '''
    This method will return a heatmap of all variables and there relation to logerror
    '''
    plt.figure(figsize=(14,12))
    heatmap = sns.heatmap(col_list.corr()[[target]].sort_values(by=target, ascending=False), annot=True, linewidth=0.5,fmt = '.0%',cmap = color, center = 0)
    heatmap.set_title('Feautures  Correlating with {}'.format(target))
    return heatmap



# plt.figure(figsize=(8,12))
# value_heatmap = sns.heatmap(df.corr()[[‘abs_logerr’]].sort_values(by=‘abs_logerr’, ascending=True), cmap=“PiYG”, vmin=-.5, vmax=.5, annot=True)
# value_heatmap.set_title(‘Feautures Correlating with Absolute Logerror’)
# plt.show()



def create_cluster(df, X, k, col_name = None):
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and scaled centroids as a dataframe"""
    scaler = StandardScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns = list(X))
    if col_name == None:
        #clusters on dataframe 
        df[f'clusters_{k}'] = kmeans.predict(X_scaled)
    else:
        df[col_name] = kmeans.predict(X_scaled)
        
    return df, X_scaled, scaler, kmeans, centroids_scaled



def create_scatter_plot(x, y, df, kmeans, X_scaled, scaler, hue_column= None):
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    plt.figure(figsize=(14, 9))
    sns.scatterplot(x = x, y = y, data = df, hue = hue_column)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')
    
    
    
    
    
def four_scatter_plots(X_scaled, col_name= 'column_one', col_name_two= 'column_two'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    for ax, k in zip(axs.ravel(), range(2, 6)):
        clusters = KMeans(k).fit(X_scaled).predict(X_scaled)
        ax.scatter(X_scaled[col_name], X_scaled[col_name_two], c=clusters)
        ax.set(title='k = {}'.format(k), xlabel=col_name, ylabel=col_name_two)
    
    
    
    
    
def show_cluster(X, clusters, cluster_name, size=None, hide=False):
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(X)
    X[cluster_name] = kmeans.predict(X)
    if hide == False:
        plt.figure(figsize=(16,9))
        plt.title('{} VS {}'.format(X.columns[0], X.columns[1]))
        sns.scatterplot(x= X.columns[0], y= X.columns[1], data = X, hue = cluster_name, size=size, sizes = (5,50))
        plt.show()
    return X