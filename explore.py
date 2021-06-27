import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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



def create_cluster(df, X, k):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    
    scaler = RobustScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return df, X_scaled, scaler, kmeans, centroids


def create_scatter_plot(x,y,df,kmeans, X_scaled, scaler):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')
    
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