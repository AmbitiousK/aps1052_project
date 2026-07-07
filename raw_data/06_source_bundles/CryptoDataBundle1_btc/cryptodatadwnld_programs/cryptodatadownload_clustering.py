
# No Express Warranty
# Import the libraries we are going to use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns


def main():
    df = load_data()  # call the function to load the data and it returns a Pandas DataFrame
    print('Downloaded data ...')

    scaler = MinMaxScaler()  # create a scale to transform the numeric data to be between 0-1 since Kmeans is sensitive to outliers
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns) # create dataframe out of the scaled data

    kmeans = KMeans(n_clusters=3)  # initialize the kmeans model object with 3 clusters
    kmeans.fit(scaled_df)  # fit the scaled data to our kmeans model component

    pca_num_components = 2  # number of PCA components to limit feature set to
    pca = PCA(n_components=pca_num_components)  # create PCA model object with 2 components
    pca_result = pca.fit_transform(scaled_df)   # fit the scaled data to the PCA analysis
    pca_df = pd.DataFrame(data=pca_result, columns=['pca1', 'pca2'])  # transform the PCA analysis back to dataframe

    scaled_df['Cluster'] = kmeans.labels_   # add our cluster labels back to the PCA so we know which points were grouped together
    df['Cluster'] = kmeans.labels_

    sns.scatterplot(x="pca1", y="pca2", hue=scaled_df['Cluster'], data=pca_df)  # create scatterplot of PCA result
    plt.title('K-MEANs Clustering with PCA Applied to Reduce Dimensions')
    plt.show()
    print('wait')


def calculate_number_centroids(data):
    """This function will calculate Kmeans for a range number of clusters and then will calculate the average distance
    between each of the centroid clusters and the data, termed "inertia" - we will plot the explained variation as a
    function of the number of clusters. Where the elbow exists is the "optimal" number of clusters in KMEANS"""
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Plotting the results onto a line graph
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # within cluster sum of squares
    plt.show()


def load_data():
    """Function will load the Binance ETHUSDT spot timeseries and the Binance meta statistics
       combine into one pandas dataframe, we will join values using the same date
        """
    url_stats = 'https://www.cryptodatadownload.com/cdd/Binance_summary_statistics_ETHUSDT_premium.csv'  # endpoint for stats data
    url_ohlc = 'https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_d.csv'  # endpoint for Binance spot OHLC file
    stats_df = pd.read_csv(url_stats)
    ohlc_df = pd.read_csv(url_ohlc, skiprows=1)  # skip header row and read data

    # change the index of each dataframe to be the date column to join on these values
    stats_df.set_index('date', inplace=True)
    ohlc_df.set_index("Date", inplace=True)

    # drop some columns we dont need/want
    stats_df.drop(columns=['symbol'], inplace=True)
    ohlc_df.drop(columns=['Unix', 'Symbol'], inplace=True)

    ohlc_df.sort_values('Date', ascending=True, inplace=True)  # sort values ascending first
    ohlc_df['log_price'] = np.log(ohlc_df['Close'])  # get log values of closes
    ohlc_df['pct_change'] = ohlc_df['log_price'].diff()  # get % change
    ohlc_df.sort_values('Date', ascending=False, inplace=True)  # resort back to most recent date first
    ohlc_df.drop(columns=['log_price'], inplace=True)  # drop column we dont need

    # join where there is a unix timestamp for spot, reset the index back to int ID afterwards, then rename back to date
    df = stats_df.join(ohlc_df, how='inner').reset_index()
    df.rename(columns={'index': 'date'}, inplace=True)
    df['HighMinusLow'] = df['High'] - df['Low']  # include a term to capture level of daily "variance"

    # drop first and last row from our dataset
    # first row has a NAN target classification cause we shifted rows down by one
    # last row is using a half of day of activity and so we will exclude it
    df.drop(index=df.index[0], axis=0, inplace=True)
    df.drop(index=df.index[-1], axis=0, inplace=True)
    df.drop(columns=['date', 'Open', 'High', 'Low', 'Close'], inplace=True)
    return df


if __name__ == "__main__":
    main()
                                   