#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stat
import pylab 
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
#%%
def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,3,1)
    #df[feature].hist(kde=True)
    sns.histplot(df[feature],kde =True)
    plt.subplot(1,3,2)
    stat.probplot(df[feature],dist='norm',plot=pylab)
    plt.subplot(1,3,3)
    sns.boxplot(df[feature])
    plt.show()

#%%
def binf(df,feature):
    binned_data = pd.cut(df[feature],bins=2)

    frequency = binned_data.value_counts()

    frequency.plot(kind='bar')

    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Data Points in Each Bin:{feature}')
    plt.show()

#%%

def kelbow(n_arr):
    SSE = {}
    for k in range(1,15):
        km = KMeans(n_clusters = k, init = 'k-means++', max_iter = 1000)
        km = km.fit(n_arr)
        SSE[k] = km.inertia_
        print(f"Silhouette score with k={k}: {km.inertia_}")
        # plot the graph for SSE and number of clusters
    visualizer = KElbowVisualizer(km, k=(1,15), metric='distortion', timings=False)
    visualizer.fit(n_arr)
    visualizer.poof()
    plt.show()
#%%
def clust_plot(df):
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    plt.title('Customer Segmentation based on Recency and Frequency')
    plt.scatter(df.loc[:,'Recency'], df.loc[:,'Monetary'], c=df['Clusters'], s=50, cmap='Set1', label='Clusters')
    plt.legend()
    plt.show()

# %%
#%%
def outlier(df,f):
    Q1 = df[f].quantile(0.25)
    Q3 = df[f].quantile(0.75)


    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR


    filtered_df = df[(df[f] >= lower_bound) & (df[f] <= upper_bound)]
    return filtered_df