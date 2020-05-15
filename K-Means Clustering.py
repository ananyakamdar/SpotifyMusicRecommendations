#importing required libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
pd.options.display.max_columns=None#to display all columns

#importing and pre-processing data
spotify_data = pd.read_excel("spotify playlist and features.xlsx")
features_drop = ["duration_ms","playlists__duration_ms","trackid"]
spotify_data=spotify_data.drop(features_drop,axis=1)#dropping unwanted columns
spotify_data['index'] = np.arange(1, len(spotify_data)+1)#adding an index

scaleCols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
             'key', 'liveness', 'loudness', 'speechiness', 'tempo','valence','mode']#columns we want to scale
scaler = StandardScaler()
scaler.fit(spotify_data.loc[:, scaleCols])
train_scaled = spotify_data.copy()
train_scaled[scaleCols] = scaler.transform(train_scaled[scaleCols])#scale cluster columns
train_scaled = train_scaled.rename(columns = {'acousticness': 'acousticness_scaled',
                                              'danceability': 'danceability_scaled',
                                              'energy': 'energy_scaled',
                                              'instrumentalness': 'instrumentalness_scaled',
                                              'key': 'key_scaled',
                                              'liveness': 'liveness_scaled',
                                              'loudness': 'loudness_scaled',
                                              'speechiness': 'speechiness_scaled',
                                              'tempo': 'tempo_scaled',
                                              'valence': 'valence_scaled',
                                              'mode': 'mode_scaled'})
joinCols =["index","pid","playlists__tracks__artist_name","playlists__tracks__track_name","playlists__tracks__album_name","playlists__tracks__duration_ms","id"]
spotify_data_new = spotify_data.merge(train_scaled, on = joinCols, how = 'outer')#merging the data using outer join
spotify_data1=spotify_data_new.copy()
spotify_data1=spotify_data1['id'].drop_duplicates()#dropping duplicate song entries to get all unique songs
values=spotify_data_new.iloc[spotify_data1.index]
values.to_csv("unique_songs.csv",header=True)#writing unique songs to a csv file

#importing unique songs data to use for clustering
values_new=pd.read_csv("unique_songs.csv")
clusterCols = ['acousticness_scaled','danceability_scaled', 
               'energy_scaled', 'instrumentalness_scaled',
               'key_scaled', 'liveness_scaled', 'loudness_scaled',
               'speechiness_scaled', 'tempo_scaled',
               'valence_scaled', 'mode_scaled']#variables to use to create clusters

#creating clusters
kmeans = KMeans(n_clusters = 5)#no. of clusters we want to create
kmeans.fit(values_new.loc[:, clusterCols])
center = kmeans.cluster_centers_#getting cluster centers
label = kmeans.labels_#getting cluster allocation of each song
values_new['cluster_label'] = label
values_new['cluster_label'] = values_new['cluster_label'] + 1#for cluster number to start from 1 rather than 0
centroids = defaultdict(list)
for col in clusterCols:
    centroids['columns'].append(col)
for a in range(1,len(center)+1):
    for b in range(len(center[0])):
        centroids['c'+ str(a)].append(center[a-1][b])
x=pd.DataFrame(centroids)#creating a table with cluster centroids
prediction_cluster = values_new[['pid','playlists__tracks__artist_name','playlists__tracks__track_name',
                                 'playlists__tracks__duration_ms','playlists__tracks__album_name','cluster_label','id']]
mode_artist = prediction_cluster.groupby(['cluster_label', 'playlists__tracks__artist_name'])['pid'].count().reset_index()
mode_artist = mode_artist.rename(columns = {'pid': 'mode_artist'})
prediction_cluster = prediction_cluster.merge(mode_artist, on = ['cluster_label', 'playlists__tracks__artist_name'])
spotify_data2 = spotify_data.merge(prediction_cluster[['id','cluster_label','mode_artist']], on = ['id'])
spotify_data2.sort_values(by=['pid'])

#creating a function for recommendations
def clustering_recs(pn):
    subset=spotify_data2[spotify_data2.pid==pn]
    clusterlabel = subset.groupby(['cluster_label'])['pid'].count().reset_index().sort_values('pid').tail(3).iloc[2,0]#to obtain cluster containing maximum number of songs from the playlist
    count=0
    artists=subset.groupby(['playlists__tracks__artist_name'])['pid'].count().reset_index().sort_values('pid').tail(5)#to obtain top 5 artists in the playlist
    artistlabel = defaultdict(list)
    for i in range(4,-1,-1):
        artist=artists.iloc[i,0]
        count=count+artists.iloc[i,1]
        artistlabel['names'].append(artist)
        for i in range(0,len(artistlabel['names'])):
            y=spotify_data2[(spotify_data2['cluster_label']==clusterlabel) & (spotify_data2['playlists__tracks__artist_name']==artistlabel['names'][i])]#recommending songs only from the dominant cluster that are by the listener's top 5 artists
            if i==0:
                recommendation=y
            else:
                recommendation=recommendation.append(y)
      
    recommendation=recommendation['id'].drop_duplicates()
    tracks_in_target_playlist = spotify_data2.loc[spotify_data2["pid"] == pn, "id"]
    song_to_recommend = recommendation.loc[~recommendation.isin(tracks_in_target_playlist)]
    final_recommendation=spotify_data2.iloc[song_to_recommend.index]
    final_recommendation=final_recommendation.sample(n=10)
    return final_recommendation[['playlists__tracks__track_name','playlists__tracks__artist_name']]

clustering_recs(2)#enter target playlist number
