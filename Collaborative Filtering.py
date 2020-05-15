#importing required libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
#importing the data
spotify_data = pd.read_excel("spotify playlist and features.xlsx")
spotify_data.head()
# Create Binary Sparse Matrix
co_mat = pd.crosstab(spotify_data.pid, spotify_data.trackid)
co_mat = co_mat.clip(upper=1)#save values greater than 1 as 1 for the matrix to be binary
assert np.max(co_mat.describe().loc['max']) == 1#checking whether maximum value in the matrix is 0
co_mat_sparse = csr_matrix(co_mat)#save the matrix as a compressed sparse matrix

#Creating the collaborative filter
col_filter = NearestNeighbors(metric='cosine', algorithm='brute')#uses brute-force computations for distance calculation

#Creating a function to get User-User recommendations
def kpredictuseruser(playlist_id):
    
    k = 10#number of recommendations we want
    knnmodel=col_filter.fit(co_mat_sparse)#applying the collaborative filter on the binary sparse matrix
    reference_songs = co_mat.columns.values[co_mat.loc[playlist_id] == 1]#list of songs already in target playlist
    dist, ind = knnmodel.kneighbors(np.array(co_mat.loc[playlist_id]).reshape(1, -1), n_neighbors = 49)#calculating the distances and indices of the nearest points in the population 
    #n_neighbours > k as some of songs in the nearest neighbours might already be in the target playlist
    rec_index = co_mat.index[ind[0]]#nearest playlists from which songs can be recommended
    n_pred = 0#counter
    pred = []
    for i in rec_index:
        potential_recs = co_mat.columns.values[co_mat.loc[i] == 1]#songs from nearest playlists
        for song in potential_recs:
            if song not in reference_songs:#only recommending songs not already in target playlist
                pred.append(song)
                n_pred += 1
                if n_pred == k:
                    break
        if n_pred == k:
            break
    pred=pd.DataFrame(pred,columns={"trackid"})
    all=pred.join(spotify_data.set_index('trackid'),on = 'trackid')#saving recommendations with their metadata
    recommendation=all.drop_duplicates('trackid')[['playlists__tracks__track_name','playlists__tracks__artist_name']]#displaying track name and artist name
    return recommendation

pi = 3067#target playlist index
kpredictuseruser(pi)#list of User to User Collaborative Filtering predictions


#Creating a function to get Item-Item recommendations
co_mat_transpose=co_mat.transpose()#transposing as we now want similarity between songs rather than playlists
def kpredictitemitem(track_id):
    knnmodel=col_filter.fit(co_mat_transpose)#applying the collaborative filter on the binary sparse matrix
    k = 10#number of recommendations we want
    dist, ind = knnmodel.kneighbors(np.array(co_mat_transpose.loc[track_id]).reshape(1, -1), n_neighbors = 49)#calculating the distances and indices of the nearest points in the population
    rec_ind = co_mat_transpose.index[ind[0]]#recommended songs
    n_pred = 0
    pred = []
    for song in rec_ind:
        if song != track_id:#only getting songs not already in target playlist
            pred.append(song)
            n_pred += 1
            if n_pred == k:
                break
        if n_pred == k:
            break
    pred=pd.DataFrame(pred,columns={"trackid"})
    all=pred.join(spotify_data.set_index('trackid'),on = 'trackid')#saving recommendations with their metadata
    recommendation=all.drop_duplicates('trackid')[['playlists__tracks__track_name','playlists__tracks__artist_name']]#displaying track name and artist name
    return recommendation

ti = '0UioblV1x795s55Ur58c6c' #target track
kpredictitemitem(ti)#list of Item to Item Collaborative Filtering predictions


#Creating a function to get User-Item Hybrid recommendations
def kpredictuseritem(playlist_id):
    knnmodel=col_filter.fit(co_mat_transpose)#applying the collaborative filter on the binary sparse matrix
    reference_songs= co_mat.columns.values[co_mat.loc[playlist_id] == 1]#list of songs already in target playlist
    rec_ind=pd.DataFrame()
    for i in reference_songs:
        dist, ind = knnmodel.kneighbors(np.array(co_mat_transpose.loc[i]).reshape(1, -1), n_neighbors = 10)#finds 10 nearest songs for each song in the target playlist
        potential_recs= co_mat_transpose.index[ind[0]]
        rec_ind=rec_ind.append(pd.DataFrame(data=potential_recs))
    preds = pd.DataFrame(np.reshape(rec_ind, (len(rec_ind),1)))#all potential recommendations
    rec=preds.trackid.value_counts()#finding frequency of potential recommendations in descending order
    final=pd.DataFrame(rec).reset_index()
    final=final.iloc[:,0]#saving potential recommendations in descending order of frequency
    k=10#number of recommendations we want
    n_pred = 0
    pred = []
    for song in final:
        if song not in reference_songs:#only getting songs not already in target playlist
            pred.append(song)
            n_pred += 1
            if n_pred == k:
                break
        if n_pred == k:
            break
    pred=pd.DataFrame(pred,columns={"trackid"})
    all=pred.join(spotify_data.set_index('trackid'),on = 'trackid')#saving recommendations with their metadata
    recommendation=all.drop_duplicates('trackid')[['playlists__tracks__track_name','playlists__tracks__artist_name']]#displaying track name and artist name
    return recommendation

pi=3067#target playlist index
kpredictuseritem(pi)#list of User to Item Collaborative Filtering Hybrid predictions