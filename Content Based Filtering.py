#importing required libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm#progress meter for loops

#importing and pre-processing the data
spotify_data = pd.read_excel("spotify playlist and features.xlsx")
spotify_data.head()
features_drop = ["info__generated_on","info__slice",'info__version',"pid","playlists__tracks__artist_name","playlists__tracks__track_name","playlists__tracks__album_name","playlists__tracks__duration_ms","playlists__duration_ms","trackid","id","playlists__name","playlists__collaborative","analysis_url","uri","type","track_href"]#list of unimportant columns
train_cleaned = spotify_data.drop(features_drop, axis =1)#dropping unimportant columns i.e axis=1
train_cleaned=train_cleaned.iloc[:,range(11,25)]
train_cleaned.head()
scaler = MinMaxScaler()
scaler.fit(train_cleaned)
train_scaled = scaler.transform(train_cleaned)#standardizing the data

#creating a function for recommendations
def content_recs(playlistnumber):
    cos=np.zeros((250,266386))#creating an empty matrix 
    j=0
    for i in tqdm(range(0,len(train_scaled))):#this loop is used to find the cosine similarity matrix only for the songs in the target playlist 
        if(spotify_data.iloc[i,5]==playlistnumber):
            y=train_scaled[i,].reshape(1,14)#reshaping the row to y 
            x=train_scaled[0:len(train_scaled),]#x is the matrix of all songs
            train_scaled_cosine_matrix = cosine_similarity(x,y)
            m=np.transpose(train_scaled_cosine_matrix)
            cos[j,]=m#storing each row into the cosine similarity matix
            j=j+1
            #the first n rows of the cos matrix are the cosine similarities between songs of the target playlist and the entire data where n is the size of the target playlist
            #the other rows are 0
    cos=np.array(cos)
    cos1=np.array(cos).flatten()#flatten the cosine matrix
    u,index=np.unique(cos1,return_index=True)#get the unique values in ascending order and their index values
    index=index%266386#dividing the index by no. of rows to get remainder, the remainder is the real index of the songs
    unique_candidate_song_sorted =spotify_data['id'][index][::-1].drop_duplicates()#get the unique songs in descending order
    tracks_in_target_playlist = spotify_data.loc[spotify_data["pid"] ==playlistnumber, "id"]#get the tracks in target playlist
    song_to_recommend = np.array(unique_candidate_song_sorted.loc[~unique_candidate_song_sorted.isin(tracks_in_target_playlist)])#get the songs that are not in target paylist in descending order
    song_to_recommend = song_to_recommend[:10]#top 10 songs to recommend
    for i in range(0,len(song_to_recommend)):#create a dataframe of all the recommended songs for the given playlist
        if i==0:
            values=spotify_data.loc[spotify_data["id"] == song_to_recommend[i],["playlists__tracks__track_name","playlists__tracks__artist_name"]]#,"playlists__tracks__album_name","playlists__tracks__duration_ms"]]
            values=pd.DataFrame((values.iloc[0]))
            values=values.T
        else:
            values1=spotify_data.loc[spotify_data["id"] == song_to_recommend[i],["playlists__tracks__track_name","playlists__tracks__artist_name"]]#,"playlists__tracks__album_name","playlists__tracks__duration_ms"]]
            values1=pd.DataFrame((values1.iloc[0]))
            values1=values1.T
            values=values.append(values1)
    return values
content_recs(3067)#enter target playlist id