#importing required libraries
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

#importing and pre-processing data
data=pd.read_excel("spotify playlist and features.xlsx")
def prep(trackname):
    trackname=trackname.lower()#converting all characters to lower case
    trackname=trackname.split("(",1)[0]#splitting the string when a "(" is encountered in track name
    trackname=trackname.strip()#removing all leading and trailing whitespaces
    return trackname
data["songname"]=data["playlists__tracks__track_name"].copy().apply(prep)
data["artistname"]=data["playlists__tracks__artist_name"]
data["songartist"]=data["songname"]+"-"+data["artistname"]
def playlist_format(playlists):#formatting all playlists as a list which contains lists of the the songs in each playlist
    documents=[]
    for i in range(0,max(playlists["pid"])+1):
        x=[]
        dataset=playlists.loc[playlists["pid"]==i]
        for i in range(0,len(dataset)):
            preprocessed=dataset.iloc[i,44]
            x.append(preprocessed)
        documents.append(x)
    return documents
new=playlist_format(data)

model = Word2Vec(new, min_count=1,size= 150,workers=3, window =2, sg = 1)#training the word2vec model
print(model)
words = list(model.wv.vocab)
#print(words) #run this if a list of unique songs is required

#creating a function to get recommendations
def w2v_recs(playlistnumber):
    tracks_in_target_playlist = data.loc[data["pid"]==playlistnumber, "songartist"]
    z=[]
    for word in tracks_in_target_playlist:
        z.append(model.wv.similar_by_word(word, topn=2, restrict_vocab=None))#getting top 2 similar songs for each song in the target playlist
    recommend=np.array(z).reshape((-1, 2))
    recommend=pd.DataFrame(recommend)
    recommendation=recommend.iloc[:,0]
    recommendation= recommendation.loc[~recommendation.isin(tracks_in_target_playlist)]
    song_to_recommend = data.loc[data["songartist"].isin(recommendation)]    
    song_to_recommend= song_to_recommend.drop_duplicates(subset=['songname'])
    final_recommendation=song_to_recommend[["songname","artistname"]]
    return final_recommendation

w2v_recs(13)#input the target playlist number