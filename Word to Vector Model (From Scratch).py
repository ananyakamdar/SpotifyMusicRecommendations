#importing required libraries
import numpy as np
import pandas as pd
from collections import defaultdict

#importing and pre-processing the data
data=pd.read_excel("spotify playlist and features.xlsx")
def prep(trackname):
    trackname=trackname.lower()
    trackname=trackname.split("(",1)[0]
    trackname=trackname.strip()
    return trackname
data["songname"]=data["playlists__tracks__track_name"].copy().apply(prep)
data["artistname"]=data["playlists__tracks__artist_name"].map(lambda x:x.lower())
data["songartist"]=data["songname"]+"-"+data["artistname"]
def playlist_format(playlists):
    documents=[]
    for index,row in playlists.iterrows():
        preprocessed=row["songartist"]
        documents.append(preprocessed)
    return documents
new=playlist_format(data)

############# creating a word2vec class and eventually the model ##############
class word2vec():
    def __init__(self):
        self.n = settings['n']#no. of neurons in hidden layer
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']#no. of context words(songs) to consider
    
    def generate_training_data(self, settings, corpus):
        training_data=[]
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word]+=1
        self.v_count = len(word_counts.keys())#number of songs
        self.words_list = list(word_counts.keys())#list of song names
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))	
        for sentence in corpus:
            sent_len=len(sentence)#number of songs in that playlist
            for i, word in enumerate(sentence):
                w_target = self.word2onehot(sentence[i])#get one-hot representation of the song
                w_context = []
                for j in range(i - self.window, i + self.window+1):#get one-hot representation of the context songs for that song
                    if j != i and j <= sent_len-1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])#saving vector representation of each song and its context songs
        return np.array(training_data)
    
    def word2onehot(self, word):#creating one-hot representation of the song
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec
    
    def train(self, training_data):
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))#initialise weight matrix with random values
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))#initialise weight matrix with random values
        for i in range(self.epochs):
			# Intialise loss to 0
            self.loss = 0
            for w_t, w_c in training_data:
                y_pred,h,u = self.forward_pass(w_t)#input the vector representation of the song
                EI = np.sum([np.subtract(y_pred,word) for word in w_c],axis=0)#error in prediction of each context word (song)
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))#calculating loss for each epoch
            print('Epoch:', i, "Loss:", self.loss)
    
    def forward_pass(self, x):
        h = np.dot(x, self.w1)
        u = np.dot(h, self.w2)
        y_c = self.softmax(u)#output on activation
        return y_c, h, u
    
    def softmax(self, x):#activation function
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        self.w1 = self.w1 - (self.lr * dl_dw1)#updating the weight matrix
        self.w2 = self.w2 - (self.lr * dl_dw2)#updating the weight matrix

	#getting vector from word (song)
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

	#returns nearest word(s)
    def vec_sim(self, word, top_n):
        v_w1 = self.word_vec(word)
        word_sim = {}
        for i in range(self.v_count):
			# Find the similary score for each word in vocab
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den
            word = self.index_word[i]
            word_sim[word] = theta
        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
        final= []
        for word, sim in words_sorted[:top_n]:
            wor=word
            final.append(wor)
            print(word, sim)
        return final

###############################################################################

#setting the parameters for the model
settings = {
	'window_size': 2,	   #context window +- center song
	'n': 30,			   #size of hidden layer
	'epochs': 100,		   #number of training epochs
	'learning_rate': 0.01  #learning rate
}

#initialise object
w2v=word2vec()

#array with one-hot representation for [target song, context songs]
training_data = w2v.generate_training_data(settings, new)
training_data.head()

#training the model
w2v.train(training_data)

#creating a function to get recommendations
def w2v_rec(playlistnumber):
    tracks_in_target_playlist = data.loc[data["playlistnumber"] ==playlistnumber, "songartist"]
    z=[]
    for word in tracks_in_target_playlist:
        z.append(w2v.vec_sim(word, 2))
    recommend=np.array(z)
    recommend=pd.DataFrame(recommend)
    recommendation=recommend.iloc[:,1]
    recommendation= recommendation.loc[~recommendation.isin(tracks_in_target_playlist)]
    song_to_recommend = data.loc[data["songartist"].isin(recommendation)]     
    song_to_recommend= song_to_recommend.drop_duplicates(subset=['songname'])
    final_recommendation=song_to_recommend[["songname","artistname"]]
    return final_recommendation

w2v_rec(13)#target playlist number