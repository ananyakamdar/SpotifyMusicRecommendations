# Spotify Music Recommendations
As part of my final college semester, I created a Music Recommendation System using Spotify data for my project. The recommendation feature is an essential component of music streaming applications such as Spotify which helps users easily create playlists which are customized to their tastes. A good recommendation system can not only increase the number of users for an application, but also generate higher revenue through subscriptions if the users are satisfied.

# Dataset Description
The data is obtained from Spotify’s RecSys Challenge held in 2018, however I have used a subset of the data containing 4000 playlists. The data provided in JSON format was converted to CSV files and contained information about the playlist name, tracks in the playlist, number of albums and number of followers. For a more detailed analysis, I extracted more data about the audio features of each song such as tempo, instrumentalness, acousticness, etc. from Spotify’s API. The link below contains the original and the processed datasets.
[Link to Dataset](https://drive.google.com/open?id=1aznWI4NTLebMqwfQPLn6XJDmVX8CO0yh)

# Approach
To generate recommendations for a given playlist, 5 different approaches have been implemented:
**1.	User-User Collaborative Filtering** – This technique focuses on the similarity between the users (playlists) so that songs from one playlist can be recommended to the similar user and vice versa. The K Nearest Neighbors (KNN) algorithm was employed to find similar playlists based on common songs, and the songs were recommended from the nearest neighbors for the target playlist.
**2.	User-Item Collaborative Filtering** – This technique finds similar songs based on the frequency of their occurrence together. The KNN algorithm now finds the nearest neighbors for each song in the target playlist, and the potential recommendations which have the highest frequency are recommended to the target playlist.
**3.	Content-Based Filtering** – This technique uses the audio features of each song in the playlist to find other similar songs using cosine similarity. The songs with highest similarity values to the songs in the target playlist are recommended.
**4.	Clustering** – This technique aims to create groups of songs based on audio features that are dominant in those songs. The K-means clustering algorithm is used to form the clusters. Next, the songs in the target playlist are divided according to the clusters and recommendations are given from the cluster that has most songs of the playlist.
**5.	NLP Neural Network** – This technique comprises a neural network model that makes use of the song names and their position in the playlist. This model, known as the Word to Vector model, creates a one-hot encoding of the songs and uses certain other songs that each song is generally played with (known as context songs) to get recommendations. Songs that have similar context songs and weights are said to be most similar and are hence recommended to the target playlist.
A combination of all or some of these techniques, depending on the data available, can be used by music streaming applications to generate robust and relevant recommendations for greater user satisfaction.

# Evaluation Metrics
Since recommendations are of a subjective nature, there are no evaluation metrics such as Accuracy or MSE that can be used. The only metric in this case can be the user’s satisfaction if the data is available.
