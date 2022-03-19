import requests
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, request, render_template

movies = {}
# loading the data from the csv file to a pandas dataframe
movies_data = pd.read_csv('movies.csv')

# printing the first 5 rows of the dataframe
# print(movies_data.head())

# selecting the relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
# print(selected_features)

# replacing the null values with null string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# combining all the 5 selected features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + \
                    movies_data['cast'] + ' ' + movies_data['director']
# print(combined_features)

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
# print(feature_vectors)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

# Initialize the flask App
app = Flask(__name__)


# default page of our web-app
@app.route('/')
def home():
    list_of_all_titles = movies_data['title'].tolist()

    i = 0
    for movie in list_of_all_titles:
        if i < 36:
            response = requests.get(f"http://www.omdbapi.com/?apikey=d2557d66&t={movie}")
            movie_data = response.json()
            movies[movie] = movie_data['Poster']
            # print(movies)
            i += 1

    return render_template('index.html', movie_list=movies)


# To use the predict button in our web-app
@app.route('/recommend', methods=['POST'])
def recommend():
    got_movies = {}
    data = request.form
    movie_name = data['movie']

    # creating a list with all the movie names given in the dataset
    list_of_all_titles = movies_data['title'].tolist()
    # print(list_of_all_titles)

    # finding the close match for the movie name given by the user
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    # print(find_close_match)

    close_match = find_close_match[0]
    # print(close_match)

    # finding the index of the movie with title
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    # print(index_of_the_movie)

    # getting a list of similar movies
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    # print(similarity_score)

    # sorting the movies based on their similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    # print(sorted_similar_movies)

    i = 0

    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if i < 30:
            response = requests.get(f"http://www.omdbapi.com/?apikey=d2557d66&t={title_from_index}")
            movie_data = response.json()
            try:
                got_movies[title_from_index] = movie_data['Poster']
            except:
                got_movies[title_from_index] = "https://i.pinimg.com/736x/fc/eb/df/fcebdf9e34ad5d5f1d8f728f781a00ac.jpg"
            # print(movies)
            i += 1

    return render_template('movies.html', movie_list=got_movies)


if __name__ == "__main__":
    app.run(debug=True)
