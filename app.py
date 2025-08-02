from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load books data
books_data = pd.read_csv('Books.csv')
selected_features = ['Book-Title', 'Book-Author', 'Publisher']

# Fill missing values and combine features
for feature in selected_features:
    books_data[feature] = books_data[feature].fillna(' ')
combined_features = books_data['Book-Title'] + ' ' + books_data['Book-Author'] + ' ' + books_data['Publisher']

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_form():
    return render_template('input.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input
    title = request.form['title']
    author = request.form['author']
    publisher = request.form['publisher']
    
    # Combine user input
    user_input = title + ' ' + author + ' ' + publisher
    
    # Get all features including user input
    all_book_features = combined_features.tolist()
    all_texts = all_book_features + [user_input]
    
    # Vectorize and calculate similarities
    vectorizer = TfidfVectorizer()
    all_vectors = vectorizer.fit_transform(all_texts)
    book_vectors = all_vectors[:-1]
    user_vector = all_vectors[-1]
    
    similarities = cosine_similarity(user_vector, book_vectors).flatten()
    
    # Get top 5 recommendations
    N = 5
    top_indices = similarities.argsort()[-N:][::-1]
    
    recommendations = []
    for index in top_indices:
        title = books_data.iloc[index]['Book-Title']
        author = books_data.iloc[index]['Book-Author']
        publisher = books_data.iloc[index]['Publisher']
        recommendations.append({'title': title, 'author': author, 'publisher': publisher})
    
    return render_template('output.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)