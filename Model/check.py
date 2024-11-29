from flask import Flask, render_template, request, jsonify
import pandas as pd
import logging
import pickle
import numpy as np
import re


# Load the pre-trained model, scaler, dataset, and features
with open('Model/book_recommender.pkl', 'rb') as model_file:
    model, scaler, df, features, idlist ,features_scaled= pickle.load(model_file)




def recommend_by_title(input_title):
    # Step 1: Check if the title exists in the DataFrame
    if input_title not in df['title'].values:
        print(f"Title '{input_title}' not found in the dataset.")
        return []
    
    # Step 2: Retrieve the index of the input title
    title_index = df[df['title'] == input_title].index[0]
    
    # Step 3: Create the feature vector for the input title
    feature_vector = features_scaled[title_index].reshape(1, -1)
    
    print("Feature Vector:", feature_vector)
    
    # Step 4: Find nearest neighbors
    distances, indices = model.kneighbors(feature_vector, n_neighbors=30)
    
    # Step 5: Retrieve recommended book titles based on indices
    recommended_books = []
    
    # Include the original book in recommendations
    original_book = df.iloc[title_index][['title', 'authors']].to_dict()
    recommended_books.append(original_book)

    for idx in indices[0]:
        if idx != title_index:  # Avoid recommending the same book again
            recommended_books.append(df.iloc[idx][['title', 'authors']].to_dict())
            if len(recommended_books) >= 30:  # Limit to 30 recommendations total
                break
    
    # Format output for all recommended books with their respective publishers
    formatted_recommendations = [
        f"{row['title']} by {row['authors']}" for row in recommended_books
    ]
    
    return formatted_recommendations
# Example usage:
# recommendations = recommend_by_title("The Changeling Sea")
# recommendations = recommend_by_title("Harry Potter and the Half-Blood Prince (Harry Potter  #6)")
recommendations = recommend_by_title("Harry Potter and the Chamber of Secrets (Harry Potter  #2)")
print(recommendations)