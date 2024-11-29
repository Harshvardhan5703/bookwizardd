import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
import time
import logging


# Set up logging
logging.basicConfig(filename='train_model.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load your dataset
df = pd.read_csv(r'KNNBookRec\dataset\books.csv')

# Data preprocessing and feature engineering
def num_to_obj(x):
    if x > 0 and x <= 1:
        return "between 0 and 1"
    if x > 1 and x <= 2:
        return "between 1 and 2"
    if x > 2 and x <= 3:
        return "between 2 and 3"
    if x > 3 and x <= 4:
        return "between 3 and 4"
    if x > 4 and x <= 5:
        return "between 4 and 5"

df['rating_obj'] = df['average_rating'].apply(num_to_obj)

rating_df = pd.get_dummies(df['rating_obj'])  # One-hot encoding
language_df = pd.get_dummies(df['language_code'])
publisher_df = pd.get_dummies(df['publisher'])
author_df = pd.get_dummies(df['authors'])



features = pd.concat([rating_df, language_df, df['average_rating'],
                      df['ratings_count'], df['title'], publisher_df, author_df], axis=1)
features.set_index('title', inplace=True)


# Separate continuous and categorical features
continuous_features = features[['average_rating', 'ratings_count']]
categorical_features = features.drop(columns=['average_rating', 'ratings_count'])



# Scale the continuous features only
scaler = StandardScaler()
continuous_features_scaled = scaler.fit_transform(continuous_features)

# Combine scaled continuous features with categorical features
# Note: categorical_features is a DataFrame, so we can concatenate it back
features_scaled = np.hstack((continuous_features_scaled, categorical_features.values))
print(features_scaled)

# columns_list = features.columns.tolist()

# # Check for the specific author string
# if "Harry Potter and the Half-Blood Prince (Harry Potter  #6)" in columns_list:
#     print("J.K. Rowling exists in the columns.")
# else:
#     print("J.K. Rowling  does not exist in the columns.")

print("model is starting ")

# Start the timer
start_time = time.time()

# Train the Nearest Neighbors model
model = neighbors.NearestNeighbors(n_neighbors=30, algorithm='ball_tree', metric='euclidean')
model.fit(features_scaled)
dist, idlist = model.kneighbors(features_scaled)

# Calculate training time
end_time = time.time()
training_duration = end_time - start_time

# Log the training duration
logging.info(f'Model trained in {training_duration:.2f} seconds.')

# Save the model and scaler
with open('Model/book_recommender.pkl', 'wb') as model_file:
    pickle.dump((model, scaler, df, features, idlist,features_scaled), model_file)

print("Model trained and saved.")