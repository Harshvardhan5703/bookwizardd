import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt



df= pd.read_csv(r"KNNBookRec\dataset\cleaned_data.csv")

df.head(5)

def num_to_obj(x):
    if x >0 and x <=1:
        return "between 0 and 1"
    if x > 1 and x <= 2:
        return "between 1 and 2"
    if x > 2 and x <=3:
        return "between 2 and 3"
    if x >3 and x<=4:
        return "between 3 and 4"
    if x >4 and x<=5:
        return "between 4 and 5"
df['rating_obj'] = df['average_rating'].apply(num_to_obj)


df['rating_obj'].value_counts()


rating_df = pd.get_dummies(df['rating_obj'])     #one hot encoding
rating_df.head()


df.drop(['text_reviews_count'],axis=1,inplace=True)


language_df = pd.get_dummies(df['language_code'])
language_df.head()


publisher_df=pd.get_dummies(df['publisher'])
publisher_df.head(5)


author_df=pd.get_dummies(df['authors'])
author_df.head(5)

features = pd.concat([rating_df,language_df, df['average_rating'],
                    df['ratings_count'], df['title'],publisher_df,author_df], axis = 1)
features.set_index('title', inplace= True)
features.head()

# Plot KDE for each numerical column
features.plot(kind='kde', subplots=True, layout=(3, 3), figsize=(15, 10), sharex=False)
plt.suptitle('KDE Plots of Features')
plt.show()


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

from sklearn import neighbors

model = neighbors.NearestNeighbors(n_neighbors=10, algorithm = 'ball_tree',
                                  metric = 'euclidean')
model.fit(features_scaled)
dist, idlist = model.kneighbors(features_scaled)


df['title'].value_counts()

def normalize_title(title):
    """ Normalize title by removing unwanted characters and converting to lowercase. """
    return re.sub(r'[^\w\s#():]', '', title).lower().strip()

def BookRecommender(book_name):
    book_list_info = []
    
    # Normalize input
    normalized_input = normalize_title(book_name)
    
    # Normalize dataset titles for comparison
    df['normalized_title'] = df['title'].apply(normalize_title)
    
    # Check if the input contains special characters
    if re.search(r'[^\w\s#():]', book_name):  # Check for any non-word character
        # Logic for inputs with symbols
        print("Input contains special characters.")
        book_id_row = df[df['normalized_title'].str.contains(re.escape(normalized_input), na=False, case=False)]
    else:
        # Logic for simple string inputs
        print("Input is a simple string.")
        book_id_row = df[df['normalized_title'].str.contains(re.escape(normalized_input), na=False, case=False)]
        
    if not book_id_row.empty:
        book_id = book_id_row.index[0]
        
        book_list_info.append(f"{df.iloc[book_id].title} by {df.iloc[book_id].authors}")
        
        # Assuming idlist is defined elsewhere in your code
        for newid in idlist[book_id]:
            if newid != book_id:
                recommended_title = df.iloc[newid].title
                recommended_author = df.iloc[newid].authors
                book_list_info.append(f"{recommended_title} by {recommended_author}")
                
    else:
        print(f"Book '{book_name}' not found in the database.")
    
    return book_list_info


def recommend_by_publisher(publisher_name):
    # Normalize the publisher name for consistency
    normalized_publisher = publisher_name.lower().strip()
    
    # Filter books by the specified publisher
    filtered_books = df[df['publisher'].str.lower() == normalized_publisher]
    
    # Check if any books are found for the publisher
    if filtered_books.empty:
        return [f"No books found for publisher '{publisher_name}'."]
    
    # Initialize an empty list to store recommended books
    book_list_info = []

    # For each book in the filtered list, find recommendations
    for index, row in filtered_books.iterrows():
        # Get the title of the book to access features
        book_title = row['title']

        # Check if the book title exists in the features DataFrame
        if book_title in features.index:
            # Get the features of the current book
            book_features = features.loc[book_title].values.reshape(1, -1)

            # Create a DataFrame from the features to preserve column names
            book_features_df = pd.DataFrame(book_features, columns=features.columns)

            # Scale the features for the k-NN model
            book_features_scaled = scaler.transform(book_features_df)

            # Get the k-nearest neighbors for the current book
            dist, idlist = model.kneighbors(book_features_scaled, n_neighbors=11)  # 10 neighbors + 1 for itself

            # Loop through the neighbors and add them to the recommendations
            for newid in idlist[0][1:]:  # Skip the first neighbor as it is the book itself
                recommended_title = features.index[newid]  # Get the title from features index
                book_list_info.append(recommended_title)  # Only append the title

    # Return unique recommendations
    return list(set(book_list_info))  # Return unique recommendations



def recommend_by_author(input_author):
    # Step 1: Check if author exists in DataFrame without normalization
    if input_author not in df['authors'].values:
        print(f"Author '{input_author}' not found in the dataset.")
        return []
    
    # Step 2: Retrieve books by the input author
    books_by_input_author = df[df['authors'] == input_author]
    recommended_books = books_by_input_author[['title', 'authors']].copy()
    
    # Check if we have enough books
    if len(recommended_books) >= 30:
        # Format output for books from the input author
        return [f"{row['title']} by {row['authors']}" for _, row in recommended_books.iterrows()]
    
    # Step 3: If not enough books, find similar authors (without normalization)
    feature_vector = np.zeros(features.shape[1])
    
    # Attempt to get the feature index directly from input author name
    try:
        feature_index = features.columns.get_loc(input_author)
        feature_vector[feature_index] = 1
    except KeyError:
        print(f"Author '{input_author}' not found in features.")
        return []
    
    # Set average values for continuous features
    average_rating = df['average_rating'].mean()  
    ratings_count = df['ratings_count'].mean()  
    
    # Prepare an array with both continuous features for scaling
    continuous_features = np.array([[average_rating, ratings_count]])
    
    # Assign scaled values for continuous features
    scaled_values = scaler.transform(continuous_features)
    
    feature_vector[features.columns.get_loc('average_rating')] = scaled_values[0][0]
    feature_vector[features.columns.get_loc('ratings_count')] = scaled_values[0][1]
    
    # Reshape for model prediction
    feature_vector = feature_vector.reshape(1, -1)
    
    print("Feature Vector:", feature_vector)
    
    # Find nearest neighbors
    distances, indices = model.kneighbors(feature_vector, n_neighbors=30)
    
    # Step 4: Retrieve recommended book titles based on indices of similar authors
    similar_authors_indices = indices[0]
    
    additional_recommendations = []
    
    for idx in similar_authors_indices:
        similar_author = df.iloc[idx]['authors']
        
        if similar_author != input_author:  # Avoid recommending from the same author again
            additional_books = df[df['authors'] == similar_author][['title', 'authors']]
            additional_recommendations.extend(additional_books.to_dict(orient='records'))
            if len(additional_recommendations) >= (30 - len(recommended_books)):
                break
    
    # Combine recommendations
    all_recommended_books = recommended_books.to_dict(orient='records') + additional_recommendations
    
    # Format output for all recommended books with their respective publishers
    formatted_recommendations = [
        f"{row['title']} by {row['authors']}" for row in all_recommended_books
    ]
    
    # Limit to maximum of 25 recommendations and return formatted list
    return formatted_recommendations[:30]



def recommend_by_rating(min_rating):
    filtered_books = df[df['average_rating'] >= min_rating]
    
    top_books = filtered_books.sort_values(by='average_rating', ascending=False).head(125)  # Get top 125 books
    
    book_list_info = [f"{row['title']} by {row['authors']}" for index, row in top_books.iterrows()]
    
    return book_list_info