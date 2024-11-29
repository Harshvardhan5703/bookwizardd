from flask import Flask, render_template, request, jsonify
import pandas as pd
import logging
import re
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
import numpy as np 
app = Flask(__name__, static_url_path='/static')

with open('Model/book_recommender.pkl', 'rb') as model_file:
    model, scaler, df, features, idlist,features_scaled = pickle.load(model_file)

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def normalize_title(title):
    """ Normalize title by removing unwanted characters and converting to lowercase. """
    return re.sub(r'[^\w\s#():]', '', title).lower().strip()

def BookRecommender(input_title):
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


def recommend_by_publisher(input_publisher):
    # Check if publisher exists in DataFrame columns
    if input_publisher not in features.columns:
        print(f"Publisher '{input_publisher}' not found in the dataset.")
        return []
    
    # Step 1: Retrieve books by the input publisher
    books_by_input_publisher = df[df['publisher'] == input_publisher]
    recommended_books = books_by_input_publisher[['title', 'authors', 'publisher']].copy()
    
    # Check if we have enough books
    if len(recommended_books) >= 30:
        # Format output for books from the input publisher
        return [f"{row['title']} by {row['authors']}, ({row['publisher']})" for _, row in recommended_books.iterrows()]
    
    # Step 2: If not enough books, find similar publishers
    feature_vector = np.zeros(features.shape[1])
    
    # Set the corresponding index for the input publisher to 1
    feature_vector[features.columns.get_loc(input_publisher)] = 1
    
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
    
    # Step 3: Retrieve recommended book titles based on indices of similar publishers
    similar_publishers_indices = indices[0]
    
    # Get unique publishers from the similar publishers' indices
    similar_publishers = df.iloc[similar_publishers_indices]['publisher'].unique()
    
    additional_recommendations = []
    
    for publisher in similar_publishers:
        if publisher != input_publisher:  # Avoid recommending from the same publisher again
            additional_books = df[df['publisher'] == publisher][['title', 'authors', 'publisher']]
            additional_recommendations.extend(additional_books.to_dict(orient='records'))
            if len(additional_recommendations) >= (30 - len(recommended_books)):
                break
    
    # Combine recommendations
    all_recommended_books = recommended_books.to_dict(orient='records') + additional_recommendations
    
    # Format output for all recommended books with their respective publishers
    formatted_recommendations = [
        f"{row['title']} by {row['authors']}, ({row['publisher']})" for row in all_recommended_books
    ]
    
    # Limit to maximum of 25 recommendations and return formatted list
    return formatted_recommendations[:30]



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



def recommend_books_by_rating(input_rating):
    # Filter books based on the input rating
    filtered_books = df[df['average_rating'] >= input_rating]
    
    # Sort the filtered books by average rating in descending order
    sorted_books = filtered_books.sort_values(by='average_rating', ascending=False)
    
    # Get the top 100 books or all available if fewer than 100
    recommended_books = sorted_books.head(100)
    
    # Format the output
    book_list_info = [f"{row['title']} by {row['authors']} " for index, row in recommended_books.iterrows()]
    
    return book_list_info


@app.route('/')
def index():
    logger.info("Accessed the index page.")
    return render_template('index.html', title_recommendations=[], rating_recommendations=[], publisher_recommendations=[], author_recommendations=[])

@app.route('/recommend_by_book', methods=['POST'])
def recommend_by_book():
    book_name = request.form.get('book_name', '').strip()
    logger.info(f"Received book recommendation request for: {book_name}")
    title_recommendations = BookRecommender(book_name) if book_name else []
    
    # Get the current page from the request
    current_page = int(request.args.get('page', 1))
    items_per_page = 30
    total_recommendations = len(title_recommendations)
    total_pages = (total_recommendations + items_per_page - 1) // items_per_page
    
    # Calculate the start and end index for slicing
    start_index = (current_page - 1) * items_per_page
    end_index = start_index + items_per_page
    paginated_recommendations = title_recommendations[start_index:end_index]  # Get recommendations for the current page
    
    if title_recommendations:
        logger.info(f"Recommendations found for {book_name}: {title_recommendations}")
    else:
        logger.warning("No recommendations found for the given book name.")
        paginated_recommendations = []
      
        
    return render_template('recommendations.html',
                           title_recommendations=paginated_recommendations,
                           rating_recommendations=[], 
                           publisher_recommendations=[],
                           author_recommendations=[],
                           current_page=current_page,
                           total_pages=total_pages,
                           input_book=book_name
                           )
    
    

@app.route('/recommend_by_publisher', methods=['GET', 'POST'])
def recommend_by_publisher_route():
    publisher = request.form.get('publisher')
    logger.info(f"Received publisher recommendation request for: {publisher}")
    
    # Get recommendations based on the publisher
    publisher_recommendations = recommend_by_publisher(publisher) if publisher else []
    
    current_page = 1  
    total_pages = 1  

    if publisher_recommendations:
        logger.info(f"Recommendations found for publisher {publisher}: {publisher_recommendations}")
        items_per_page = 30
        total_recommendations = len(publisher_recommendations)
        total_pages = (total_recommendations + items_per_page - 1) // items_per_page
        paginated_recommendations = publisher_recommendations[:items_per_page]
        
    else:
        # No recommendations found
        logger.warning("No recommendations found for the given publisher.")
        paginated_recommendations = []
        

    return render_template('recommendations.html',
                           title_recommendations=[], 
                           rating_recommendations=[], 
                           author_recommendations=[],
                           publisher_recommendations=paginated_recommendations,
                           current_page=current_page,
                           total_pages=total_pages,
                           input_publisher=publisher,  # Pass the input publisher name
                           )  # Pass message if exists
    
@app.route('/recommend_by_rating', methods=['GET', 'POST'])
def recommend_by_rating_route():
    if request.method == 'POST':
        rating = request.form.get('rating')
        min_rating = float(rating) if rating else 0.0
        logger.info(f"Received rating recommendation request for minimum rating: {min_rating}")
        
        # Get recommendations using the new function
        all_recommendations = recommend_books_by_rating(min_rating)
        
        page = request.args.get('page', 1, type=int)
        per_page = 25
        total_books = len(all_recommendations)
        start = (page - 1) * per_page
        end = start + per_page
        recommendations_to_display = all_recommendations[start:end]
        total_pages = (total_books // per_page) + (1 if total_books % per_page > 0 else 0)

        if recommendations_to_display:
            logger.info(f"Recommendations found: {recommendations_to_display}")
        else:
            logger.warning("No recommendations found for the given minimum rating.")
        
        return render_template('recommendations.html', 
                               title_recommendations=[], 
                               rating_recommendations=recommendations_to_display, 
                               author_recommendations=[],
                               publisher_recommendations=[], 
                               current_page=page, 
                               total_pages=total_pages)
    
    else:
        min_rating = request.args.get('rating', type=float, default=0.0)
        
        # Get recommendations using the new function
        all_recommendations = recommend_books_by_rating(min_rating)
        
        page = request.args.get('page', 1, type=int)
        per_page = 25
        total_books = len(all_recommendations)
        start = (page - 1) * per_page
        end = start + per_page
        recommendations_to_display = all_recommendations[start:end]
        total_pages = (total_books // per_page) + (1 if total_books % per_page > 0 else 0)

        return render_template('recommendations.html',
                               title_recommendations=[],
                               rating_recommendations=recommendations_to_display,
                               publisher_recommendations=[],
                               current_page=page,
                               total_pages=total_pages)

@app.route('/recommend_by_author', methods=['GET','POST'])
def recommend_by_author_route():
    author = request.form.get('author', '').strip()
    logger.info(f"Received author recommendation request for: {author}")
    author_recommendations = recommend_by_author(author) if author else []
    current_page = 1
    total_pages = 1

    if author_recommendations:
        logger.info(f"Recommendations found for author {author}: {author_recommendations}")
        items_per_page = 30
        total_recommendations = len(author_recommendations)
        total_pages = (total_recommendations + items_per_page - 1) // items_per_page
        paginated_recommendations = author_recommendations[:items_per_page]
    else:
        logger.warning("No recommendations found for the given author.")
        paginated_recommendations = []

    return render_template('recommendations.html',
                           title_recommendations=[], 
                           rating_recommendations=[], 
                           publisher_recommendations=[], 
                           author_recommendations=paginated_recommendations,
                           current_page=current_page,
                           total_pages=total_pages,
                           author=author)

@app.route('/search_books')
def search_books():
    query = request.args.get('query', '').strip()
    normalized_query = normalize_title(query)
    matching_books = df[df['title'].apply(normalize_title).str.contains(normalized_query, na=False)]
    book_titles = matching_books['title'].tolist()
    return jsonify(book_titles)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)