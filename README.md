# Book-Recommendation-system

📚 Book Recommendation System

Welcome to the Book Recommendation System! 🌟 This dynamic web application is designed to help you discover your next favorite read. With personalized recommendations based on various criteria, you'll find books tailored just for you!

📋 Table of Contents

1.Features
2.Technologies Used
3.Installation
4.Usage
5.How It Works
6.Contributors

✨ Features

1.Search by Book Title: Get recommendations based on your favorite book. 📖
2.Search by Minimum Rating: Discover top-rated books that meet your rating criteria. ⭐
3.Search by Publisher: Find books published by your favorite publishers. 🏢
4.Search by Author: Get recommendations based on your favorite authors. ✍️
5.Live Search Suggestions: As you type a book title, suggestions will appear to help you find the right book quickly. 🔍
6.Responsive Design: The application is designed to work seamlessly on various devices. 📱💻

⚙️ Technologies Used

1.Flask: A lightweight WSGI web application framework in Python.
2.Pandas: For data manipulation and analysis.
3.NumPy: For numerical computations.
4.Scikit-learn: For implementing machine learning algorithms for book recommendations.
5.HTML/CSS: For front-end development and styling.
6.JavaScript: For dynamic content updates and user interaction.

🛠️ Installation
To set up the project locally, follow these steps:
Clone the repository:

git clone https://github.com/Harshvardhan5703/bookwizardd.git
cd bookwizardd

Install the required packages:

pip install -r requirements.txt

Prepare the dataset:
Place your books.csv file in the dataset directory.

Run the application:
python app.py

Access the application in your browser at http://127.0.0.1:5000 after running python app.py

💻 Usage
Navigate to the home page.

Use the search forms to find book recommendations based on title, rating, publisher, or author.

View the recommendations displayed on a new page.

🔍 How It Works

The application uses a collaborative filtering approach with a nearest neighbors algorithm to recommend books based on user input:

Data Preparation: The dataset is cleaned and processed using Pandas, with relevant features extracted for modeling.

Recommendation Logic:
By Title: Normalizes input titles and finds similar titles using a nearest neighbors model.
By Rating: Filters books based on the minimum rating specified by the user.
By Publisher: Retrieves books from a specific publisher and sorts them by average rating.
By Author: Filters books by a specific author and returns recommendations sorted by average rating.
User Interface: Built with HTML/CSS for structure and styling, enhanced with JavaScript for dynamic interactions.

🤝 Contributors
This project is maintained by:
Harshit Mishra 👤
Harshvardhan Singh Shekhawat 👤

