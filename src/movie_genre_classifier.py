import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Load data
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'IMDB-Movie-Data.csv')
data = pd.read_csv(data_path)

# Preprocessing
features = ['Description', 'Actors', 'Rating', 'Metascore']
X = data[features]
y = data['Genre']

# Split data
# Fill missing values in numerical columns with mean
X = X.copy()
# Fill missing values (pandas CoW-safe assignments)
X['Rating'] = X['Rating'].fillna(X['Rating'].mean())
X['Metascore'] = X['Metascore'].fillna(X['Metascore'].mean())
X['Description'] = X['Description'].fillna('')
X['Actors'] = X['Actors'].fillna('')

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Fit and transform TF-IDF vectorizer on the training data
x_train_description_tfidf = tfidf_vectorizer.fit_transform(x_train['Description'])

# Initialize CountVectorizer for actors
count_vectorizer = CountVectorizer(max_features=1000)

# Fit and transform CountVectorizer on the training data
x_train_actors_count = count_vectorizer.fit_transform(x_train['Actors'])

# Concatenate TF-IDF and CountVectorizer output for training data
# Use named token columns from vectorizers for consistent feature names at train & predict time
x_train_processed = pd.concat([
    pd.DataFrame(x_train_description_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()),
    pd.DataFrame(x_train_actors_count.toarray(), columns=count_vectorizer.get_feature_names_out()),
    x_train[['Rating', 'Metascore']].reset_index(drop=True)
], axis=1)

def classify_genre():
    try:
        plot = description_entry.get()
        actors = actors_entry.get()
        rating = float(rating_entry.get())  # Convert rating to float
        metascore = float(metascore_entry.get())  # Convert metascore to float
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for Rating and Metascore.")
        return
        
    if not classifier_combobox.get():
        messagebox.showerror("Input Error", "Please select a classifier.")
        return
    
    # Preprocess the input data
    input_data = pd.DataFrame({'Description': [plot], 'Actors': [actors], 'Rating': [rating], 'Metascore': [metascore]})
    
    # Transform movie description using TF-IDF
    input_description_tfidf = tfidf_vectorizer.transform(input_data['Description'])
    input_description_array = input_description_tfidf.toarray()
    input_description_df = pd.DataFrame(input_description_array, columns=tfidf_vectorizer.get_feature_names_out())
    
    # Transform actors using CountVectorizer
    input_actors_count = count_vectorizer.transform(input_data['Actors'])
    input_actors_array = input_actors_count.toarray()
    input_actors_df = pd.DataFrame(input_actors_array, columns=count_vectorizer.get_feature_names_out())
    
    # Concatenate numerical features with transformed features
    input_data_processed = pd.concat([input_description_df, input_actors_df, input_data[['Rating', 'Metascore']]], axis=1)
    
    # Select the model based on the chosen classifier
    if classifier_combobox.get() == "Decision Tree":
        model = DecisionTreeClassifier()
    elif classifier_combobox.get() == "KNN":
        model = KNeighborsClassifier()
    elif classifier_combobox.get() == "Naive Bayes":
        model = GaussianNB()
    elif classifier_combobox.get() == "Logistic Regression":
        model = LogisticRegression()

    # Train the selected model
    model.fit(x_train_processed, y_train)

    # Predict the genre
    genre = model.predict(input_data_processed)[0]

    # Show the predicted genre in a message box
    messagebox.showinfo("Genre Prediction", f"The predicted genre is: {genre}")

# GUI Setup
root = tk.Tk()
root.title("Movie Genre Classifier")

# Description Entry
description_label = tk.Label(root, text="Enter the movie description:")
description_label.pack()
description_entry = tk.Entry(root, width=50)
description_entry.pack()

# Actors Entry
actors_label = tk.Label(root, text="Enter the actors (comma-separated):")
actors_label.pack()
actors_entry = tk.Entry(root, width=50)
actors_entry.pack()

# Rating Entry
rating_label = tk.Label(root, text="Enter the rating:")
rating_label.pack()
rating_entry = tk.Entry(root, width=50)
rating_entry.pack()

# Metascore Entry
metascore_label = tk.Label(root, text="Enter the metascore:")
metascore_label.pack()
metascore_entry = tk.Entry(root, width=50)
metascore_entry.pack()

# Classifier Selection
classifier_label = tk.Label(root, text="Select the classifier:")
classifier_label.pack()
classifiers = [  "Decision Tree", "KNN", "Naive Bayes", "Logistic Regression"]
classifier_combobox = ttk.Combobox(root, values=classifiers)
classifier_combobox.pack()

# Classify Button
classify_button = tk.Button(root, text="Classify Genre", command=classify_genre)
classify_button.pack()

root.mainloop()