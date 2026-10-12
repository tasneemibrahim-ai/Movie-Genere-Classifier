import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# --- Step 1: verify required packages ---
print("=== STEP 1: Package imports ===")
try:
    import pandas as pd
    print(f"  pandas: PASS ({pd.__version__})")
except ImportError as e:
    print(f"  pandas: FAIL ({e})")
    sys.exit(1)

try:
    import numpy as np
    print(f"  numpy: PASS ({np.__version__})")
except ImportError as e:
    print(f"  numpy: FAIL ({e})")
    sys.exit(1)

try:
    import sklearn
    print(f"  scikit-learn: PASS ({sklearn.__version__})")
except ImportError as e:
    print(f"  scikit-learn: FAIL ({e})")
    sys.exit(1)

try:
    import tkinter
    import tkinter.ttk
    print(f"  tkinter: PASS")
    print(f"  tkinter.ttk: PASS")
except ImportError as e:
    print(f"  tkinter/ttk: FAIL ({e})")

# --- Step 2: load dataset directly ---
print("\n=== STEP 2: Dataset loading ===")
import os
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'IMDB-Movie-Data.csv')
try:
    data = pd.read_csv(data_path)
    print(f"  Dataset loaded: PASS  (shape={data.shape})")
    print(f"  Columns: {list(data.columns)}")
    print(f"  Target 'Genre' sample: {data['Genre'].iloc[0]}")
except Exception as e:
    print(f"  Dataset loaded: FAIL ({e})")
    sys.exit(1)

# --- Step 3: preprocessing pipeline ---
print("\n=== STEP 3: Preprocessing pipeline ===")
try:
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    features = ['Description', 'Actors', 'Rating', 'Metascore']
    X = data[features].copy()
    y = data['Genre']

    X['Rating'].fillna(X['Rating'].mean(), inplace=True)
    X['Metascore'].fillna(X['Metascore'].mean(), inplace=True)
    X['Description'].fillna('', inplace=True)
    X['Actors'].fillna('', inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    x_train_description_tfidf = tfidf_vectorizer.fit_transform(x_train['Description'])

    count_vectorizer = CountVectorizer(max_features=1000)
    x_train_actors_count = count_vectorizer.fit_transform(x_train['Actors'])

    x_train_processed = pd.concat([
        pd.DataFrame(x_train_description_tfidf.toarray()),
        pd.DataFrame(x_train_actors_count.toarray()),
        x_train[['Rating', 'Metascore']].reset_index(drop=True)
    ], axis=1)

    print(f"  Train shape: {x_train_processed.shape}  PASS")

    # transform test input same way
    x_test_desc_tfidf = tfidf_vectorizer.transform(x_test['Description'])
    x_test_actors_count = count_vectorizer.transform(x_test['Actors'])
    x_test_processed = pd.concat([
        pd.DataFrame(x_test_desc_tfidf.toarray()),
        pd.DataFrame(x_test_actors_count.toarray()),
        x_test[['Rating', 'Metascore']].reset_index(drop=True)
    ], axis=1)

    print(f"  Test shape:  {x_test_processed.shape}  PASS")
except Exception as e:
    print(f"  Preprocessing: FAIL ({e})")
    sys.exit(1)

# --- Step 4: train and predict with each classifier ---
print("\n=== STEP 4: Model training & prediction ===")
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# one sample input from test set
sample_input = x_test_processed.iloc[[0]]

classifiers = {
    'Decision Tree':     DecisionTreeClassifier(),
    'KNN':               KNeighborsClassifier(),
    'Naive Bayes':       GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
}

for name, model in classifiers.items():
    try:
        model.fit(x_train_processed, y_train)
        prediction = model.predict(sample_input)[0]
        acc = accuracy_score(y_test, model.predict(x_test_processed))
        print(f"  {name}: PASS  (sample prediction='{prediction}', test_accuracy={acc:.3f})")
    except Exception as e:
        print(f"  {name}: FAIL ({e})")

# --- Step 5: GUI widget import check (no display needed) ---
print("\n=== STEP 5: GUI import verification ===")
try:
    from tkinter import ttk, messagebox
    print("  from tkinter import ttk: PASS")
    print("  from tkinter import messagebox: PASS")
except Exception as e:
    print(f"  tkinter imports: FAIL ({e})")

print("\n=== ALL CHECKS COMPLETE ===")
