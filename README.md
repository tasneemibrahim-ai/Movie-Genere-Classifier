# 🎬 Movie Genre Classifier

> An interactive machine learning application that predicts a movie's genre from its plot description, cast, IMDb rating, and Metascore — powered by four classic ML algorithms and wrapped in a clean Tkinter GUI.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Model Accuracy](#model-accuracy)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

Automatically categorizing movies into genres based on textual descriptions and metadata is a classic **Natural Language Processing (NLP) and multi-class classification** problem. This project builds an interactive desktop tool that:

- Accepts movie details as user input (description, actors, rating, metascore)
- Preprocesses text via **TF-IDF** and **Count Vectorization**
- Trains one of four user-selected ML models on the IMDB dataset
- Returns an instant genre prediction through a clean **Tkinter GUI**

---

## Demo

| Field | Example Input |
|---|---|
| **Description** | A young wizard discovers his magical heritage and embarks on an adventure |
| **Actors** | Daniel Radcliffe, Emma Watson, Rupert Grint |
| **Rating** | 7.6 |
| **Metascore** | 65 |
| **Classifier** | Logistic Regression |
| **Prediction** | `Fantasy` |

---

## Dataset

The project uses the **IMDB-Movie-Data.csv** dataset (`data/` folder), containing 1000 movies with the following key columns:

| Column | Type | Description |
|---|---|---|
| `Title` | Text | Movie title |
| `Genre` | Text | Target label (movie genre) |
| `Description` | Text | Short plot summary |
| `Actors` | Text | Comma-separated list of lead actors |
| `Rating` | Float | IMDb user rating (0–10) |
| `Metascore` | Float | Metacritic critic score (0–100) |

---

## Feature Engineering

The raw features are transformed through the following pipeline before being fed to any classifier:

```
Raw Input
   │
   ├─ Description  ──► TF-IDF Vectorizer (top 1000 features, English stop-words removed)
   │
   ├─ Actors       ──► Count Vectorizer  (top 1000 features)
   │
   ├─ Rating       ──► Mean imputation → kept as-is
   │
   └─ Metascore    ──► Mean imputation → kept as-is
                              │
                              ▼
                  Concatenated Feature Matrix (2002 columns)
                              │
                              ▼
                      ML Classifier → Genre
```

- Missing `Rating` / `Metascore` values are filled with the **column mean**.
- Missing `Description` / `Actors` values are replaced with **empty strings**.
- Data is split **80% train / 20% test** with `random_state=42` for reproducibility.

---

## Machine Learning Models

The application supports four classifiers, selectable from a dropdown at runtime:

| Model | Description |
|---|---|
| **Decision Tree** | Builds a tree of feature-based decisions; interpretable but prone to overfitting |
| **K-Nearest Neighbors (KNN)** | Classifies by majority vote of the K closest training samples |
| **Naive Bayes (GaussianNB)** | Probabilistic model assuming feature independence; fast and lightweight |
| **Logistic Regression** | Linear model with softmax for multi-class; generally strong baseline |

---

## Model Accuracy

Accuracy on the 20% hold-out test set (IMDB-Movie-Data.csv, `random_state=42`):

| Model | Test Accuracy |
|---|---|
| Decision Tree | ~35% |
| KNN | ~25% |
| Naive Bayes | ~25% |
| Logistic Regression | ~35% |

> **Note:** Accuracy is limited by the dataset size (~1000 movies) and the fact that many movies carry multiple genres (e.g., `Action,Adventure,Sci-Fi`), making exact string-match classification inherently harder. See [Limitations](#limitations--future-work) for planned improvements.

---

## Project Structure

```
Movie_genere_classifier/
│
├── data/
│   └── IMDB-Movie-Data.csv          # Raw IMDB dataset (1000 movies)
│
├── docs/
│   └── ML Project report.docx       # Full project report
│
├── src/
│   └── movie_genre_classifier.py    # Main app — ML pipeline + Tkinter GUI
│
├── test_runtime.py                  # Headless runtime / sanity-check script
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python **3.8+**
- `tkinter` (bundled with standard CPython on Windows/macOS; on Linux install `python3-tk`)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Movie_genere_classifier.git
cd Movie_genere_classifier

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Verify Installation (optional)

Run the headless sanity-check script to confirm all packages load and all four models train correctly **without** opening the GUI:

```bash
python test_runtime.py
```

Expected output:
```
=== STEP 1: Package imports ===
  pandas: PASS (2.x.x)
  numpy:  PASS (1.x.x)
  scikit-learn: PASS (1.x.x)
  tkinter: PASS

=== STEP 2: Dataset loading ===
  Dataset loaded: PASS  (shape=(1000, 12))
  ...

=== ALL CHECKS COMPLETE ===
```

---

## Usage

Launch the Tkinter GUI:

```bash
python src/movie_genre_classifier.py
```

A window will open with the following fields:

1. **Movie Description** — paste a short plot summary
2. **Actors** — enter actor names separated by commas
3. **Rating** — IMDb rating as a decimal (e.g., `7.5`)
4. **Metascore** — Metacritic score as an integer (e.g., `80`)
5. **Classifier** — choose one from the dropdown menu
6. Click **"Classify Genre"** — the predicted genre appears in a pop-up dialog

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, and feature assembly |
| `numpy` | Numerical operations and array handling |
| `scikit-learn` | TF-IDF, CountVectorizer, classifiers, train/test split, accuracy metrics |
| `tkinter` | Cross-platform desktop GUI |

---

## Limitations & Future Work

| # | Current Limitation | Potential Improvement |
|---|---|---|
| 1 | **Single-label prediction** — predicts one genre even though many movies span multiple | Switch to a **multi-label classifier** (e.g., `MultiOutputClassifier` with One-vs-Rest) |
| 2 | **On-demand training** — the model retrains every time "Classify" is clicked, causing a noticeable delay | Pre-train models and serialize with **`joblib`/`pickle`** for instant inference |
| 3 | **Bag-of-words representations** — TF-IDF and Count Vectorizer lose word order and context | Replace with **Word2Vec**, **GloVe**, or a **BERT**-based sentence embedding |
| 4 | **Small dataset** — only ~1000 movies limits generalization | Extend with larger datasets (e.g., MovieLens, TMDB API) |
| 5 | **No hyperparameter tuning** — default sklearn parameters used | Add `GridSearchCV` / `RandomizedSearchCV` for optimal hyperparameters |

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ for learning Machine Learning & NLP</p>
