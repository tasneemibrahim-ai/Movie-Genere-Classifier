# 🎬 Movie Genre Classifier

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![UI Framework](https://img.shields.io/badge/GUI-Tkinter-green.svg)](https://docs.python.org/3/library/tkinter.html)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

An interactive Natural Language Processing (NLP) and Machine Learning application that predicts a movie's primary genre based on its plot description, lead cast, IMDb rating, and Metascore. Powered by four classic scikit-learn classifiers and packaged inside a lightweight, cross-platform Tkinter GUI desktop interface.

---

## 📌 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Dataset Overview](#-dataset-overview)
- [Feature Engineering Pipeline](#-feature-engineering-pipeline)
- [Machine Learning Classifiers](#-machine-learning-classifiers)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Project Structure](#-project-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [GUI Walkthrough & Usage](#-gui-walkthrough--usage)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [Future Enhancements & Roadmap](#-future-enhancements--roadmap)
- [License & Acknowledgments](#-license--acknowledgments)

---

## ✨ Key Features

- **Multi-Algorithm Engine:** Choose between four distinct machine learning algorithms (**Logistic Regression**, **Decision Trees**, **K-Nearest Neighbors**, and **Naive Bayes**) at runtime.
- **Dual Text Feature Extraction:** Uses **TF-IDF Vectorization** for plot summaries and **Count Vectorization** for lead actor names to extract meaningful textual signals.
- **Robust Feature Fusion:** Dynamically combines text sparse matrices with mean-imputed numerical ratings (**IMDb Rating** & **Metascore**) into a cohesive 2,002-dimensional feature space.
- **Interactive Desktop UI:** Built using Python's native **Tkinter** framework for instant input entry, algorithm switching, and pop-up prediction visualizers.
- **Zero External API Dependencies:** Runs completely offline without requiring cloud services or external API keys.
- **Headless Verification Script:** Includes an automated diagnostic tool ([`test_runtime.py`](file:///e:/tasnim/Tasneem_Career_AI/Github_Projects/completed%20projects/Movie_genere_classifier/test_runtime.py)) to verify data integrity, vectorizer shapes, and model execution before launching the GUI.

---

## 🏗 System Architecture

```
                                  ┌────────────────────────────────┐
                                  │           USER INPUT           │
                                  ├────────────────────────────────┤
                                  │ • Plot Description (Text)      │
                                  │ • Cast / Actors (Text)         │
                                  │ • IMDb Rating (Float)          │
                                  │ • Metascore (Float)            │
                                  └───────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌──────────────────────────────────┐
                                 │   PREPROCESSING & VECTORIZATION  │
                                 ├──────────────────────────────────┤
                                 │ Description ──► TF-IDF (1000)    │
                                 │ Actors      ──► CountVec (1000)  │
                                 │ Rating      ──► Mean Imputed (1) │
                                 │ Metascore   ──► Mean Imputed (1) │
                                 └───────────────┬──────────────────┘
                                                 │
                                                 ▼
                                 ┌──────────────────────────────────┐
                                 │     CONCATENATED FEATURE MATRIX  │
                                 │         (2,002 Columns)          │
                                 └───────────────┬──────────────────┘
                                                 │
                                                 ▼
                                 ┌──────────────────────────────────┐
                                 │       CLASSIFIER SELECTION       │
                                 ├──────────────────────────────────┤
                                 │ [Logistic Regression / Decision  │
                                 │  Tree / KNN / Naive Bayes]       │
                                 └───────────────┬──────────────────┘
                                                 │
                                                 ▼
                                 ┌──────────────────────────────────┐
                                 │       PREDICTED MOVIE GENRE      │
                                 └──────────────────────────────────┘
```

---

## 📊 Dataset Overview

The dataset used for model training and validation is [`IMDB-Movie-Data.csv`](file:///e:/tasnim/Tasneem_Career_AI/Github_Projects/completed%20projects/Movie_genere_classifier/data/IMDB-Movie-Data.csv), located inside the `data/` folder. It contains curated metadata for **1,000 top-rated movies** from the IMDb database.

### Key Attributes Used in Feature Matrix

| Feature | Data Type | Preprocessing Strategy | Feature Contribution |
|---|---|---|---|
| `Description` | String (Text) | Lowercased, English stop-words removed, top 1,000 TF-IDF n-grams | `[1,000]` TF-IDF features |
| `Actors` | String (Text) | Tokenized comma-separated names, top 1,000 Count-vectorized terms | `[1,000]` Frequency features |
| `Rating` | Float (0.0 - 10.0) | Column mean imputation for missing values | `[1]` Numerical feature |
| `Metascore` | Float (0.0 - 100.0) | Column mean imputation for missing values | `[1]` Numerical feature |
| **`Genre` (Target)** | String | Multi-class label representing combinations or primary genre types | Ground truth label `y` |

---

## 🛠 Feature Engineering Pipeline

Combining textual descriptions, categorical metadata, and scalar ratings requires a hybrid feature assembly strategy:

1. **Text Plot Vectorization (TF-IDF):**
   - Extracts term frequency-inverse document frequency vectors from the `Description` field.
   - Standard English stop-words (`the`, `is`, `at`, etc.) are stripped.
   - Constrained to `max_features=1000` to capture dominant story themes while controlling model dimensionality.

2. **Actor Tokenization (CountVectorizer):**
   - Processes lead actor names in `Actors`.
   - Tokenized with `max_features=1000` to establish statistical correlation between specific actors and movie genres.

3. **Numerical Imputation:**
   - Any missing values in `Rating` or `Metascore` are imputed using pandas mean calculation, preventing data leakage or runtime execution errors during matrix matrix concatenation.

4. **Matrix Concatenation:**
   - Text matrices and numerical arrays are concatenated horizontally to form a feature vector $X \in \mathbb{R}^{N \times 2002}$:
   $$\text{Features} = \text{TF-IDF}(\text{Description})_{1000} \;\Vert\; \text{CountVec}(\text{Actors})_{1000} \;\Vert\; [\text{Rating}]_1 \;\Vert\; [\text{Metascore}]_1$$

---

## 🤖 Machine Learning Classifiers

The application allows users to experiment with four algorithms, demonstrating performance tradeoffs across sparse text data:

### 1. Decision Tree Classifier (`DecisionTreeClassifier`)
- **Concept:** Builds a hierarchical binary decision tree splitting on key word frequencies and ratings.
- **Characteristics:** Highly interpretable splits, non-linear relationships, but vulnerable to overfitting on high-dimensional text vectors.

### 2. K-Nearest Neighbors (`KNeighborsClassifier`)
- **Concept:** Classifies new movie entries by measuring Euclidean distance to the $K$ closest training samples ($K=5$ default).
- **Characteristics:** Non-parametric, distance-based classification. Performance relies heavily on feature scale alignment.

### 3. Naive Bayes (`GaussianNB`)
- **Concept:** Probabilistic classifier applying Bayes' theorem with Gaussian distribution assumptions on continuous term frequencies.
- **Characteristics:** Fast computation and lightweight memory usage. Serves as a standard NLP benchmark baseline.

### 4. Logistic Regression (`LogisticRegression`)
- **Concept:** Fits a multi-class softmax regression model mapping high-dimensional vectors to genre class probabilities.
- **Characteristics:** Robust baseline for sparse text classification with standard $L_2$ regularization handling collinearity across top TF-IDF tokens.

---

## 📈 Model Performance & Evaluation

Models are trained on an **80% train / 20% test split** (`random_state=42`) using the [`IMDB-Movie-Data.csv`](file:///e:/tasnim/Tasneem_Career_AI/Github_Projects/completed%20projects/Movie_genere_classifier/data/IMDB-Movie-Data.csv) dataset.

### Test Set Accuracy Comparison

| Classifier | Hold-out Test Accuracy | Strengths | Limitations |
|---|---|---|---|
| **Logistic Regression** | **~35%** | Strong linear separator for text n-grams | Single-label limitation on multi-genre target strings |
| **Decision Tree** | **~35%** | Captures non-linear feature interactions | High variance, prone to leaf node overfitting |
| **Naive Bayes (GaussianNB)** | **~25%** | Fast inference time | Assumes independence across correlated vocabulary |
| **K-Nearest Neighbors (KNN)** | **~25%** | No training overhead | Sensitive to high-dimensional feature sparsity |

> 💡 **Analytical Note on Accuracy:**  
> In the raw IMDb dataset, the `Genre` target column often contains multi-genre combinations (e.g., `Action,Adventure,Sci-Fi` or `Drama,Romance`). Because the model treats exact string matches as distinct categorical classes, predicting an exact combination string presents a high-cardinality classification challenge. See [Future Enhancements](#-future-enhancements--roadmap) for multi-label binarization plans.

---

## 📁 Project Structure

```
Movie_genere_classifier/
│
├── data/
│   └── IMDB-Movie-Data.csv          # IMDb dataset (1,000 movies, 12 features)
│
├── docs/
│   └── ML Project report.docx       # Project documentation & analysis report
│
├── src/
│   └── movie_genre_classifier.py    # Main Application (ML Pipeline + Tkinter GUI)
│
├── test_runtime.py                  # Headless verification & diagnostic script
├── requirements.txt                 # Core dependencies (pandas, numpy, scikit-learn)
├── .gitignore                       # standard git ignore file
└── README.md                        # Documentation & setup guide
```

---

## ⚙️ Installation & Quickstart

### 1. Prerequisites

- **Python 3.8+** installed on your system.
- **Tkinter package:**
  - **Windows & macOS:** Bundled automatically with Python standard library.
  - **Linux (Debian/Ubuntu):** May require system package installation:
    ```bash
    sudo apt-get install python3-tk
    ```

### 2. Clone Repository & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/Movie_genere_classifier.git
cd Movie_genere_classifier

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate

# On Windows (PowerShell / CMD):
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run System Verification Script

Before launching the GUI, run the diagnostic script to verify dependencies, dataset pathing, vectorization, and model training:

```bash
python test_runtime.py
```

*Expected Verification Output:*
```text
=== STEP 1: Package imports ===
  pandas: PASS (2.2.3)
  numpy: PASS (2.2.1)
  scikit-learn: PASS (1.6.1)
  tkinter: PASS
  tkinter.ttk: PASS

=== STEP 2: Dataset loading ===
  Dataset loaded: PASS  (shape=(1000, 12))
  Target 'Genre' sample: Action,Adventure,Sci-Fi

=== STEP 3: Preprocessing pipeline ===
  Train shape: (800, 2002)  PASS
  Test shape:  (200, 2002)  PASS

=== STEP 4: Model training & prediction ===
  Decision Tree: PASS  (sample prediction='Action,Adventure,Sci-Fi', test_accuracy=0.345)
  KNN: PASS  (sample prediction='Comedy,Drama', test_accuracy=0.255)
  Naive Bayes: PASS  (sample prediction='Action,Adventure,Sci-Fi', test_accuracy=0.250)
  Logistic Regression: PASS  (sample prediction='Action,Adventure,Sci-Fi', test_accuracy=0.355)

=== ALL CHECKS COMPLETE ===
```

---

## 💻 GUI Walkthrough & Usage

Launch the interactive GUI application with:

```bash
python src/movie_genre_classifier.py
```

### Application Interface Fields

1. **Movie Description:** Enter plot keywords or a short summary (e.g., *"A young wizard attends a school of magic and faces a dark lord"*).
2. **Actors:** Input comma-separated lead actor names (e.g., *"Daniel Radcliffe, Emma Watson, Rupert Grint"*).
3. **Rating:** Provide an IMDb rating float from `0.0` to `10.0` (e.g., `7.6`).
4. **Metascore:** Enter a Metacritic score integer from `0` to `100` (e.g., `65`).
5. **Select Classifier:** Choose your desired algorithm from the dropdown:
   - `Logistic Regression`
   - `Decision Tree`
   - `KNN`
   - `Naive Bayes`
6. **Classify Genre Button:** Triggers real-time vectorization, model fitting, and displays the predicted genre in a popup dialog.

---

## ❓ Troubleshooting & FAQs

<details>
<summary><b>1. ModuleNotFoundError: No module named 'tkinter' on Linux</b></summary>
<br>
On some Linux distributions, Tkinter is not bundled with standard Python installations. Run:
```bash
sudo apt-get install python3-tk
```
</details>

<details>
<summary><b>2. FileNotFoundError: IMDB-Movie-Data.csv not found</b></summary>
<br>
Ensure you execute scripts from the root directory or via <code>python src/movie_genre_classifier.py</code>. The script uses dynamic relative path resolution based on <code>os.path.dirname(__file__)</code> to locate the <code>data/</code> folder correctly.
</details>

<details>
<summary><b>3. Why does classification take a couple of seconds after clicking "Classify"?</b></summary>
<br>
The current implementation fits the selected ML model dynamically on the 800-sample training matrix upon button click. See the roadmap below for pre-trained model serialization using <code>joblib</code>.
</details>

---

## 🚀 Future Enhancements & Roadmap

- [ ] **Multi-Label Classification:** Transition from single multi-class labels to multi-label binarization (`MultiOutputClassifier` or `Binary Relevance`) to evaluate individual genres (`Action`, `Drama`, `Comedy`) independently.
- [ ] **Model Persistence (`joblib` serialization):** Pre-train models and vectorizers to disk so the GUI loads instantly without retraining on click.
- [ ] **Transformer-based Embeddings:** Upgrade from TF-IDF bag-of-words to contextual dense embeddings using `sentence-transformers` (e.g., BERT, RoBERTa) or SpaCy.
- [ ] **Hyperparameter Tuning:** Implement `GridSearchCV` to optimize $K$ for KNN, tree depth for Decision Trees, and $C$ regularization parameters for Logistic Regression.
- [ ] **Modern Web Interface:** Develop a Streamlit or Flask/FastAPI web deployment with interactive charts and confidence probability scores.

---

## 📜 License & Acknowledgments

Distributed under the **MIT License**. See `LICENSE` for details.

- Dataset provided by IMDb movie metadata collection.
- Built with Python, [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), and [Tkinter](https://docs.python.org/3/library/tkinter.html).

---
## Author

Tasneem Ibrahim
