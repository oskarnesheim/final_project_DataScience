import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Constants
DATA_LOCATION = './data'
DATA_FILE = 'movies_balanced.json'
GENRES_FILE = 'popular_genres.json'


# Downloading necessary resources
nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset
data = pd.read_json(f'{DATA_LOCATION}/{DATA_FILE}')

# load genres
genres = pd.read_json(f'{DATA_LOCATION}/{GENRES_FILE}')
genres = genres.to_numpy().flatten()


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)

    # Tokenize text
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join words back to string
    return ' '.join(tokens)


# Preprocessing descriptions
data['overview'] = data['overview'].apply(preprocess_text)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    data['overview'], data['genre'], test_size=0.3, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(
    1, 2), max_features=5000, sublinear_tf=True)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Model (Kanskje utdype denne?)
model = MultinomialNB()
param_grid = {'alpha': [0.5, 0.55, 0.6, 0.65,
                        0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]}
# grid_search = RandomizedSearchCV(
#     model, param_distributions=param_grid, n_iter=12, cv=5, random_state=42)
grid_search = GridSearchCV(
    estimator=model,  # Modellen som skal tunes
    param_grid={'alpha': param_grid['alpha']},  # Parameterområde
    cv=10,  # Antall fold i kryssvalidering
    scoring='f1_weighted',  # Evalueringsmetode
    n_jobs=-1,  # Parallell prosessering for raskere søk
    verbose=3  # For detaljerte logger under søket
)

grid_search.fit(X_train_vectors, y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_vectors)
y_prob = best_model.predict_proba(X_test_vectors)

# Predictions and Evaluation using the best model
y_pred = best_model.predict(X_test_vectors)
print("Best Model:", best_model)
print("Accuracy:", best_model.score(X_test_vectors, y_test))
print(classification_report(y_test, y_pred))

# Creating the confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report Visualization
report = pd.DataFrame(classification_report(
    y_test, y_pred, output_dict=True)).transpose()
report.drop(['accuracy'], inplace=True)
report['support'] = report['support'].apply(int)
fig, ax = plt.subplots(figsize=(8, 5))
report[['precision', 'recall', 'f1-score']].plot(kind='barh', ax=ax)
ax.set_title('Classification Report')
ax.set_xlim([0, 1])
plt.show()

# Visualizing Classification Report in table
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(report[['precision', 'recall', 'f1-score']], annot=True,
            cmap='Blues', fmt=".2f", ax=ax)
ax.set_title('Classification Report')
plt.show()
