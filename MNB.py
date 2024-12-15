import re
import time
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
DATA_LOCATION = './data'
DATA_FILE = 'movies_balanced.json'
GENRES_FILE = 'popular_genres.json'

start_time = time.time()

# Downloading necessary resources
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

# Splitting the data into training, validation and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    data['overview'], data['genre'], test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(
    1, 1), max_features=5000, sublinear_tf=True)
X_train_vectors = vectorizer.fit_transform(X_train)
X_val_vectors = vectorizer.transform(X_val)
X_test_vectors = vectorizer.transform(X_test)

# Model (Kanskje utdype denne?)
model = MultinomialNB()
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(
    estimator=model,  # Model to tune
    param_grid={
        'alpha': param_grid['alpha'],
    },  # Hyperparameters to tune
    cv=10,  # Folds for cross-validation
    scoring='f1_weighted',  # Scoring metric
    n_jobs=-1,  # Use all available cores
    verbose=3  # For printing out progress
)

grid_search.fit(X_train_vectors, y_train)
best_model = grid_search.best_estimator_

# Use validation set to evaluate the model
y_pred = best_model.predict(X_val_vectors)
y_prob = best_model.predict_proba(X_val_vectors)

# Predictions and Evaluation using the best model with the test set
y_pred = best_model.predict(X_test_vectors)

# Print the best model and its accuracy
print("Best Model:", best_model)
print("Accuracy:", best_model.score(X_test_vectors, y_test))
print(classification_report(y_test, y_pred))

# Print the time taken
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# --- Visualizations ---

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
