import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('stopwords')

# Constants
DATA_LOCATION = './data'
DATA_FILE = 'movies_balanced.json'
GENRES_FILE = 'popular_genres.json'

# Load dataset
data = pd.read_json(f'{DATA_LOCATION}/{DATA_FILE}')

# load genres
genres = pd.read_json(f'{DATA_LOCATION}/{GENRES_FILE}')
genres = genres.to_numpy().flatten()

# #Function to preprocess text by lowercasing, removing non-alphanumeric characters, removing stopwords, and lemmatizing.


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

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

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
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Model (Kanskje utdype denne?)
model = MultinomialNB()
param_grid = {'alpha': [0.01, 0.05, 0.1, 0.2,
                        0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
grid_search = RandomizedSearchCV(
    model, param_distributions=param_grid, n_iter=12, cv=5, random_state=42)
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
