import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Preprocess the text (if needed)
# For example, remove punctuation, convert to lowercase, etc.
df['statement']=df['statement'].str.lower()
df['statement']=df['statement'].apply(lambda x: ''.join(char for char in x if char.isalpha() or char.isspace()))

# Split the dataset into features and target
X = df['statement']
y = df['class']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert text data into feature vectors using CountVectorizer (binary=True for Bernoulli NB)
vectorizer = CountVectorizer(binary=True)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Bernoulli Naïve Bayes classifier with La-Place smoothing
model = BernoulliNB(alpha=1.0, binarize=None)
model.fit(X_train_vectorized, y_train)

# Predict the class for the given new sentence
new_sentence = ["chinese chinese chinese tokyo japan"]
new_sentence_vectorized = vectorizer.transform(new_sentence)
predicted_class = model.predict(new_sentence_vectorized)

print("Predicted Class:", predicted_class[0])

# Optionally, evaluate the model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
