import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('Dataset.csv', encoding='utf-8')

X = df['statement']
y = df['class']

df['statement']=df['statement'].str.lower()
df['statement'] = df['statement'].apply(lambda x: ''.join(char for char in x if char.isalpha() or char.isspace()))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train Multinomial Naïve Bayes classifier with LaPlace smoothing
clf = MultinomialNB(alpha=1.0, fit_prior=True) # alpha is laplace smoothing
clf.fit(X_train_counts, y_train)

# Predict the class for a new statement
new_statement = ["chinese chinese chinese tokyo japan"]
new_statement_counts = vectorizer.transform(new_statement)
predicted_class = clf.predict(new_statement_counts)

print("Predicted class for the new statement:", predicted_class[0])

# Evaluate the classifier
y_pred = clf.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)