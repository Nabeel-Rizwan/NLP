from sklearn.feature_extraction.text import TfidfVectorizer

# Read text from the file
with open('txt2.txt', 'r') as file:
    text = file.read()

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text data to get the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform([text])

# Convert the TF-IDF matrix to an array for easier handling
tfidf_array = tfidf_matrix.toarray()

# Print the TF-IDF matrix
print(tfidf_matrix)
print(tfidf_array)