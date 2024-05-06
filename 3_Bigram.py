import pandas as pd
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = "hello today world how are you doing doing today"

# Create bigrams
bigram_series = pd.Series(ngrams(data.split(), 2))
print(bigram_series)
# Convert bigrams to strings
prob=bigram_series.value_counts(normalize=True)
print(prob)

bigram_strings = bigram_series.apply(lambda x: ' '.join(x))

# Initialize CountVectorizer to convert bigrams to vectors
vectorizer = CountVectorizer()

# Fit and transform bigrams to vector representation
bigram_vectors = vectorizer.fit_transform(bigram_strings)

# Calculate cosine similarity between each pair of bigram vectors
cosine_similarities = cosine_similarity(bigram_vectors, bigram_vectors)

# Print cosine similarities matrix
print("Cosine Similarities Matrix:")
print(cosine_similarities)
