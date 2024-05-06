import re
import numpy as np
from collections import Counter

# Function to preprocess text and generate unigrams
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text into words
    words = text.split()
    return words

# Function to calculate unigram probabilities
def calculate_unigram_probabilities(unigrams):
    # Count occurrences of each word
    word_counts = Counter(unigrams)
    # Calculate total number of words
    total_words = sum(word_counts.values())
    # Calculate probabilities of each word
    unigram_probabilities = {word: count / total_words for word, count in word_counts.items()}
    return unigram_probabilities

# Read text from the file
with open('txt2.txt', 'r') as file:
    text = file.read()

# Preprocess text and generate unigrams
unigrams = preprocess_text(text)

# Calculate unigram probabilities
unigram_probabilities = calculate_unigram_probabilities(unigrams)

# Extract vocabulary from the unigram probabilities
vocabulary = sorted(unigram_probabilities.keys())

# Generate the unigram matrix using unigram probabilities and vocabulary
unigram_matrix = np.array([unigram_probabilities[word] for word in vocabulary])

print(unigram_matrix)
