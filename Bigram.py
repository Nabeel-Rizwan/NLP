import re
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Function to preprocess text and generate bigrams
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text into words
    words = text.split()
    # Generate bigrams
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    return bigrams

# Function to calculate bigram probabilities
def calculate_bigram_probabilities(bigrams):
    # Count occurrences of each bigram
    bigram_counts = Counter(bigrams)
    # Calculate total number of bigrams
    total_bigrams = sum(bigram_counts.values())
    # Calculate probabilities of each bigram
    bigram_probabilities = {bigram: count / total_bigrams for bigram, count in bigram_counts.items()}
    return bigram_probabilities

# Function to generate the bigram matrix using bigram probabilities
def generate_bigram_matrix(bigram_probabilities, vocabulary):
    # Initialize the bigram matrix with zeros
    bigram_matrix = np.zeros((len(vocabulary), len(vocabulary)))
    # Iterate through the bigram probabilities
    for bigram, probability in bigram_probabilities.items():
        # Get the indices of the words in the bigram
        index1 = vocabulary.index(bigram[0])
        index2 = vocabulary.index(bigram[1])
        # Update the corresponding cell in the bigram matrix with the probability
        bigram_matrix[index1][index2] = probability
        bigram_matrix[index2][index1] = probability  # Since it's symmetric, update the other cell as well
    return bigram_matrix

# Function to calculate cosine similarity between two words
def calculate_cosine_similarity(word1, word2, bigram_matrix, vocabulary):
    # Get the indices of the words in the vocabulary
    index1 = vocabulary.index(word1)
    index2 = vocabulary.index(word2)
    # Get the vectors for word1 and word2 from the bigram matrix
    vector1 = bigram_matrix[index1]
    vector2 = bigram_matrix[index2]
    # Calculate cosine similarity
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity

# Read text from the file
with open('txt2.txt', 'r') as file:
    text = file.read()

# Preprocess text and generate bigrams
bigrams = preprocess_text(text)

# Calculate bigram probabilities
bigram_probabilities = calculate_bigram_probabilities(bigrams)

# Extract vocabulary from the bigram probabilities
vocabulary = sorted(set(word for bigram in bigram_probabilities.keys() for word in bigram))

# Generate the bigram matrix using bigram probabilities and vocabulary
bigram_matrix = generate_bigram_matrix(bigram_probabilities, vocabulary)

print(bigram_matrix)

# Example: Calculate cosine similarity between two words
word1 = 'hello'
word2 = 'world'
similarity = calculate_cosine_similarity(word1, word2, bigram_matrix, vocabulary)
print(f"Cosine Similarity between '{word1}' and '{word2}': {similarity}")

