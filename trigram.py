import re
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Function to preprocess text and generate trigrams
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text into words
    words = text.split()
    # Generate trigrams
    trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
    return trigrams

# Function to calculate trigram probabilities
def calculate_trigram_probabilities(trigrams):
    # Count occurrences of each trigram
    trigram_counts = Counter(trigrams)
    # Calculate total number of trigrams
    total_trigrams = sum(trigram_counts.values())
    # Calculate probabilities of each trigram
    trigram_probabilities = {trigram: count / total_trigrams for trigram, count in trigram_counts.items()}
    return trigram_probabilities

# Function to generate the trigram matrix using trigram probabilities and vocabulary
def generate_trigram_matrix(trigram_probabilities, vocabulary):
    # Initialize the trigram matrix with zeros
    trigram_matrix = np.zeros((len(vocabulary), len(vocabulary), len(vocabulary)))
    # Iterate through the trigram probabilities
    for trigram, probability in trigram_probabilities.items():
        # Get the indices of the words in the trigram
        index1 = vocabulary.index(trigram[0])
        index2 = vocabulary.index(trigram[1])
        index3 = vocabulary.index(trigram[2])
        # Update the corresponding cell in the trigram matrix with the probability
        trigram_matrix[index1][index2][index3] = probability
    return trigram_matrix

# Function to calculate cosine similarity between two words
def calculate_cosine_similarity(word1, word2, trigram_matrix, vocabulary):
    # Get the indices of the words in the vocabulary
    index1 = vocabulary.index(word1)
    index2 = vocabulary.index(word2)
    # Get the vectors for word1 and word2 from the trigram matrix
    vector1 = trigram_matrix[index1]
    vector2 = trigram_matrix[index2]
    # Flatten the vectors to compute cosine similarity
    vector1_flat = vector1.flatten()
    vector2_flat = vector2.flatten()
    # Calculate cosine similarity
    similarity = cosine_similarity([vector1_flat], [vector2_flat])[0][0]
    return similarity

# Read text from the file
with open('txt2.txt', 'r') as file:
    text = file.read()

# Preprocess text and generate trigrams
trigrams = preprocess_text(text)

# Calculate trigram probabilities
trigram_probabilities = calculate_trigram_probabilities(trigrams)

# Extract vocabulary from the trigram probabilities
vocabulary = sorted(set(word for trigram in trigram_probabilities.keys() for word in trigram))

# Generate the trigram matrix using trigram probabilities and vocabulary
trigram_matrix = generate_trigram_matrix(trigram_probabilities, vocabulary)
print(trigram_matrix)
# Example: Calculate cosine similarity between two words
word1 = 'hello'
word2 = 'world'
similarity = calculate_cosine_similarity(word1, word2, trigram_matrix, vocabulary)
print(f"Cosine Similarity between '{word1}' and '{word2}': {similarity}")
