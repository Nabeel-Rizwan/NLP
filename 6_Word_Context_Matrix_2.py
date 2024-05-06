# Generate word count matrix for a given matrix based on a specified window size.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def generate_word_context_matrix(corpus, window_size):
    words = corpus.split()
    word_to_index = {}
    index_to_word = {}
    word_count = 0

    # Build vocabulary and assign indices to words
    for word in words:
        if word not in word_to_index:
            word_to_index[word] = word_count
            index_to_word[word_count] = word
            word_count += 1

    # Initialize word-context matrix
    matrix = np.zeros((word_count, word_count))

    # Populate word-context matrix
    for i, target_word in enumerate(words):
        target_index = word_to_index[target_word]
        start_index = max(0, i - window_size)
        end_index = min(len(words), i + window_size + 1)
        context_words = words[start_index:i] + words[i + 1:end_index]
        for context_word in context_words:
            if context_word in word_to_index:
                context_index = word_to_index[context_word]
                matrix[target_index][context_index] += 1

    return matrix, word_to_index, index_to_word

def generate_word_context_matrix_with_respect_to_targets(corpus, window_size, target_words):
    words = corpus.split()
    word_to_index = {}
    index_to_word = {}
    target_indices = []

    # Build vocabulary and assign indices to words
    for word in words:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
            index_to_word[len(index_to_word)] = word

    # Get indices of target words
    for target_word in target_words:
        if target_word in word_to_index:
            target_indices.append(word_to_index[target_word])

    # Initialize word-context matrix
    matrix = np.zeros((len(target_indices), len(word_to_index)))

    # Populate word-context matrix focusing only on target words
    for i, target_index in enumerate(target_indices):
        start_index = max(0, target_index - window_size)
        end_index = min(len(words), target_index + window_size + 1)
        context_words = words[start_index:target_index] + words[target_index + 1:end_index]
        for context_word in context_words:
            if context_word in word_to_index:
                context_index = word_to_index[context_word]
                matrix[i][context_index] += 1

    return matrix, word_to_index, index_to_word

def calculate_cosine_similarity(matrix, target_word_indices):
    similarity_scores = {}
    for target_word_index in target_word_indices:
        similarity_scores[target_word_index] = cosine_similarity(matrix[target_word_index].reshape(1, -1), matrix)
    return similarity_scores

def find_similar_word(similarity_scores, index_to_word, top_n=5):
    similar_words_dict = {}
    for target_word_index, scores in similarity_scores.items():
        word_index = target_word_index
        similar_word_indices = np.argsort(scores[0])[::-1][:top_n]
        similar_words = [(index_to_word[index], scores[0][index]) for index in similar_word_indices if index != word_index]
        similar_words_dict[index_to_word[target_word_index]] = similar_words
    return similar_words_dict

# Example corpus
corpus = "data stored memory sup system connect you to computer"

# Example parameters
window_size = 2
target_words = ["memory", "system", "computer"]

# Generate word-context matrix
matrix, word_to_index, index_to_word = generate_word_context_matrix_with_respect_to_targets(corpus, window_size, target_words)

# Print word-context matrix
print("Word-Context Matrix with respect to the target words '{}':".format(target_words))
print(matrix)

# Calculate cosine similarity
similarity_scores = calculate_cosine_similarity(matrix, range(len(target_words)))

# Find similar words
similar_words_dict = find_similar_word(similarity_scores, index_to_word)

# Display similar words for each target word
for target_word, similar_words in similar_words_dict.items():
    print("\nSimilar words for '{}' are:".format(target_word))
    for word, score in similar_words:
        print("{} (Similarity Score: {:.4f})".format(word, score))
