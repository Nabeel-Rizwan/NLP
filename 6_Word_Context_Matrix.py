import numpy as np
from gensim.models import Word2Vec
import nltk

# Load and preprocess the corpus

corpus_text = "i i i hello world"

# Tokenize the corpus
corpus_sentences = nltk.sent_tokenize(corpus_text)
corpus_words = [nltk.word_tokenize(sentence.lower()) for sentence in corpus_sentences]

# Train a word2vec model with the specified window size
model = Word2Vec(corpus_words, min_count=1, window=2)

# Get vocabulary size
vocab_size = len(model.wv)

# Initialize co-occurrence matrix
cooccurrence_matrix = np.zeros((vocab_size, vocab_size))

# Iterate through each word pair in the corpus
for context in corpus_words:
    for i, target_word in enumerate(context):
        target_index = model.wv.key_to_index[target_word]
        for j, context_word in enumerate(context):
            context_index = model.wv.key_to_index[context_word]
            cooccurrence_matrix[target_index][context_index] += 1

print(cooccurrence_matrix)
