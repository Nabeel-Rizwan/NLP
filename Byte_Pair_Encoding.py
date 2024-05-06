# Byte Pair Encoding

from collections import Counter, defaultdict

def get_vocab(text):
    # Initialize vocabulary with frequency of each word in text
    vocab = Counter(text.split())
    return {word: freq for word, freq in vocab.items()}

def get_stats(vocab):
    # Get frequency of adjacent symbol pairs (bigrams) in vocabulary
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    # Merge most frequent pair in all vocabulary words and update frequency
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

# Sample text data
text = "low lower newest widest low"

# Convert each word in initial vocabulary to space-separated string of characters
vocab = get_vocab(text)
vocab = {' '.join(word): freq for word, freq in vocab.items()}
print("Initial vocabulary:", vocab)

# Number of BPE iterations
num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    # Get the most frequent pair
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)
    print(f"After iteration {i+1}, Best pair: {best_pair}")
    print("Updated vocabulary:", vocab)