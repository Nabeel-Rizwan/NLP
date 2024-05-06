import pandas as pd
from nltk.util import ngrams

data="hello today world how are you doing doing today"

bigram = pd.Series(ngrams(data.split(),2)) ### creating the bigrams of ordered pairs like:

print(bigram)

probabilities = bigram.value_counts(normalize=True)  ### getting probability of each ordered pair
letters = pd.Series(data.split()).unique()  ### getting each chord 
prob = bigram.value_counts(normalize=True)

print(probabilities)

print("Vocabulary: ", letters)
