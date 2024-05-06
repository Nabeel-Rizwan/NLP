import pandas as pd
from nltk.util import ngrams

data="hello today world how are you doing doing today"

bigram = pd.Series(ngrams(data.split(),2)) ### creating the bigrams of ordered pairs like:

### 0      (F, Em7) 
### 1     (Em7, A7) 
### 2      (A7, Dm)

probabilities = bigram.value_counts(normalize=True)  ### getting probability of each ordered pair
letters = pd.Series(data.split()).unique()  ### getting each chord 
prob = bigram.value_counts(normalize=True)

mat = (
    pd.Series(
        prob, index=[prob.index.str[0], prob.index.str[1]])
    .unstack(fill_value=0).round(3)
)

print(mat)