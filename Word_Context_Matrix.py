from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

corpus=['I love AI','I love deep learning','I enjoy learning']

def build_co_occurrence_matrix(corpus,window_size):
    #build unique words
    unique_words=set()
    for text in corpus:
        for word in word_tokenize(text):
            unique_words.add(word)
    word_search_dict={word:np.zeros(shape=(len(unique_words))) for word in unique_words}
    word_list=list(word_search_dict.keys())
    for text in corpus:
        text_list=word_tokenize(text)
        for idx,word in enumerate(text_list):
            #pick word in the size range
            i=max(0,idx-window_size)
            j=min(len(text_list)-1,idx+window_size)
            search=[text_list[idx_] for idx_ in range(i,j+1)]
            search.remove(word)
            for neighbor in search:
                # get neighbor idx in word_search_dict
                nei_idx=word_list.index(neighbor)
                word_search_dict[word][nei_idx]+=1
    return word_search_dict

coo_dict=build_co_occurrence_matrix(corpus,window_size=1)
print(coo_dict)
print("\n")
table=pd.DataFrame(coo_dict,index=coo_dict.keys()).astype('int')
print(table)