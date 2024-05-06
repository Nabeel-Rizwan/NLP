from bs4 import BeautifulSoup
import requests
import os.path as path
import math

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.stem import PorterStemmer
import pandas as pd


URLS={
    "football":["https://en.wikipedia.org/wiki/Football","https://en.wikipedia.org/wiki/American_football","https://en.wikipedia.org/wiki/Association_football","https://en.wikipedia.org/wiki/Australian_rules_football","https://en.wikipedia.org/wiki/Gaelic_football"],
    "algorithm":["https://en.wikipedia.org/wiki/Algorithm","https://en.wikipedia.org/wiki/Analysis_of_algorithms","https://en.wikipedia.org/wiki/Computational_complexity","https://en.wikipedia.org/wiki/Worst-case_complexity","https://en.wikipedia.org/wiki/Average-case_complexity"],
}

BASE_URL="Documents"

stop_words=(stopwords.words('english'))

def getHTMLFromURL(url):
    page = requests.get(url).content
    soup = BeautifulSoup(page, 'html.parser')
    return soup;

def getBodyTextFromHTML(soup):
    paragraphs = soup.find_all('p')
    text = '\n'.join([para.get_text().strip() for para in paragraphs])
    return text;

def writeInTextFile(topic,filename,text):
    file = open(path.join(BASE_URL,topic,filename),"w")
    file.write(text)
    file.close()

def scrapeFunction():
    for topic in URLS:
        fileno=1;
        for url in URLS[topic]:
            soup = getHTMLFromURL(url)
            text = getBodyTextFromHTML(soup)
            filename = f"{fileno}.txt"
            writeInTextFile(topic,filename,text)
            print("Done for ",topic,filename)
            fileno=fileno+1

scrapeFunction()

unique_word_set={}
unique_word_dict={}

def getUniqueWords(text):
    unique_words={}
    words = word_tokenize(text)
    for word in words:
        word=word.lower()
        if word in stop_words:
            continue
        unique_word_set[word]=1
        if word in unique_words:
            unique_words[word]=unique_words[word]+1
        else:
            unique_words[word]=1
    return unique_words;

def getUniqueWordsFromFiles():
    for topic in URLS:
        if topic not in unique_word_dict:
            unique_word_dict[topic]={}
        for i in range(1,6):
            filename = f"{i}.txt"
            file = open(path.join(BASE_URL,topic,filename),"r")
            text = file.read()
            file.close()
            unique_words=getUniqueWords(text);

            for word in unique_words:
                if word in unique_word_dict[topic]:
                    unique_word_dict[topic][word]=unique_word_dict[topic][word]+unique_words[word]
                else:
                    unique_word_dict[topic][word]=1
                    

    print("Total Unique Words from files are ",len(unique_word_set))

getUniqueWordsFromFiles()

df = pd.DataFrame(unique_word_dict)
print(df.transpose().fillna(0))


def getUniqueBigramsWords(text):
    unique_words={}
    words = word_tokenize(text)
    prev="<string>"
    for word in words:
        word=word.lower()
        if word in stop_words:
            continue
        bigram=prev+" "+word
        unique_bigram_set[bigram]=1
        if bigram in unique_words:
            unique_words[bigram]=unique_words[bigram]+1
        else:
            unique_words[bigram]=1
        prev=word
    return unique_words;

unique_bigram_set={}
unique_bigram_dict={}
unique_bigram_length={}

def getUniqueBigramsFromFiles():
    for topic in URLS:
        if topic not in unique_bigram_length:
                unique_bigram_length[topic]=0
        if topic not in unique_bigram_dict:
                unique_bigram_dict[topic]={}
        for i in range(1,6):
            filename = f"{i}.txt"
            file = open(path.join(BASE_URL,topic,filename),"r")
            text = file.read()
            file.close()
            unique_bigrams=getUniqueBigramsWords(text);
            
            for bigram in unique_bigrams:
                if bigram in unique_bigram_dict[topic]:
                    unique_bigram_dict[topic][bigram]=unique_bigram_dict[topic][bigram]+unique_bigrams[bigram]
                else:
                    unique_bigram_dict[topic][bigram]=1
                
                unique_bigram_length[topic]+=1;
                    

    print("Total Unique Words from files are ",len(unique_bigram_set))

getUniqueBigramsFromFiles();

df = pd.DataFrame(unique_bigram_dict)
df.transpose().fillna(0)

unique_bigram_prob_dict={}
for topic in unique_bigram_dict:
    unique_bigram_prob_dict[topic]={}
    for bigram in unique_bigram_set:
        unique_bigram_prob_dict[topic][bigram]=0;
        if bigram in unique_bigram_dict[topic]:
            unique_bigram_prob_dict[topic][bigram]=unique_bigram_dict[topic][bigram]/unique_bigram_length[topic]

df=pd.DataFrame(unique_bigram_prob_dict)
df.transpose().fillna(0)

unique_word_dict_tfidf={}
unique_word_doc_count={}
def getUniqueWordsFromFiles():
    for topic in URLS:
        unique_word_dict_tfidf[topic]={}
        unique_word_doc_count[topic]={}
        for i in range(1,6):
            filename = f"{i}.txt"
            file = open(path.join(BASE_URL,topic,filename),"r")
            text = file.read()
            file.close()
            unique_words=getUniqueWords(text);
            
            unique_word_dict_tfidf[topic][i]={}
            unique_word_doc_count[topic][i]=0

            for word in unique_words:
                unique_word_dict_tfidf[topic][i][word]=unique_words[word]
                unique_word_doc_count[topic][i]+=unique_words[word]

getUniqueWordsFromFiles();

for topic in unique_word_dict_tfidf:
    for i in unique_word_dict_tfidf[topic]:
        for word in unique_word_dict_tfidf[topic][i]:
            total=len(unique_word_dict_tfidf[topic])
            count=0
            for j in unique_word_dict_tfidf[topic]:
                if word in unique_word_dict_tfidf[topic][j]:
                    count+=1
            idf=math.log10(total/count)
            tf=unique_word_dict_tfidf[topic][i][word]/unique_word_doc_count[topic][i]
            unique_word_dict_tfidf[topic][i][word]=tf*idf;

pd.DataFrame(unique_word_dict_tfidf['football']).fillna(0)
pd.DataFrame(unique_word_dict_tfidf['algorithm']).fillna(0)

my_sentence="I like algorithms and football"
def removeStopWords(sentence):
    words = word_tokenize(sentence)
    new_sentence=""
    for word in words:
        if word.lower() not in stop_words:
            new_sentence+=word+" "
    return new_sentence;

words=word_tokenize(removeStopWords(my_sentence))

prob1=1
prob2=1;
total_word1=0;
total_word2=0;

for word in unique_word_dict['football']:
    total_word1+=unique_word_dict['football'][word];

for word in unique_word_dict['algorithm']:
    total_word2+=unique_word_dict['algorithm'][word];

for word in words:

    # for class football
    if word in unique_word_dict['football']:
        prob1*=unique_word_dict['football'][word]/(total_word1);
    else:
        # laplace
        prob1*=(1)/(len(unique_word_dict['football'])+total_word1)

    # for class algorithm
    if word in unique_word_dict['algorithm']:
        prob2*=unique_word_dict['algorithm'][word]/(total_word2);
    else:
        # laplace
        prob2*=(1)/(len(unique_word_dict['algorithm'])+total_word2)

print('Football : ',(prob1/(prob1+prob2))*100)
print('Algorithm : ',(prob2/(prob1+prob2))*100)