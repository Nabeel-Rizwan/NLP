from nltk.corpus import stopwords
import re

def preprocess(txt):
    txt=txt.lower()   
    txt=re.sub(r'<.*?>','',txt)
    txt=re.sub(r'http\S+','',txt)
    special_char=set("QWERTYUIOPASDFGHJKLZXCVBNM qwertyuiopasdfghjklzxcvbnm")
    txt= ''.join(char for char in txt if char in special_char)
    stop_word=set(stopwords.words('english'))
    txt=txt.split()
    txt=[char for char in txt if char not in stop_word]
    txt=' '.join(txt)
    return txt
3
if __name__ == "__main__":
    text=input("Enter text: ")
    text=preprocess(text)
    print(text)