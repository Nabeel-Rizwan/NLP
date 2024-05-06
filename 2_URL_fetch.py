import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

def fetch (url):
    response=requests.get(url)
    if response.status_code == 200:
        print("Successfully fetched url")
        soup=BeautifulSoup(response.text,"html.parser")
        main_content=soup.find("main")
        if main_content:
            print("Main content exists")
            text=main_content.get_text()
            # print(text)
            lines=[text]
            with open('text.txt','w') as file:
                
                for lines in lines:
                    file.write(lines)


            with open('text.txt','r', encoding='utf-8') as file:
                lines=file.read()
                special_char=set('qwertyuiopasdfghjklzxcvbnm QWERTYUIOPASDFGHJKLZXCVBNM')
                lines=[char for char in lines if char in special_char]
                lines=''.join(lines)
                stop_words=set(stopwords.words('english'))
                lines=lines.split()
                
                lines=[char for char in lines if char not in stop_words]
                lines=' '.join(lines)
                lines=lines.lower()

                print(lines) 
                words=1
                lines=lines.split()
                words+=len(lines)
                print(words)


if __name__ == "__main__":
    url='https://aws.amazon.com/what-is/java/#:~:text=Java%20is%20a%20widely%2Dused,as%20a%20platform%20in%20itself.'
    fetch(url)

