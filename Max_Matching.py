import nltk
from nltk.corpus import words

string = input("Enter the statement: ")

lowercase = [x.lower() for x in words.words()]

tokens = []
i = 0

while i < len(string):
    maxword = ""
    for j in range(i, len(string)):
        temp = string[i:j + 1]
        if temp in lowercase and len(temp) > len(maxword):
            maxword = temp

    i = i + len(maxword)
    tokens.append(maxword)

print(tokens)