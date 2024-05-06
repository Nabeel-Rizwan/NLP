from nltk.corpus import words

string="helloworld"
lowercase=[x.lower() for x in words.words()]

token=[]
i=0

while i<len(string):
    maxword=""
    for j in range(i,len(string)):
        temp=string[i:j+1]
        if temp in lowercase and len(temp)>len(maxword):
            maxword=temp
    i=i+len(maxword)
    token.append(maxword)        
    
print(token)