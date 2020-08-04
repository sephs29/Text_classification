#! python3.6 

import pandas as pd
import xlrd #to work with excel #additional dependency
import string
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

xls = pd.ExcelFile('sentiment.xlsx')
sheet_name  = xls.sheet_names[0]
exceldata  = xls.parse(sheet_name)

label = exceldata.columns
list_label = []
list_name = []
for ele in label:
    column = exceldata[ele].dropna()
    n = len(column)    
    list_label.extend([ele] * n)
    list_name.extend(column)

df  = pd.DataFrame(columns = ['name','label'])
df.label = list_label

## stemming is reducing extensively like offensive becomes offens, unpretentious becomes unpretenti
##porter = PorterStemmer()
##list_name = [porter.stem(word) for word in list_name]
##print(list_name)

wordnet_lemmatizer = WordNetLemmatizer()
list_name = [wordnet_lemmatizer.lemmatize(word, pos= "v") for word in list_name]
df.name = list_name

stop_words = set(stopwords.words("english"))
vectorizer = HashingVectorizer(stop_words = stop_words, alternate_sign= False)
vectorizer.fit(df.name)   
x_train = vectorizer.transform(df.name)
y_train = df.label  
model = ComplementNB()
#model = MultinomialNB(alpha = 1.0e-10)
model.fit(x_train,y_train) 

#-----------------------------test data---------------------------
#text = 'The movie had a slow start, but as the time proceeded, the movie took unpredictable twists and turns. It was so fascinating and surprising, that it was hard to take ones eyes of the screen. It was so cleverly directed that it kept me completely engaged. Super imaginative. For me, it is a 5-star movie. Highly recommended. '#positive
#text = 'The movie has a pleasant start, and that is all to it. It has such an ordinary story line that you can predict the next scene. It just getting more silly and stupid with time. By the end of the watch, you will realise what a complete waste of time it was.It is such a dumb watch. Well, it was a yawn-inducing, bland and senseless movie. Not recommended. One-star from me.'#negative
#text = 'The movie had a very original start and it was comparative slow. But in my opinion, it had a thought-provoking idea. It gets uninteresting and tiresome in the middle, and a little predictable. But on the whole, it was unpretentious and tender. 3-star from me.'#neutral

text = text.lower()
text = word_tokenize(text)
stop_words=set(stopwords.words("english"))
wordnet_lemmatizer = WordNetLemmatizer()
text = [wordnet_lemmatizer.lemmatize(word, pos= "v") for word in text if not word in stop_words and not word in string.punctuation]
print(text)
x_test = vectorizer.transform(text)
y_pred = model.predict(x_test)
label = model.classes_
prob = model.predict_proba(x_test)
prob = [True if np.amax(ele) > 0.60 else False for ele in prob ]
y_pred1 = [y_pred[i] if ele else ''  for i,ele in enumerate(prob) ]
print(y_pred1)
y_pred = [y_pred[i]   for i,ele in enumerate(prob) if ele]
y_pred = [y_pred.count(ele) for ele in label]
print(label)
print(y_pred)
y_pred = y_pred - np.amax(y_pred)
y_pred = [True if ele >= 0 else False for ele in y_pred ]
print(y_pred)
if sum(y_pred) == 1:
    result = [label[i]  for i,ele in enumerate(y_pred) if ele]
else:
    result = 'Neutral'
print(result)


exit()

