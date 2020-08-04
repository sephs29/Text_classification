#! python3.6 


#read first sheet in excel file, given each column for each class/label
import pandas as pd
import xlrd #to work with excel #additional dependency
import numpy as np

from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

'''
#read excel file option #1
xls = pd.ExcelFile('datasheet1.xlsx')
sheet_name  = xls.sheet_names[0]
df  = xls.parse(sheet_name)
''
#read excel file option #2
df = pd.read_excel ('datasheet1.xlsx', sheet_name=0)
'''
#read excel file option #1
xls = pd.ExcelFile('datasheet1.xlsx')
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
df.name = list_name

#count dataframe df1
vectoriser = CountVectorizer()
vectoriser.fit(df.name)
x_train = vectoriser.transform(df.name)
y_train = df.label
model = MultinomialNB(alpha = 1)
model.fit(x_train,y_train)

unique_name = sorted(df.name.unique(), key = str.lower)
unique_label = sorted(df.label.unique(), key = str.lower)

df1 = pd.DataFrame(model.feature_count_)
df1 = df1.astype(int)
df1.columns = unique_name
#df1.index = unique_label
df1.index = model.classes_

#probability dataframe df2
probability = model.feature_log_prob_
probability = np.array(probability)
probability = np.exp(probability)
np.set_printoptions(suppress=True)

df2 = pd.DataFrame(probability)
df2.columns = unique_name
#df2.index = unique_label
df2.index = model.classes_

print(df1)
print(df2)

columns = [''] + list(df1.index) + ['predicted']
df3 = pd.DataFrame(columns = columns)

x_test1 = 'penn'#'bat'
print(x_test1)

#row = ['count']+list(df1[x_test1])+['']
#row_to_append = pd.Series(row,index= columns)
#df3 = df3.append(row_to_append,ignore_index=True)

#predict word x_test
model = [MultinomialNB(),MultinomialNB(alpha =  1.0e-10),ComplementNB(),ComplementNB(alpha = 1.0e-10)]
for ele in model:
    x_test = x_test1#'sun'#'mouse'#
    
    vectoriser = HashingVectorizer(alternate_sign= False)#CountVectorizer()#
    vectoriser.fit(df.name)   
    x_train = vectoriser.transform(df.name)
    y_train = df.label    
    model =  ele
    model.fit(x_train,y_train)  
    x_test = word_tokenize(x_test)
    x_test = vectoriser.transform(x_test)
    y_pred = model.predict(x_test)
    prob = model.predict_proba(x_test)
    prob = prob.tolist()[0]
    row = [(str(model))] + prob + [y_pred]
    row_to_append = pd.Series(row,index= columns)
    df3 = df3.append(row_to_append,ignore_index=True)
print(df3)

exit()
