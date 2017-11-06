
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import datetime as dt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[2]:


def file_parser(file):
    for line in file:
        line = re.sub("\n", "", line)
        yield re.split(r',', line, 1)

parser = file_parser(open('Teste_Classificacao_Out_01-15_GENERO-PRODUTO.csv', encoding="ISO-8859-1"))
columns = next(parser)


df = pd.DataFrame(parser, columns=columns)
df_test=df


# In[3]:


y = df_test['GENERO']
X = df_test['PRODUTO'].tolist()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)


# In[4]:


vectorizer_train = CountVectorizer()
X_train = vectorizer_train.fit_transform(X_train)


# In[5]:


vectorizer_test = CountVectorizer(vocabulary=vectorizer_train.vocabulary_)
X_test = vectorizer_test.transform(X_test)


# In[6]:


model = MultinomialNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))


# In[7]:


#exportar dados da predição do teste para csv
def relatorio_classificacao_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        data=list(filter(None,row_data))
        row['class'] = data[0].strip()
        row['precision'] = data[1].strip()
        row['recall'] = data[2].strip()
        row['f1_score'] = data[3].strip()
        row['support'] = data[4].strip()
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('Relatorio_classificacao_'+ dt.datetime.today().strftime("%d%m%Y%H%M%S")+ '.csv', index = False)


# In[8]:


report = classification_report(y_test, y_pred)
relatorio_classificacao_csv(report)


# In[7]:


from sklearn import metrics


# In[8]:


score = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)


# In[9]:


print('{0:f}'.format(score))


# In[10]:


teste_predict=['EPSON MAGENTO','LILLO BICO','BOV AMERICANA KG','KING 100ML JASMIM']
teste_predict_vect = vectorizer_train.transform(teste_predict) 
model.predict(teste_predict_vect)


# In[11]:


pred_prob_list = model.predict_proba(teste_predict_vect).tolist()
pred_prob_list


# In[12]:


greatest_prob = max(pred_prob_list[0])
greatest_prob


# In[13]:


def get_greatest_probabilities(pred_prob_list):
    max_probabilities = []
    for i in pred_prob_list:
       max_probabilities.append(max(i))
    return max_probabilities


# In[14]:


get_greatest_probabilities(pred_prob_list)


# In[15]:


def prediction_script(text):
    predict_vect = vectorizer_train.transform(text)
    yield model.predict(teste_predict_vect)


# In[16]:


list(prediction_script(['EPSON MAGENTO','LILLO BICO','BOV AMERICANA KG','KING 100ML JASMIM']))


# In[17]:


filename = 'model_v2.pkl'


# In[18]:


from sklearn.externals import joblib


# In[19]:


with open(filename, 'wb') as file:
	joblib.dump(model, file)


# In[20]:


file.close()


# In[21]:


vocab_filename = 'model_v2_vocabulary.pkl'


# In[22]:


with open(vocab_filename, 'wb') as file2:
    joblib.dump(vectorizer_train.vocabulary_, vocab_filename)


# In[23]:


vectorizer_train.vocabulary_

