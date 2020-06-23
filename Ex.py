import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
sciezkaplik = "datasets_spam.csv"
tekst = pd.read_csv(sciezkaplik, encoding="latin-1")
pd.set_option('display.max_columns',5)

tekst = tekst[['v1','v2']]
tekst = tekst.rename(columns={'v1':'Label','v2':'SMS'})
print(tekst.head())
Suma_null_SMS = pd.isnull(tekst["SMS"].sum())
Suma_null_Label = pd.isnull(tekst["Label"].sum())

if(Suma_null_SMS == True or Suma_null_Label == True):
   tekst = tekst.dropna(how='any',axis=0)
   print("Wartości null zostały usunięte")
else:
    print("Nie ma nulli")


#znaki interpunkcyjne
normalizowany = tekst['SMS'].str.replace(r'[\W]+', ' ')
#znaki dolara itp
normalizowany = normalizowany.str.replace(r'\€|\¥|\$', 'pieniadz')
#numer tel
normalizowany = normalizowany.str.replace(r'\d{3}-\d{3}-\d{3}', 'telefon')
#numer sms
normalizowany = normalizowany.str.replace(r'\d{4}', 'nrsms')
#adres www
normalizowany = normalizowany.str.replace('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b','stronaint')
#małe litery
normalizowany = normalizowany.str.lower()
print(normalizowany)
X = normalizowany
Y = tekst['Label']


s_words = set(stopwords.words('english'))
vector = CountVectorizer(stop_words=s_words)

X_train, X_test, Y_test, Y_train = train_test_split(X,Y,test_size=0.33)
X_train_vector = vector.fit_transform(X_train)



