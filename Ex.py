import inline as inline
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import string

nltk.download('punkt')
nltk.download('stopwords')
sciezkaplik = "datasets_spam.csv"

slowaSpam = None
slowaHam = None

tekst = pd.read_csv(sciezkaplik, encoding="latin-1")
pd.set_option('display.max_columns', 5)

tekst = tekst[['v1', 'v2']]
tekst = tekst.rename(columns={'v1': 'Label', 'v2': 'SMS'})  # label= ham/spam, sms= treść sms
print(tekst.head())  # sprawdzenie czy poprawnie się wczytało

# zliczanie nulli
Suma_null_SMS = pd.isnull(tekst["SMS"].sum())
Suma_null_Label = pd.isnull(tekst["Label"].sum())

if (Suma_null_SMS == True or Suma_null_Label == True):
    tekst = tekst.dropna(how='any', axis=0)
    print("Wartości null zostały usunięte")
else:
    print("Nie ma nulli")

# train-test split- testowanie modelu (mozna było by zastosować filtr Naive Bayes
wszystkieSMS = tekst['SMS'].shape[0]
indexTreningu, testIndex = list(), list()
for i in range(tekst.shape[0]):
    if np.random.uniform(0, 1) < 0.7:
        indexTreningu += [i]
    else:
        testIndex += [i]
daneTreningu = tekst.loc[indexTreningu]
daneTestu = tekst.loc[testIndex]
print('dane TESTU:')
print(daneTestu)
print('dane TRENINGU')
print(daneTreningu)

daneTreningu.reset_index(inplace=True)
daneTreningu.drop(['index'], axis=1, inplace=True) #usuwanie określonych etykiet z wiersza lub kolumny
daneTreningu.head()


# przetwarzanie wstępne do treningu modelu
# znaki dolara itp
normalizowany = tekst['SMS'].str.replace(r'\€|\¥|\$', 'pieniadz')
# numer tel
normalizowany = normalizowany.str.replace(r'\d{3}-\d{3}-\d{3}', 'telefon')
# numer sms
normalizowany = normalizowany.str.replace(r'\d{4}', 'nrsms')
# adres www
normalizowany = normalizowany.str.replace('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'stronaint')
# usunięcie białych znaków (spacje w tekście, na początku i końcu tekstu)
normalizowany = normalizowany.str.replace(r'[^\w\d\s]', ' ')
normalizowany = normalizowany.str.replace(r'\s+', ' ')
normalizowany = normalizowany.str.replace(r'^\s+|\s+?$', '')

# usuwanie słów stopu
s_words = nltk.corpus.stopwords.words('english')
print('stop words:')
print(s_words[:5])

# znaki interpunkcyjne
interpunkcja = string.punctuation
print('interpunkcja')
print(interpunkcja)

# przekształcanie wszystkiego w małe litery
#normalizowany = normalizowany.str.lower() #lower zwraca ciąg małych loiter z podanego ciągu
#print(normalizowany)

def przetwarzanieWstepne(SMS):
    usunInterpunkcje="".join([word.lower() for word in SMS if word not in interpunkcja])
    tokenizacja = nltk.tokenize.word_tokenize(usunInterpunkcje)
    usunStopWords = [word for word in tokenizacja if word not in s_words]
    return usunStopWords

# tworzenie dodatkowej kolumny z przetworzonymi danymi
tekst['przetworzone'] = tekst['SMS'].apply(lambda x: przetwarzanieWstepne(x))
print(tekst['przetworzone'].head())

#normalizowany = normalizowany.apply(lambda x: ' '.join(word for word in x.split() if word not in set(s_words)))
#print(normalizowany)

#sprawdzenie aktualnego stanu tekstu
print('tekst:')
print(tekst)

# Kategoryzowanie i liczenie tokenów
def kategoryzacjaSlow():
    _slowaSpam = []
    _slowaHam = []
    for SMS in tekst['przetworzone'][tekst['Label'] == 'spam']:
        for words in SMS:
            _slowaSpam.append(words)

        for SMS in tekst['przetworzone'][tekst['Label'] == 'ham']:
            for word in SMS:
                _slowaHam.append(word)

        slowaSpam = _slowaSpam
        slowaHam = _slowaHam

    return [slowaSpam, slowaHam]

slowa = kategoryzacjaSlow()
slowaHam = slowa[1]
slowaSpam = slowa[0]
slowaSpam, slowaHam = kategoryzacjaSlow()

# sprawdzenie czy słowa się zapisały
print('ham: {}'.format(slowaHam[:10]))
print('spam: {}'.format(slowaSpam[:10]))


# przewidywanie funkcji
def przewidywanieFunkcji(SMS):
    licznikSpam = 0
    licznikHam = 0
    for word in SMS:
        licznikSpam += slowaSpam.count(word)
        licznikHam += slowaHam.count(word)
    print('*WYNIKI*')

    if licznikHam > licznikSpam:
        precyzja = round((licznikHam / (licznikSpam + licznikSpam) * 100))
        print('wiadomość nie jest spamem na {} %'.format(precyzja))
    elif licznikHam == licznikSpam:
        print('wiadomość niestety może być spam')
    else:
        precyzja = round((licznikSpam / (licznikHam + licznikSpam) * 100))
        print('wiadomość jest spamem na {} %'.format(precyzja))


# zbieranie danych wejściowych od użytkownika w celu sprawdzenia naszej funkcji
user_input = input(
    "Wpisz dowolną wiadomość (najlepiej spam lub ham) by sprawdzić czy nasza funkcja prawidłowo przewiduje dane: ")

przewidywanieFunkcji(user_input)
