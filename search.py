import requests
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

base_uri = 'https://api.dictionaryapi.dev/api/v2/entries/en/'

text = input('Paste the text you want to search here: ')
search_term = input('Input the word you want to search for here: ')

search_uri = base_uri + search_term
word = requests.get(search_uri).json()[0]
definitions = []
# note-to-self I have the [0] because d['definitions'] returns a list of definitions
# currently just using the first; could add loop to get all of em if there are multiple
definitions += [d['definitions'][0]['definition'] for d in word['meanings']]
df_defs = pd.DataFrame(definitions, columns=['definitions'])

porter = PorterStemmer()
def clean(text):
    punc = string.punctuation
    no_punc = str.maketrans('', '', punc)
    stops = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    words = [porter.stem(w.translate(no_punc)) for w in tokens if (w not in stops and w != '``' and w != "''" and w not in punc)]
    return words

df_defs['def_stems'] = df_defs['definitions'].apply(lambda row: clean(row))

print(df_defs)