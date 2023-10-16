import requests, string, pandas as pd, numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base_uri = 'https://api.dictionaryapi.dev/api/v2/entries/en/'

text = input('Paste the text you want to search here: ')
search_term = input('Input the word you want to search for here: ')

search_uri = base_uri + search_term
word = requests.get(search_uri).json()[0]
definitions = []
sentences = sent_tokenize(text)
# note-to-self I have the [0] because d['definitions'] returns a list of definitions
# currently just using the first; could add loop to get all of em if there are multiple
definitions += [d['definitions'][0]['definition'] for d in word['meanings']]
df_defs = pd.DataFrame(definitions, columns=['definitions'])
df_sents = pd.DataFrame(sentences, columns=['sentences'])

porter = PorterStemmer()
def clean(sentence):
    punc = string.punctuation
    no_punc = str.maketrans('', '', punc)
    stops = set(stopwords.words('english'))
    tokens = word_tokenize(sentence)
    words = [porter.stem(w.translate(no_punc)) for w in tokens if (w not in stops and w != '``' and w != "''" and w not in punc)]
    return words

df_defs['def_stems'] = df_defs['definitions'].apply(lambda row: clean(row))
df_sents['text_stems'] = df_sents['sentences'].apply(lambda row: clean(row))

def vectorize(text, definition):
    vectorizer = TfidfVectorizer()
    text_str = ' '.join(text)
    def_str = ' '.join(definition)
    corpus = [text_str, def_str]
    matrix = vectorizer.fit_transform(corpus)
    cos_sim = cosine_similarity(matrix, matrix)
    return cos_sim[0, 1]

for i in range(len(df_defs)):
    df_sents[f'similarity_{i}'] = df_sents['text_stems'].apply(lambda row: vectorize(row, df_defs['def_stems'].loc[i]))
    i += 1

print(df_sents)