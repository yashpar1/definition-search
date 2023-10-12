import requests

base_uri = 'https://api.dictionaryapi.dev/api/v2/entries/en/'

text = input('Paste the text you want to search here: ')
search_term = input('Input the word you want to search for here: ')

search_uri = base_uri + search_term
word = requests.get(search_uri).json()[0]
definitions = []
definitions += [d['definitions'][0]['definition'] for d in word['meanings']]