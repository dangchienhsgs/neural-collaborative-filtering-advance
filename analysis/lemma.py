from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import numpy as np
import re

stemmer = PorterStemmer()
pattern = re.compile('([^\s\w]|_)+')
stop = set(stopwords.words("english"))


def remove_stop_words(words):
    return [word for word in words if word.lower() not in stop]


def remove_special_char(value):
    return pattern.sub('', value)


def stem(word):
    return stemmer.stem(word=word)


f = open('imdb/plot-summaries')
f2 = open("imdb/plot-summaries-analyzed", 'w')

word_dict = {}

for line in f:
    args = line.split("||")

    movie_id = args[0]
    name = args[1]
    types = args[2]

    types = remove_special_char(types).lower()
    types = ' '.join(types.split())

    content = remove_special_char(args[3])
    x = content.lower().split()
    x = remove_stop_words(x)
    x = [stem(word) if not word.isdigit() else 'is_number' for word in x]

    for word in x:
        if word not in word_dict:
            word_dict[word] = 1

    content = ' '.join(x)
    f2.write('%s,%s,%s,%s\n' % (movie_id, name, types, content))

f2.close()

f3 = open("imdb/words.csv", "w")
for word in word_dict.keys():
    f3.write("%s\n" % word)
