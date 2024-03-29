# Testing TextNomalizer and GensimVectorizer
import time
from h_readsqlite import SqliteCorpusReader
from i_vectorizer import GensimVectorizer, TextNormalizer

start_time = time.time()
PATH =  "DB/StackOverflow.sqlite"
corpus_reader = SqliteCorpusReader(path=PATH)

docs = corpus_reader.docs(2022)
normal = TextNormalizer()
normal.fit(docs)

docs = list(normal.transform(docs))
# vect = GensimVectorizer("other/lexicon.pkl", True) # One Hot Encoding
vect = GensimVectorizer("other/lexicon.pkl", False, True)
vect.fit(docs)
docs = vect.transform(docs)
n = 0
for i in docs:
    # print(len(i), i)
    n += 1
print(n)
print(len(i), i)

print("Finished")


"""

vect = GensimVectorizer('lexicon.pkl')
vect.fit(docs)
docs = vect.transform(docs)
print(next(docs))
"""