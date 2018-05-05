from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import sys

model_path = 'models/word2vec.gensim.model'

w2v = Word2Vec.load(model_path)

while True:
    print("知りたい言葉は？")
    word=input()
    if word=='Q':
        break
    tmp=w2v.most_similar(word,topn=2)
    word_sim1=tmp[0][0]
    word_sim2=tmp[1][0]
    print("'"+word+"'とは'"+word_sim1+"'で'"+word_sim2+"'のようなものである。\n")
