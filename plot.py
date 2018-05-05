from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import sys

FONTPATH='font/Osaka.ttc'

model_path = 'models/word2vec.gensim.model'

vectors = Word2Vec.load(model_path)

def draw_2d_2groups(vectors, target1, target2, topn=100):
    similars1 = [w[0] for w in vectors.most_similar(target1, topn=topn)]
    similars1.insert(0, target1)
    similars2 = [w[0] for w in vectors.most_similar(target2, topn=topn)]
    similars2.insert(0, target2)
    print(similars1)
    print(similars2)
    similars = similars1 + similars2
    colors = ['b']+['g']*(topn)+ ['r']+['orangered']*(topn+1)
    X = [vectors.wv[w] for w in similars]
    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(X)
    prop = font_manager.FontProperties(fname=FONTPATH)
    plt.figure(figsize=(20,20))
    plt.scatter(Y[:, 0], Y[:,1], color=colors)
    for w, x, y, c in zip(similars[:], Y[:, 0], Y[:,1], colors):
        plt.annotate(w, xy=(x, y), xytext=(3,3), textcoords='offset points', fontproperties=prop, fontsize=16, color=c)
    plt.show()

args=sys.argv
word1=args[1]
word2=args[2]

draw_2d_2groups(vectors,word1,word2,topn=30)
