import numpy as np
import pandas as pd
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from Scores.score import *
from util.read import *


def clustering(X, y, km):
    h_max = 0
    for i in range(1, 101):
        print('\r'+str(i), end='')
        try:
            km.n_clusters = i
            km.fit(X)
            # print(km.labels_)
            hm = harmonicMean_score(y, km.labels_)
            h_max = max(hm, h_max)
        finally:
            pass
    print('')
    return h_max


bs = read_input_list(sys.argv[1])

kms = []
spc = []
agl = {'ward': [], 'complete': [], 'average': [], 'single': []}
res = pd.DataFrame()

for b in bs:
    df = pd.read_csv(b, header=None)
    # print(df.head())

    print(b)

    ct = list(range(df.shape[1] - 1))
    X = np.array(df[ct])
    y = df[[df.shape[1] - 1]][df.shape[1] - 1]

    # Kmeans
    kms.append(clustering(X, y, KMeans()))

    # Spectral
    spc.append(clustering(X, y, SpectralClustering()))

    # Agglomerate
    for t in ['ward', 'complete', 'average', 'single']:
        agl[t].append(clustering(X, y, AgglomerativeClustering(linkage=t)))

res['KMeans'] = kms
res['Spectral'] = spc
res['Ward'] = agl['ward']
res['Complete'] = agl['complete']
res['Average'] = agl['average']
res['Single'] = agl['single']

res.index = bs

res.to_csv(sys.argv[1]+'Agrupamento.csv')
print(res)
