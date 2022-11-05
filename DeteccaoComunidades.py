import sys
import pandas as pd
from Scores.score import *
from graphGen.graphGen import *
from util.read import *
from util.similarity import *


def community(X, y, g, cm):
    h_max = 0.
    if g not in [eNGraph, eSKnnGraph, eMKnnGraph, eSKnnMST, eMKnnMST]:
        for i in range(2, 31):
            print("\r"+str(i), end='')
            gr = g(X, i)
            cl = cm(gr)
            if type(cl) != ig.VertexClustering:
                cl = cl.as_clustering()
            h_max = max(h_max, media_harmonica(cl, y))
    else:
        md = 0.
        for i in X:
            for j in X:
                md += euclides(i, j)
        md /= (len(X)-1)*len(X)
        if g is eNGraph:
            for i in range(1, 11):
                print("\r"+str(i/10.), end='')
                gr = g(X, (i/10.)*md)
                cl = cm(gr)
                if type(cl) != ig.VertexClustering:
                    cl = cl.as_clustering()
                h_max = max(h_max, media_harmonica(cl, y))
        else:
            for i in range(2, 31):
                for j in range(1, 11):
                    print("\r"+str(i),(j/10.0), end='')
                    gr = g(X, i,(j/10.)*md)
                    cl = cm(gr)
                    if type(cl) != ig.VertexClustering:
                        cl = cl.as_clustering()
                    h_max = max(h_max, media_harmonica(cl, y))
    print("")
    return h_max


fg = []
wt = []
lp = []
ml = []
ip = []

bs = read_input_list(sys.argv[1])

for i in [sKnnGraph, mKnnGraph, eNGraph, eSKnnGraph, eMKnnGraph, eSKnnMST, eMKnnMST,]:
    fg = []
    lp = []
    wt = []
    ml = []
    ip = []

    for b in bs:
        df = pd.read_csv(b, header=None)

        print(b)

        ct = list(range(df.shape[1] - 1))
        X = np.array(df[ct])
        y = df[[df.shape[1] - 1]][df.shape[1] - 1]

        for j, l in zip(
                [ig.Graph.community_fastgreedy, ig.Graph.community_label_propagation, ig.Graph.community_walktrap,
                 ig.Graph.community_multilevel, ig.Graph.community_infomap],
                [fg, lp, wt, ml, ip]):
            l.append(community(X, y, i, j))

    res = pd.DataFrame()

    res["Fastgreedy"] = fg
    res["label propagation"] = lp
    res["walktrap"] = wt
    res["multilevel"] = ml
    res["infomap"] = ip

    res.index = bs

    res.to_csv(sys.argv[1] + i.__name__ + "Comunidade.csv")
