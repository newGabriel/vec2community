import numpy as np
from cv2 import compareHist


def hist_correl(X1, X2):
    return compareHist(X1, X2, 0)


def hist_chisqr(X1, X2):
    return compareHist(X1, X2, 1)


def hist_intersect(X1, X2):
    return compareHist(X1, X2, 2)


def bhattacharyya(X1, X2):
    return compareHist(X1, X2, 3)


def manhattan(X1: np.ndarray, X2: np.ndarray):
    dif = 0.
    for i, j in zip(X1, X2):
        dif += abs(i - j)
    return dif


def euclides(X1: np.ndarray, X2: np.ndarray):
    dif = 0.
    for i, j in zip(X1, X2):
        dif += (i - j)**2
    return dif**0.5
