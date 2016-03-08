#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def box_filter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)

    imCum = np.cumsum(img, 0)
    imDst[0 : r+1, :] = imCum[r : 2*r+1, :]
    imDst[r+1 : rows-r, :] = imCum[2*r+1 : rows, :] - imCum[0 : rows-2*r-1, :]
    imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1 : rows-r-1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0 : r+1] = imCum[:, r : 2*r+1]
    imDst[:, r+1 : cols-r] = imCum[:, 2*r+1 : cols] - imCum[:, 0 : cols-2*r-1]
    imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1], [r, 1]).T - imCum[:, cols-2*r-1 : cols-r-1]

    return imDst


def guided_filter(I, p, r, eps):
    (rows, cols) = I.shape
    N = box_filter(np.ones([rows, cols]), r)

    meanI = box_filter(I, r) / N
    meanP = box_filter(p, r) / N
    meanIp = box_filter(I * p, r) / N
    covIp = meanIp - meanI * meanP

    meanII = box_filter(I * I, r) / N
    varI = meanII - meanI * meanI

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = box_filter(a, r) / N
    meanB = box_filter(b, r) / N

    q = meanA * I + meanB
    return q

