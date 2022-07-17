import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image

W = np.matrix([ [1, 0, 0, 0, -1],
                [-1, 1, 0, 0, 0],
                [0, -1, 1, 0, 0],
                [0, 0, -1, 1, 0],
                [0, 0, 0, -1, 1]])

W1 = np.matrix([ [-1, 0, 0, 0, 1],
                [1, -1, 0, 0, 0],
                [0, 1, -1, 0, 0],
                [0, 0, 1, -1, 0],
                [0, 0, 0, 1, -1]])


im = Image.open("descarga.jpg") 
im_arr = np.asarray(im)
 
aux = []

def procedure(matrix):
    return [[abs(x) for x in row] for row in matrix]

for i, row in enumerate(im_arr):
    for j, col in enumerate(row):
        aux.append(np.mean(col))
        if(j%5 == 4):
            u = np.asmatrix(aux)
            v = np.dot(W, u.getT())
            v = procedure(v)
            aux.clear()
            for x, a in enumerate(v):
                im_arr[i, (j - x)] = v[(x+4)%5]


im_mod = Image.fromarray(im_arr)
im_mod.save('temp.png')
open("temp.png")

