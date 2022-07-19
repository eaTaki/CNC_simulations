import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from re import U
import numpy as np
from functools import partial
import scipy.integrate
import scipy
import networkx as nx 
import bisect
from progress.bar import IncrementalBar


######################   1   ##########################


# Parameter definitions
V_rest = -70 # mV
V_reset= -80 # mV
R_m    =  10 # Mohm
tau_m  =  10 # ms
V_th   = -54 # mV
I      =  1.7 # nA
T      = 255
gmax   = 50 #nS
tau_s  = 5.4
EsE    = 0 #mV
EsI    = -80 #mV
tau_p  = 2.5
P_max  = .0001


neurons = np.zeros(shape=[3, T])
for neuron in neurons:
    neuron[0] = V_reset

M_7 = np.matrix([ [6, 0, -6], 
                [-9, 9, 0], 
                [0, -9, 9], 
                [6, 0, -6], 
                [-9, 9, 0], 
                [0, -9, 9], 
                [6, 0, -6]])

M_5 = np.matrix([ [8, 0, -8], 
                [-8, 8, 0], 
                [0, -16, 16], 
                [8, 0, -8], 
                [-8, 8, 0]])

M_3 = np.matrix([ [16, 0, -16], 
                [-16, 16, 0],
                [0, -16, 16]])
M = {7: M_7, 5: M_5, 3: M_3}


# Euler method

def AMPA(t):
    return np.e**-(t/tau_s)
def GABA(t):
    return (t/tau_p)*np.e**(1-(t/tau_p))


def ltstSpk(iSpks, t):
    spks = iSpks.copy()
    i = bisect.bisect_left(spks, t)
    return spks[i - 1]

def gateP (t, iSpksV, opt = 0, prec = 0):
    ret = []
    for iSpks in iSpksV[:prec]:
        if(len(iSpks) > 0 and t > iSpks[0]):
            if(opt == 1):
                ret.append(gmax*GABA(t - ltstSpk(iSpks, t))*P_max)
            else:
                ret.append(gmax*AMPA(t - ltstSpk(iSpks, t))*P_max)
        else:
            ret.append(gmax*0)
    return ret

def F_func(u, m, v, num = -1, prec = 0):
    return np.dot(m[num], np.transpose(v))

def inSpikes(U, spikes):
    for i in range(T):
        for n, u in enumerate(U):
            if(u == 0):
                spikes[n] = []
            elif((i%int(T/u)) == (int(T/u) - 1)):
                spikes[n].append(i)
    return spikes

def eulerMeth(option, V_rest1 = V_rest, V_reset1 = V_reset, R_m1 = R_m, tau_m1 = tau_m, V_th1 = V_th, I1 = I, T1 = T, I_E = EsE, AG = 0, u = [0, 0, 0, 0, 0], prec = 0, size = 0):
    
    Ivec = []
    tau_gsra = 100
    spikes = []
    for s in range(prec + size):
        spikes.append([])
    spikes = inSpikes(np.squeeze(np.asarray(u)), spikes)
    for i in range(1,T1):
        for num, neuron in enumerate(neurons):
            dV1 = F_func(u, M[prec].getT(), [y*-((neuron[i - 1] - I_E)*R_m1) for y in gateP(i, spikes, AG, prec = prec)], num = num, prec = prec)#gateP(i, spikes, AG)*(neuron[i - 1] - I_E)*R_m1)
            neuron[i] = neuron[i-1] + dV1/tau_m1
            if neuron[i] > V_th1:
                neuron[i-1] = 50
                neuron[i] = V_reset1
                spikes[prec+num].append(i)

    
    return [10*len(spikes[-3]), 10*len(spikes[-2]), 10*len(spikes[-1])]





######################   2   ##########################





def procedure(matrix):
    return [[x.clip(min=0) for x in row] for row in matrix]

def modIMM(im_arr, prec = 5, size = 3, cmpct = 0):

    ofst = int((prec-size)/2)
    bar = IncrementalBar('Processing', max=len(im_arr), suffix='%(percent)d%% - %(elapsed)d(%(eta)ds remaining)')
    for i, row in enumerate(im_arr):
        ROW = ofst*[np.mean(row[0])/10] + [(np.mean(x)/10) for x in row] + ofst*[np.mean(row[-1])/10]
        for j, col in enumerate(row):
            if(j%size == (size-1) or cmpct):
                aux = ROW[(j-size+1):(j+1+2*ofst)]
                u = np.asmatrix(aux)
                v = eulerMeth(1, u = u, prec = prec, size = size)
                aux.clear()
                if(cmpct):
                    im_arr[i, j] = max(v)
                else:
                    for x, a in enumerate(v):
                        im_arr[i, (j - x)] = a

        bar.next()
    bar.finish()
    return im_arr

def transpIm(im):
    final_im = np.zeros(shape=[len(im), len(im[0]), 3], dtype="uint8")
    for i in range(len(im[0])):
        for j, row in enumerate(im):
            final_im[i, j] = row[i]
    return final_im


im = Image.open("descargaa.jpg") 
im_arr = np.asarray(im)
im_arr = modIMM(transpIm(im_arr), 3, 3, cmpct=1)
im_mod = Image.fromarray(transpIm(im_arr))
im_fin = Image.fromarray(np.hstack((np.asarray(im), im_mod)))
im_fin.show()

