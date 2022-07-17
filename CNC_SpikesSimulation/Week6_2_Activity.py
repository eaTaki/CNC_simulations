from re import U
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import scipy.integrate
import scipy
import networkx as nx 

# Parameter definitions
V_rest = -70 # mV
V_reset= -80 # mV
R_m    =  10 # Mohm
tau_m  =  10 # ms
V_th   = -54 # mV
I      =  1.7 # nA
T      = 300
gmax   = 50 #nS
tau_s  = 5.4
EsE    = 0 #mV
EsI    = -80 #mV
tau_p  = 2.5
P_max  = .0001


neurons = np.zeros(shape=[8, T])
for neuron in neurons:
    neuron[0] = V_reset
#neurons[1][0] = 50

W = np.matrix([ [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]])

M = np.matrix([ [0, 0, 0, 0, 0, 9, 0, -9],
                [0, 0, 0, 0, 0, -9, 9, 0],
                [0, 0, 0, 0, 0, 0, -18, 18],
                [0, 0, 0, 0, 0, 9, 0, -9],
                [0, 0, 0, 0, 0, -9, 9, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]])
u1 = [0, 1 , 2 , 1 ,0]
u2 = [75, 60 , 0 , 60 ,75]
u3 = [0, 12, 25, 35, 50, 0, 0, 0]
au4 = [30, 30 , 30 , 0 , 0]
u5 = [0, 0 , 0 , 60 , 60]
#u = [1, 4.5, 4.5, 4.5, 1]

G = nx.from_numpy_matrix((M), create_using=nx.DiGraph)
raw_labels = ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8"]
G = nx.relabel_nodes(G, dict(zip(range(8), raw_labels)))
#layout = nx.shell_layout(G)
layout = nx.bipartite_layout(G, raw_labels[:5], align="horizontal")
labels = nx.get_edge_attributes(G, "weight")
lab_node = dict(zip(range(8), raw_labels))

nx.draw(G, layout, with_labels=True)
nx.draw_networkx_nodes(G, layout)
nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels, alpha = 1, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white'), label_pos = 0.2)
#nx.draw_networkx_labels(G, layout, labels=lab_node, font_size=10, font_family='sans-serif')

#plt.show(block=False)

# Euler method

def AMPA(t):
    return np.e**-(t/tau_s)
def GABA(t):
    return (t/tau_p)*np.e**(1-(t/tau_p))

def getI(I, dt, w, u):
    u = np.asmatrix(u)
    dI = -I + np.dot(w, u.getT())[0, 0]
    return (I + (dI/dt))

def gateP (t, iSpksV, opt = 0):
    ret = []
    for iSpks in iSpksV:
        if(len(iSpks) > 0):
            if(opt == 1):
                ret.append(gmax*GABA(t - iSpks[-1])*P_max)
            else:
                ret.append(gmax*AMPA(t - iSpks[-1])*P_max)
        else:
            ret.append(gmax*0)
    return ret

def F_func(w, u, m, v):
    return np.dot(w, np.transpose(u)) + np.dot(m, np.transpose(v))

def eulerMeth(option, V_rest1 = V_rest, V_reset1 = V_reset, R_m1 = R_m, tau_m1 = tau_m, V_th1 = V_th, I1 = I, T1 = T, I_E = 0, AG = 0, u = u1):
    Ivec = []
    tau_gsra = 100
    spikes = []
    for s in range(len(neurons)):
        spikes.append([])
    for i in range(1,T1):
        for num, neuron in enumerate(neurons):
            dV1 = F_func(W.getT()[num], u, M.getT()[num], [y*-((neuron[i - 1] - I_E)*R_m1) for y in gateP(i, spikes, AG)])#gateP(i, spikes, AG)*(neuron[i - 1] - I_E)*R_m1)
            neuron[i] = neuron[i-1] + dV1/tau_m1
            if neuron[i] > V_th1:
                neuron[i-1] = 50
                neuron[i] = V_reset1
                spikes[num].append(i)

    return [neurons, Ivec, spikes]

fig, ax = plt.subplots(8, 2, figsize=(10, 6))
f = eulerMeth(1, u = u3)

for i in range(len(neurons)):
    ax[i, 0].plot(f[0][i],  label=f"N{i+1}")
    ax[i, 0].legend(fontsize = 'x-small', loc='upper right')


#ax[0][0].set_title('Graph 1: Simple Integrate-and-fire model')
ax[4][0].set_xlabel('t(ms)')
ax[int(len(neurons)/2)][0].set_ylabel('Voltage(mV)')
ax[0][0].legend(fontsize = 'x-small', loc='upper right')

x = range(T)

def B(ss):
    y = np.zeros(T)
    for i in ss:
        y[i] = 1
    return y

for i in range(len(neurons)):
    ax[i, 1].bar(x, B(f[2][i]),  label=f"N{i+1}")


for i in range(len(neurons)):
    lenS = (len(f[2][i]))
    toWrite = f"Spks={lenS}"
    if(lenS > 2):
        fr = f[2][i][-1] - f[2][i][-2]
        toWrite += f"\nFr={fr}"
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=2)
    t = ax[i, 1].text(270, 0.25, toWrite, ha="center", va="center",
                size=8,
                bbox=bbox_props)
    ax[i, 0].set(ylim=(-100, 100))
    ax[i, 1].set(xlim=(0, 300), ylim=(0, 1))
    ax[i, 1].axes.yaxis.set_visible(False)

plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.5)
plt.show()
