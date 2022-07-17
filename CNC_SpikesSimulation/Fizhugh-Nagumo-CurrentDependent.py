
import numpy as np
import math
from functools import partial
import matplotlib.pyplot as plt
import scipy.integrate
import scipy
import collections

fig, ax = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'height_ratios' : [1, 1]})




finalTime = 300
sc1 = {"a":-.3, "b":1.4, "tau":20, "I":0.23}

def PhasePlane1(x, t, a, b, tau, I):
    return np.array([x[0] - x[0]**3 - x[1] + I, 
                     (x[0] - a - b * x[1])/tau])

def get_displacement1(param, dmax=0.5,time_span=np.linspace(0,finalTime, 1000), number=20):
    # We start from the resting point...
    ic = scipy.integrate.odeint(partial(PhasePlane1, **param),
                                                      y0=[0,0],
                                                      t= np.linspace(0,999, 1000))[-1]
    # and do some displacement of the potential.  
    return (scipy.integrate.odeint(partial(PhasePlane1, **param),
                                                    y0=ic+np.array([0,0]),
                                                    t=time_span))

traject1 = get_displacement1(sc1, number=1, time_span=np.linspace(0, finalTime, num=1500), dmax=0.5)
print(type(traject1))

ax[1][0].set(xlabel='Time', ylabel='v, w',
                title="Graph 1: Phase Plane(Fitzhugh-Nagumo) model")

v = ax[0][0].plot(np.linspace(0, finalTime, num=1500),traject1[:,0], color='C0')
w = ax[0][0].plot(np.linspace(0, finalTime, num=1500),traject1[:,1], color='C1', alpha=.5)
ax[0][0].legend([v[0],w[0]],['v','w'])



######################################################################



finalTime = 100
sc = {"a":-.3, "b":1.4, "tau":20}
def PhasePlane(x, t, a, b, tau, I):
    if(hasattr(I, "__len__")):
        if(t <= finalTime):
            return np.array([x[0] - x[0]**3 - x[1] + I[int(t*(1500/finalTime) - 1)], 
                     (x[0] - a - b * x[1])/tau])
        else:
            return np.array([x[0] - x[0]**3 - x[1] + I[-1], 
                     (x[0] - a - b * x[1])/tau])
    else:
        return np.array([x[0] - x[0]**3 - x[1] + I, 
                     (x[0] - a - b * x[1])/tau])

def get_displacement(param, dmax=0.5,time_span=np.linspace(0,finalTime, 1000), number=20, i = np.linspace(0, 0.5, 1000)):
    # We start from the resting point...
    ic = scipy.integrate.odeint(partial(PhasePlane, **param, I = i[0]),
                                                      y0=[0,0],
                                                      t= np.linspace(0,999, 1000))[-1]
    # and do some displacement of the potential.  
    p = partial(PhasePlane, **param, I = i)
    print(time_span[-1])
    num = 0
    return (scipy.integrate.odeint(p ,
                                                    y0=ic+np.array([0,0]),
                                                    t=time_span))





ax[1][0].set(xlabel='Time', ylabel='w',
                title="Graph 3: Phase Plane(Fitzhugh-Nagumo) model")

def getFR(params = sc, I = np.zeros(1500), time = finalTime):
    return get_displacement(params, number=1, time_span=np.linspace(0, finalTime, num=1500), dmax=0.5, i = I)[:,1]

#I = np.zeros(1500)
#I[700:800] = 0.23
#I = np.linspace(0.18, 0.23, 1500)
I = np.cos(np.arange(1500))

w = ax[1][0].plot(np.linspace(0, finalTime, num=1500), getFR(sc, I, finalTime) , color='C1', alpha=.5)

ax[1][1].plot(np.linspace(0, finalTime, num=1500), I)

plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.5)
plt.show()
