import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import scipy.integrate
import scipy
# Parameter definitions
V_rest = -70 # mV
V_reset= -80 # mV
R_m    =  10 # Mohm
tau_m  =  10 # ms
V_th   = -54 # mV
I      =  1.7 # nA
T      = 300

spike_times = []
V = np.zeros(T)
V[0] = V_rest

# Euler method
def eulerMeth(option, V_rest1 = V_rest, V_reset1 = V_reset, R_m1 = R_m, tau_m1 = tau_m, V_th1 = V_th, I1 = I, T1 = T):
    tau_gsra = 100
    g_sra = 0
    firrate = []
    for i in range(1,T1):
        dV1 = V_rest1 - V[i-1] - g_sra*(V[i-1] - V_rest1) + R_m1*I1
        if(option == 3):
            g_sra = g_sra - (g_sra/tau_gsra)
        V[i] = V[i-1] + dV1/tau_m1

        if V[i] > V_th1:
            V[i-1] = 50
            V[i] = V_reset1
            if(option == 3):
                g_sra += 0.06
            firrate.append(i)

    if(option == 0 or option == 3):
            return V
    if(option == 1):
        if(len(firrate)>2):
            return 1000/(firrate[1]-firrate[0])
        else:
            return 0


fig, ax = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'height_ratios' : [1, 1]})
ax[0][0].plot(eulerMeth(option = 0), color='#DCDCDC', linestyle='dashed',  label='Simple Model')
ax[0][0].plot(eulerMeth(option = 3), label='Spike Rate Adaptation')
ax[0][0].set_title('Graph 1: Simple Integrate-and-fire model')
ax[0][0].set_xlabel('t(ms)')
ax[0][0].set_ylabel('Voltage(mV)')
ax[0][0].legend(fontsize = 'x-small', loc='upper right')


I_var = np.arange(1, 3, 0.1)
firing_rate = []
for i in I_var:
    firing_rate.append(eulerMeth(option = 1, I1 = i))
ax[0][1].plot(firing_rate)
ax[0][1].set_xticks(np.arange(24, step = 4), np.arange(10, 34, 4)/10)
ax[0][1].set_title('Graph 2: Firing Rate changes to Current')
ax[0][1].set_xlabel('Current(nA)')
ax[0][1].set_ylabel('Firing Rate(Hz)')

finalTime = 300
sc = {"a":-.3, "b":1.4, "tau":20, "I":0.23}

def PhasePlane(x, t, a, b, tau, I):
    return np.array([x[0] - x[0]**3 - x[1] + I, 
                     (x[0] - a - b * x[1])/tau])

def get_displacement(param, dmax=0.5,time_span=np.linspace(0,finalTime, 1000), number=20):
    # We start from the resting point...
    ic = scipy.integrate.odeint(partial(PhasePlane, **param),
                                                      y0=[0,0],
                                                      t= np.linspace(0,999, 1000))[-1]
    # and do some displacement of the potential.  
    return (scipy.integrate.odeint(partial(PhasePlane, **param),
                                                    y0=ic+np.array([0,0]),
                                                    t=time_span))

traject = get_displacement(sc, number=1, time_span=np.linspace(0, finalTime, num=1500), dmax=0.5)

ax[1][0].set(xlabel='Time', ylabel='v, w',
                title="Graph 3: Phase Plane(Fitzhugh-Nagumo) model")

v = ax[1][0].plot(np.linspace(0, finalTime, num=1500),traject[:,0], color='C0')
w = ax[1][0].plot(np.linspace(0, finalTime, num=1500),traject[:,1], color='C1', alpha=.5)
ax[1][0].legend([v[0],w[0]],['v','w'])

def plot_vector_field(ax, param, xrange, yrange, steps=50):
    # Compute the vector field
    x = np.linspace(xrange[0], xrange[1], steps)
    y = np.linspace(yrange[0], yrange[1], steps)
    X,Y = np.meshgrid(x,y)
    
    dx,dy = PhasePlane([X,Y],0,**param)   
    
    # streamplot is an alternative to quiver 
    # that looks nicer when your vector filed is
    # continuous.
    ax.streamplot(X,Y,dx, dy, color=(0,0,0,.1))
    
    ax.set(xlim=(xrange[0], xrange[1]), ylim=(yrange[0], yrange[1]))

xrange = (-1, 1)
yrange = [(1/sc['b'])*(x-sc['a']) for x in xrange]
plot_vector_field(ax[1][1], sc, xrange, yrange)
ax[1][1].set(xlabel='v', ylabel='w',
        title="Graph 4: Fitzhugh-Nagumo vector field(gradient)")

plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.5)
plt.show()
