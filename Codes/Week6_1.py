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
gmax   = 50 #nS
tau_s  = 5.4
EsE    = 0 #mV
EsI    = -80 #mV
tau_p  = 2.5
P_max  = .0001


V = np.zeros(T)
V[0] = V_rest
V2 = np.zeros(T)
V2[0] = V_reset
neurons = [V, V2]
#neurons[1][0] = 50

# Euler method

def AMPA(t):
    return np.e**-(t/tau_s)
def GABA(t):
    return (t/tau_p)*np.e**(1-(t/tau_p))

def gateP (t, iSpks):
    if(len(iSpks) > 0):
        return gmax*GABA(t - iSpks[-1])*P_max
    else:
        return gmax*0

def eulerMeth(option, V_rest1 = V_rest, V_reset1 = V_reset, R_m1 = R_m, tau_m1 = tau_m, V_th1 = V_th, I1 = I, T1 = T, I_E = 0):
    tau_gsra = 100
    spikes = [[], []]
    for i in range(1,T1):
        for num, neuron in enumerate(neurons):
            dV1 = V_rest1 - neuron[i-1] + R_m1*I1
            if(option == 1):
                dV1 -= gateP(i, spikes[(num+1)%2])*(neuron[i - 1] - I_E)*R_m1
            neuron[i] = neuron[i-1] + dV1/tau_m1

            if neuron[i] > V_th1:
                neuron[i-1] = 50
                neuron[i] = V_reset1
                spikes[num].append(i)

    return neurons


fig, ax = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'height_ratios' : [1, 1]})
ax[0][0].plot(eulerMeth(0)[0],  label='V0=-70')
ax[0][0].plot(eulerMeth(0)[1],  label='V0=-80')
ax[0][0].set_title('Graph 1: Simple Integrate-and-fire model')
ax[0][0].set_xlabel('t(ms)')
ax[0][0].set_ylabel('Voltage(mV)')
ax[0][0].legend(fontsize = 'x-small', loc='upper right')

neurons = [V, V2]
n = eulerMeth(1, I_E = EsE)

ax[0][1].plot(n[0],  label='V0=-70')
ax[0][1].plot(n[1],  label='V0=-80')
ax[0][1].set_title('Graph 2: Integrate-and-fire model(Excitatory)' )
ax[0][1].set_xlabel('t(ms)')
ax[0][1].set_ylabel('Voltage(mV)')
ax[0][1].legend(fontsize = 'x-small', loc='upper right')

neurons = [V, V2]
n = eulerMeth(1, I_E = EsI)

ax[1][0].plot(n[0],  label='V0=-70')
ax[1][0].plot(n[1],  label='V0=-80')
ax[1][0].set_title('Graph 3: Integrate-and-fire model(Inhibitory)')
ax[1][0].set_xlabel('t(ms)')
ax[1][0].set_ylabel('Voltage(mV)')
ax[1][0].legend(fontsize = 'x-small', loc='upper right')


plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.5)
plt.show()
