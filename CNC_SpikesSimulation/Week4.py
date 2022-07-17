from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

max_n = 150   # Maximum count 
max_r = 70    # Maximum rate 
rate  = np.arange(2, max_r)
countFR = np.arange(0,max_n)
countLK = np.arange(0, 50, 0.1)
n_trials = 100
bins     = np.arange(0,max_n+1)-0.5

def myCurve(theta, theta_p = 0, c = 1.0, r_max = 30, K = 20, sigma = 60):
    A = c*r_max
    return A*np.exp(-0.5*(theta-theta_p)**2/(2*sigma**2)) + K

fig, ax = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'height_ratios' : [1, 1]})

entropy_pmf = np.zeros(len(rate))
entropy_hist2 = np.zeros(len(rate))
entropy_hist = np.zeros(len(rate))
probA = []
probB = []
for i, r in enumerate(rate):
    # Generate samples
    s1 = np.random.poisson(r,size=n_trials)
    s2 = np.random.poisson(r,size=n_trials)
    # Generate the probability mass function
    p_pmf = poisson.pmf(countFR,r)
    p1 = np.histogram(s1,bins,density = True)[0]
    p2 = np.histogram(s2,bins,density = True)[0]
    probA.append(p1)
    probB.append(p2)
    # Calculate the entropy
    H_pmf = -np.sum(p_pmf*np.log2(p_pmf))
    H1 = -np.sum(p1[p1>0]*np.log2(p1[p1>0]))
    H2 = -np.sum(p2[p2>0]*np.log2(p2[p2>0]))
    entropy_pmf[i] = H_pmf
    entropy_hist[i] = H1
    entropy_hist2[i] = H2

ax[0][0].plot(entropy_pmf, label='Theoretical Entropy Function')
ax[0][0].plot(entropy_hist, label='Simulated Entropy Function')
ax[0][0].set_title('Graph 1: Entropy levels on changing firing rate')
ax[0][0].set_xlabel('Firing Rate')
ax[0][0].set_ylabel('Entropy(H)')
ax[0][0].legend(fontsize = 'x-small')

ax[0][1].plot(entropy_hist, label='Neuron A')
ax[0][1].plot(entropy_hist2, label='Neuron B')

entropy_histj = np.zeros(len(rate))
for i, (pA, pB) in enumerate(zip(probA, probB)):
    p = pA*pB + (1-pA)*(1-pB)
    H = -np.sum(p[p>0]*np.log2(p[p>0]))
    entropy_histj[i] = H

entropy_histj = (entropy_hist + entropy_hist2) - entropy_histj

avgS = []
avgS.append(sum(entropy_hist)/len(entropy_hist))
avgS.append(sum(entropy_hist2)/len(entropy_hist2))
avgS.append(sum(entropy_histj)/len(entropy_histj))
avgS.append(avgS[0]+avgS[1]-avgS[2])
toSay = "avgA = {}".format('%.3f'%(avgS[0])) + '\n'
toSay += "avgB = {}".format('%.3f'%(avgS[1])) + '\n'
toSay += "avgJ = {}".format('%.3f'%(avgS[2]))
ax[0][1].text(0.9,0.8,toSay,horizontalalignment='center', fontsize = 'x-small',
     verticalalignment='center', transform = ax[0][1].transAxes)

ax[0][1].plot(entropy_histj, label='Mutual Entropy Function')
ax[0][1].set_title('Graph 2: Mutual Entropy of two neurons')
ax[0][1].set_xlabel('Firing Rate')
ax[0][1].set_ylabel('Entropy(H)')
ax[0][1].legend(fontsize = 'x-small')

entropy_dist1 = np.zeros(len(countLK))
entropy_dist2 = np.zeros(len(countLK))
entropy_distA = np.zeros(len(countLK))
entropy_distB = np.zeros(len(countLK))
KVD = []
for i, diff in enumerate(countLK):
    
    sA = np.random.poisson(myCurve(0, theta_p=50+diff, c=1),size=n_trials)
    sB = np.random.poisson(myCurve(0, theta_p=50-diff, c=1),size=n_trials)
    pA = np.histogram(sA,bins,density = True)[0]
    pB = np.histogram(sB,bins,density = True)[0]
    
    p1 = pA*pB + (1-pA)*(1-pB)
    p2 = 1 - p1
    KVD.append(np.sum(pA[(pB>0) & (pA>0)]*np.log2(pA[(pB>0) & (pA>0)]/pB[(pB>0) & (pA>0)])))

    Ha = -np.sum(pA[pA>0]*np.log2(pA[pA>0]))
    Hb = -np.sum(pB[pB>0]*np.log2(pB[pB>0]))
    H1 = Ha + Hb + np.sum(p1[p1>0]*np.log2(p1[p1>0]))
    H2 = -np.sum(p2[p2>0]*np.log2(p2[p2>0]))
    entropy_dist1[i] = H1
    entropy_dist2[i] = H2
    entropy_distA[i] = Ha
    entropy_distB[i] = Hb

ax[1][0].plot(entropy_distA, label="Neuron +")
ax[1][0].plot(entropy_distB, label='Neuron -')
ax[1][0].plot(entropy_dist1, label="Mutual Entropy Function")
#ax[1][1].plot(KVD, label="KVD")
ax[1][0].set_title('Graph 3: Entropy levels on changing distributions')
ax[1][0].set_xlabel('Stimulus LikelyHood diference')
ax[1][0].set_ylabel('Entropy(H)')
ax[1][0].legend(fontsize = 'x-small')

ax[1][1].plot(entropy_dist2, color='g')
ax[1][1].set_title('Graph 4: Joint Entropy level on changing distributions')
ax[1][1].set_xlabel('Stimulus LikelyHood diference')
ax[1][1].set_ylabel('Entropy(H)')

plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.5)
plt.show()