
import numpy as np
import statistics
import matplotlib.pyplot as plt

# Function for modeling the firing rate. Parameters as described in text
def r(theta, theta_p = 0, c = 1.0, r_max = 30, K = 20, sigma = 60):
    A = c*r_max
    return A*np.exp(-0.5*(theta-theta_p)**2/(2*sigma**2)) + K
    
# Simulate n_trials with the preferred and the orthogonal stimulus
n_trials = 500
r_plus = np.random.poisson(r(0, c=1), n_trials)
r_min  = np.random.poisson(r(90,c=1), n_trials) 
r_precision = 0.001


#simulated response function f(theta = stimulus)
t1 = np.arange(-100, 100, r_precision)
t2 = np.arange(0, 1, r_precision)

fig, ax = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'height_ratios' : [1, 1]})
ax[0][0].set_title('Graph 1: Simulated MT neuron tuning curve')
ax[0][0].set_xlabel('Stimulus')
ax[0][0].set_ylabel('Firing Rate(spikes/s)')
ax[0][0].plot(r(t1), label='c=1')
ax[0][0].plot(r(t1, c=0.6), label='c=0.6')
ax[0][0].set_xticks(np.arange(2*100*(1/r_precision) + 2*10*(1/r_precision), 
                              step = 2*10*(1/r_precision)), np.arange(start = -100, stop = 120, step = 20))
ax[0][0].legend(fontsize = 'x-small')

#simulating spikes count distribution 
kwargs = dict(alpha=0.3, bins=40, ec="k")
ax[1][0].hist(r_plus, **kwargs, label='r_plus distribution')
ax[1][0].hist(r_min, **kwargs, label='r_min distribution')
ax[1][0].set_title('Graph 2: Spike count distribution')
ax[1][0].set_xlabel('Spike count')
ax[1][0].set_ylabel('Frequency')

#calculating optimal threshold and proportion of correct predictions of the threshold 
r_threshold = statistics.mean([statistics.mean(r_plus), statistics.mean(r_min)])
ax[1][0].axvline(x=r_threshold, label='threshold at {}'.format(r_threshold), c='k', ls=':')
ax[1][0].legend(fontsize = 'x-small')
def rProp():
    counter = 0
    for pp in r_plus:
        if pp >= r_threshold:
            counter+=1
    for mm in r_min:
        if mm < r_threshold:
            counter+=1
    r_min0 = r_min
    r_min0[r_min0 < r_threshold] = 1
    return counter/(n_trials*2)
toSay = 'Acc: ' + str(rProp())
ax[1][0].text(0.9,0.65,toSay,horizontalalignment='center', fontsize = 'x-small',
     verticalalignment='center', transform = ax[1][0].transAxes)


#calculating again with coherence = 0
r_plus = np.random.poisson(r(0, c=0), n_trials)
r_min  = np.random.poisson(r(90,c=0), n_trials)
r_threshold = statistics.mean([statistics.mean(r_plus), statistics.mean(r_min)])
kwargs = dict(alpha=0.3, bins=40, ec="k")
ax[0][1].hist(r_plus, **kwargs)
ax[0][1].hist(r_min, **kwargs)
ax[0][1].axvline(x=r_threshold, label='threshold at {}'.format(r_threshold), c='k', ls=':')
ax[0][1].legend(fontsize = 'x-small')
toSay = 'Acc: ' + str(rProp())
ax[0][1].text(0.9,0.8,toSay,horizontalalignment='center', fontsize = 'x-small',
     verticalalignment='center', transform = ax[0][1].transAxes)
ax[0][1].set_title('Graph 3: Spike count distribution(c = 0)')


#calculating proportion of correct predictions of the threshold varying c
r_plus = np.random.poisson(r(0, c=1), n_trials)
r_min  = np.random.poisson(r(90,c=1), n_trials)
r_threshold = statistics.mean([statistics.mean(r_plus), statistics.mean(r_min)])
precision2 = 0.001
def r2(c3):
    it = 0
    
    counter2 = [0]*int((1/precision2))
    for ca in c3:
        r2_plus = np.random.poisson(r(0, c=ca), n_trials)
        r2_min  = np.random.poisson(r(90,c=ca), n_trials) 
        for pp in r2_plus:
            if pp >= r_threshold:
                counter2[it]+=1
        for mm in r2_min:
            if mm < r_threshold:
                counter2[it]+=1
        counter2[it]/=(n_trials*2)
        it+=1
    return counter2

ax[1][1].plot(r2(np.arange(0,1, step = precision2)), linewidth = 1)
ax[1][1].set_xticks(np.arange(12*(0.1/precision2), step = 2*(0.1/precision2)), np.arange(120, step = 20))
ax[1][1].set_yticks(np.arange(0.499, 1.099, 0.1), np.arange(50, 110, 10))
ax[1][1].set_title('Graph 4: Threshold accuracy')
ax[1][1].set_ylabel('Accuracy(%)')
ax[1][1].set_xlabel('coherence level(%)')


plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.25)
plt.show()

