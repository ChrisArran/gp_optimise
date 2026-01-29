import numpy as np
import matplotlib.pyplot as plt
from QEDcascPy_positrons import sim_positrons_angle,sim_rr_angle
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
import gp_optimise
import importlib
import time

var = 'l0'
label = '7.5GeV_20umJitter'

# Sim parameters
Npoints = 10

Nsims = 100
up_scale = 1e3
Nsamples = 1e2

# Expt parameters
Etot = 100 # J
s = 0.25
gamma0 = 1e9/0.511e6 # for 10 J
E0 = 10 # 10 J
gamma_spread = 0.1

# Best compression
duration0 = 25 # fs
waist0 = 2*1.22*0.8 # microns

# Random jitter
t_std = 25e-15
r_std = 20e-6

# Electron energy scaling
gscale = 1

# Set up the variables etc
I0 = Etot/((duration0*1e-15)*(waist0*1e-6)**2) * (4*np.log(2)/np.pi)**1.5
a0 = np.sqrt(I0/2.1378e22)

z_offset = np.random.default_rng().normal(0, 0.5*t_std*3e8, Nsims)
x_offset = np.random.default_rng().normal(0, r_std, Nsims)
y_offset = np.random.default_rng().normal(0, r_std, Nsims)

# Define the MC functions to return positron number and rr rate
# Keep the offsets outside the function so I can use the same consistent offsets throughout

#dims = [{'name':'duration','type':'log-uniform','min':5,'max':100}]
#dims = [{'name':'Esplit','type':'uniform','min':0,'max':1}]
dims = [{'name':'l0','type':'log-uniform','min':0.001,'max':1}]

def estimate_positrons(params): # params = [l0]
    t0 = time.time()
    positron = np.zeros((Nsims))
    for i in range(Nsims):
        gamma = gamma0*((1-s)*Etot/E0)**gscale
        gamma = 7.5e9/0.511e6
        gamma_spread = 0.1
        positron[i] = sim_positrons_angle(a0*np.sqrt(s), gamma, gamma_spread*gamma, angle=15,
                            z_offset=z_offset[i], x_offset=x_offset[i], y_offset=y_offset[i],l0=params[0],
                            duration_fwhm = duration0*1e-15, waist_fwhm = waist0*1e-6,
                            Nsamples=Nsamples, up_scale=up_scale, model="Quantum")

    mu = np.mean(positron)
    sigma = np.std(positron)
    if (not mu>0):
        mu = 1.0/(Nsamples*up_scale)**2
        sigma = 1.0/(Nsamples*up_scale)**1.5

    exponent = np.floor(np.log10(mu))
    t1 = time.time()
    print("Finished %0i sims in %0.2e s at %s=%0.3f, giving (%0.2f+-%0.2f)x10^%0i positrons per electron"
          % (Nsims,t1-t0,dims[0]['name'],params[0],mu/10**exponent,sigma/np.sqrt(Nsims)/10**exponent,exponent))
    logmu = np.log10(mu)
    logsigma = np.sqrt(np.log(1+sigma**2/mu**2)/Nsims)/np.log(10)

    return logmu,logsigma


kernel = 1*RBF(length_scale_bounds=(1e-1, 1e1))# + WhiteKernel(noise_level_bounds=(1e-2, 1e0))
gpo = gp_optimise.Gp_optimise(estimate_positrons,dims,kernel)

explore2 = 0.1

# Run the code
gpo.initialise(Ninitial=3)

X_test = gpo.uniform_Xgrid(100)
y_test,std_test = gpo.predict(X_test)
acq_test2 = gpo.acquisition_function(gpo.X_to_Xnorm(X_test),explore=explore2,acq_fn='EI')
nxt2 = gpo.next_acquisition(Nacq=10,explore=explore2,acq_fn='EI')
acq2 = gpo.acquisition_function(nxt2,explore=explore2,acq_fn='EI')
print('Predicted next point at %s=%0.2f, EI=%0.2f' % (var,gpo.Xnorm_to_X(nxt2)[0,0],acq2[0]))

np.savetxt('%s_example%i_%s.txt'%(var,len(gpo.y),label),np.transpose(np.vstack((gpo.X[:,0],gpo.y,gpo.yerr))),header=' %s\t log(N+/N-)\t error' % (var))
np.savetxt('%s_model%i_%s.txt'%(var,len(gpo.y),label),np.transpose(np.vstack((X_test[:,0],y_test,std_test))),header=' %s\t log(N+/N-)\t error' % (var))
np.savetxt('%s_acq%i_%s.txt'%(var,len(gpo.y),label),np.transpose(np.vstack((X_test[:,0],acq_test2))),header=' %s\t EI' % (var))

N_new = 1
for i in range(Npoints-len(gpo.y)):
    gpo.optimise(N_new,Nacq=10,explore=explore2,acq_fn='EI')
    y_test,std_test = gpo.predict(X_test)
    acq_test2 = gpo.acquisition_function(gpo.X_to_Xnorm(X_test),explore=explore2,acq_fn='EI')
    nxt2 = gpo.next_acquisition(Nacq=10,explore=explore2,acq_fn='EI')
    acq2 = gpo.acquisition_function(nxt2,explore=explore2,acq_fn='EI')
    print('Predicted next point at %s=%0.2f, EI=%0.2f' % (var,gpo.Xnorm_to_X(nxt2)[0,0],acq2[0]))

    np.savetxt('%s_example%i_%s.txt'%(var,len(gpo.y),label),np.transpose(np.vstack((gpo.X[:,0],gpo.y,gpo.yerr))),header=' %s\t log(N+/N-)\t error' % (var))
    np.savetxt('%s_model%i_%s.txt'%(var,len(gpo.y),label),np.transpose(np.vstack((X_test[:,0],y_test,std_test))),header=' %s\t log(N+/N-)\t error' % (var))
    np.savetxt('%s_acq%i_%s.txt'%(var,len(gpo.y),label),np.transpose(np.vstack((X_test[:,0],acq_test2))),header=' %s\t EI' % (var))

figname = '%s_%s.png' % (var,label)
gpo.example_plot1d([3,4,5],a=0,N=51,explore=0.1,acq_fn='EI',figname=figname,figsize=(10,4))
