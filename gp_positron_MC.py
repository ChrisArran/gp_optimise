import numpy as np
import matplotlib.pyplot as plt
from QEDcascPy_positrons import sim_positrons_angle,sim_rr_angle
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
import gp_optimise
import importlib
import time
import pickle

# Sim parameters
Npoints = 10
Nsims = 100
up_scale = 1e3
Nsamples = 1e2

# Expt parameters
Etot = 100 # J
gamma0 = 10e9/0.511e6
gamma_spread = 0.1 * gamma0

# Best compression
duration0 = 25 # fs
waist0 = 2*1.22*0.8 # microns

# Random jitter
t_std = 25e-15
r_std = 10e-6

# Electron energy scaling
gscale = 1

# Set up the variables etc
I0 = Etot/((duration0*1e-15)*(waist0*1e-6)**2) * (4*np.log(2)/np.pi)**1.5
a0 = np.sqrt(I0/2.1378e22)

# Define the MC functions to return positron number and rr rate
# Keep the offsets outside the function so I can use the same consistent offsets throughout

#dims = [{'name':'duration','type':'log-uniform','min':5,'max':100}]
dims = [{'name':'Esplit','type':'uniform','min':0,'max':1},
        {'name':'l0','type':'log-uniform','min':0.001,'max':1},
        {'name':'waist','type':'log-uniform','min':waist0,'max':10*waist0}]

z_offset = np.random.default_rng().normal(0, 0.5*t_std*3e8, Nsims)
x_offset = np.random.default_rng().normal(0, r_std, Nsims)
y_offset = np.random.default_rng().normal(0, r_std, Nsims)

gammas = np.random.default_rng().normal(gamma0, 0.1*gamma0, Nsims)

def estimate_positrons(params): # params = [Energy split,l0,waist]
    t0 = time.time()
    positron = np.zeros((Nsims))
    for i in range(Nsims):
        positron[i] = sim_positrons_angle(a0*np.sqrt(params[0])*(waist0/params[2]),
                            gammas[i]*(1-params[0])**gscale, 0.1*gammas[i]*(1-params[0])**gscale,angle=15, l0=params[1],
                            z_offset=z_offset[i], x_offset=x_offset[i], y_offset=y_offset[i],
                            duration_fwhm = duration0*1e-15, waist_fwhm = params[2]*1e-6,
                            Nsamples=Nsamples, up_scale=up_scale, model="Quantum")

    mu = np.mean(positron)
    sigma = np.std(positron)
    if mu<=0:
        mu = 1.0/(Nsamples*up_scale)**2
        sigma = 1.0/(Nsamples*up_scale)**1.5

    exponent = np.floor(np.log10(mu))
    t1 = time.time()
    print("Finished %0i sims in %0.2e s at %s=%0.2f,%s=%0.3f,%s=%0.1f, giving (%0.2f+-%0.2f)x10^%0i positrons per electron"
          % (Nsims,t1-t0,dims[0]['name'],params[0],dims[1]['name'],params[1],dims[2]['name'],params[2],mu/10**exponent,sigma/np.sqrt(Nsims)/10**exponent,exponent))

    #logmu = np.log(mu**2/np.sqrt(mu**2+sigma**2))
    logmu = np.log10(mu)
    logsigma = np.sqrt(np.log(1+sigma**2/mu**2)/Nsims)/np.log(10)

    return logmu,logsigma

def estimate_rr(params): # params = [Energy split,l0,waist]
    t0 = time.time()
    rr = np.zeros(Nsims)
    Egamma = gamma0*(1-params[0])**gscale
    for i in range(Nsims):
        rr[i] = sim_rr_angle(a0*np.sqrt(params[0])*(waist0/params[2]), 
                            gamma0*(1-params[0])**gscale, gamma_spread,angle=15,l0=params[1], 
                            z_offset=z_offset[i], x_offset=x_offset[i], y_offset=y_offset[i],
                            duration_fwhm = duration0*1e-15, waist_fwhm = params[2]*1e-6,
                            Nsamples=Nsamples, up_scale=up_scale, model="Quantum")

    mu = np.nanmean(rr*Egamma*0.511e-3)
    sigma = np.nanstd(rr*Egamma*0.511e-3)/np.sqrt(Nsims)

    t1 = time.time()
    print("Finished %0i sims in %0.2e s at %s=%0.2f,%s=%0.3f,%s=%0.1f giving %0.3f+-%0.3f GeV in radiation per electron" 
          % (Nsims,t1-t0,dims[0]['name'],params[0],dims[1]['name'],params[1],dims[2]['name'],params[2],mu,sigma))

    return mu,sigma

kernel = 1*RBF(length_scale_bounds=(1e-2, 1e1))# + WhiteKernel(noise_level_bounds=(1e-2, 1e0))
gpo = gp_optimise.Gp_optimise(estimate_positrons,dims,kernel)

# Run the thing
gpo.initialise(Ninitial=Npoints)

for i in range(9):
    gpo.optimise(Npoints,Nacq=100,explore=0.2,acq_fn='EI')

    with open('gp_positron_MC_rejection_Nsim100_jitter10um.pickle','wb') as f:
        pickle.dump(gpo,f)
