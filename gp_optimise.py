#!/usr/bin/python3
# Chris Arran, March 2024
# Bayesian Optimisation using Gaussian Process Regression

import numpy as np
from scipy.stats import uniform,loguniform,norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel

class Gp_optimise:

	def __init__(self,fun,dims,kernel=1*RBF(length_scale_bounds=(1e-1, 1e1))):
	# Constructor for the optimizer object
	# 	fun is a noisy function which takes M inputs
	# 	dims is a length M list of dictionaries containing details of the inputs
	#		dims['name'] is the name of the dimension
	#		dims['type'] is one of 'uniform', 'log-uniform', or 'normal'
	#		dims['min'] is the minimum of a uniform or log-uniform distribution
	#		dims['max'] is the minimum of a uniform or log-uniform distribution
	#		dims['mean'] is the mean of a normal distribution
	#		dims['std'] is the standard deviation of a normal distribution

		self.fun = fun
		self.dims = dims
		self.kernel = kernel

	def create_Xgrid(self,N):
	# Create a list of points randomly along each of dims
		X = np.zeros((N,len(self.dims)))
		for i,d in enumerate(self.dims):
			if d['type'] == 'uniform':
				X[:,i] = uniform.rvs(loc=d['min'],scale=d['max']-d['min'],size=N)
			elif d['type'] == 'log-uniform':
				X[:,i] = loguniform.rvs(d['min'],d['max'],size=N)
			elif d['type'] == 'norm':
				X[:,i] = norm.rvs(loc=d['mean'],scale=d['std'],size=N)
			else:
				raise Exception('Not a recognised distribution for input dimension %s: %s' % (d['name'],d['type']))
		return X
		

	def initialise(self,Ninitial=10,n_restarts=10):
	# Initialises a Gaussian Process Regressor
	# 	Ninitial is the number of initial measurement points to build the regressor with
	#	n_restarts is the number of times to restart the GPR optimiser

		self.X = self.create_Xgrid(Ninitial)
		self.y = np.zeros((Ninitial))
				
		for n in range(Ninitial):
			self.y[n] = self.fun(self.X[n,:])
			
		self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=n_restarts, normalize_y=True)
		self.gaussian_process.fit(self.X, self.y)
		

	def acquisition_function(self,X_acq,explore=1.0,acq_fn='UCB'):
	# Returns the chosen acquisition function for finding the next place to sample
	# Thanks to Martin Krasser at krasserm.github.io
	#	X_acq is the place to calculate the acquisition function
	#	explore describes the amount the algorithm should weight exploration over optimisation
	#	model describes what model to use (UCB for Upper Confidence Bound, EI for expected improvement)
	
		y_acq,sigma_acq = self.gaussian_process.predict(X_acq, return_std=True)
		
		if (acq_fn=='UCB'):	# Maximise upper confidence bound
			acq = y_acq + explore*sigma_acq
			
		elif (acq_fn=='EI'):	# Maximise expected improvement
			y_exp = self.gaussian_process.predict(self.X)
			y_max = np.max(y_exp)
			
			imp = y_acq - y_max - explore
			z = imp / sigma_acq  
			acq = imp*norm.cdf(z) + sigma_acq*norm.pdf(z)
			acq[sigma_acq==0] = 0
			
		return acq


	def next_acquisition(self,Nacq=10,explore=1.0,acq_fn='UCB'):
	# Finds the next place to acquire with an option for what acquisition function to use
	#	Nacq is the number of places the algorithm starts from to choose the best next place
	# 	explore describes the amount the algorithm should weight exploration over optimisation
	#	model describes what model to use (UCB for Upper Confidence Bound, EI for expected improvement)
	
		X_start = self.create_Xgrid(Nacq)
		bounds = []
		for d in self.dims:
			bounds.append((d['min'],d['max']))
		
		def min_acq_fn(X_acq): # make acquisition function negative to use minimise
			return -self.acquisition_function(X_acq.reshape(1,-1),explore=explore,acq_fn=acq_fn)
		
		min_val = min_acq_fn(X_start[0])
		min_x = X_start[0]
		for x0 in X_start:
			res = minimize(min_acq_fn, x0=x0, bounds=bounds, method='L-BFGS-B')
			if res.fun < min_val:
				min_val = res.fun[0]
				min_x = res.x
			
		return min_x.reshape(1,-1)

			
	def optimise(self,N,Nacq=10,explore=1,acq_fn='UCB'):
	# Iteratively improve the GPR using measurements in a place chosen by the acquisition function
	#	N gives the number of iterations to use
		
		sz = np.shape(self.X)
		self.X = np.resize(self.X,(sz[0]+N,sz[1]))
		self.y = np.resize(self.y,(sz[0]+N))
		
		for n in range(N):
			X_new = self.next_acquisition(Nacq=Nacq,explore=1,acq_fn='UCB')
			y_new = self.fun(X_new)
			self.X[sz[0]+n,:] = X_new
			self.y[sz[0]+n] = y_new
			self.gaussian_process.fit(self.X[:sz[0]+n+1,:],self.y[:sz[0]+n+1])
			
			
