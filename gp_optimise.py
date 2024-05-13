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
	# Create a list of points randomly along each of dims according to given distributions
	# X is the real space point, Xnorm is the cdf, so uniform and normalised to 0<=Xnorm<=1
		X = np.zeros((N,len(self.dims)))
		Xnorm = np.zeros((N,len(self.dims)))
		for i,d in enumerate(self.dims):
			if d['type'] == 'uniform':
				X[:,i] = uniform.rvs(loc=d['min'],scale=d['max']-d['min'],size=N)
				Xnorm[:,i] = uniform.cdf(X[:,i],loc=d['min'],scale=d['max']-d['min'])
			elif d['type'] == 'log-uniform':
				X[:,i] = loguniform.rvs(a=d['min'],b=d['max'],size=N)
				Xnorm[:,i] = loguniform.cdf(X[:,i],a=d['min'],b=d['max'])
			elif d['type'] == 'norm':
				X[:,i] = norm.rvs(loc=d['mean'],scale=d['std'],size=N)
				Xnorm[:,i] = norm.cdf(X[:,i],loc=d['mean'],scale=d['std'])
			else:
				raise Exception('Not a recognised distribution for input dimension %s: %s' % (d['name'],d['type']))
		return X,Xnorm

	def Xnorm_to_X(self,Xnorm):
	# Convert from normalised units Xnorm, the cdf, to real space units X
		X = np.zeros_like(Xnorm)
		for i,d in enumerate(self.dims):
			if d['type'] == 'uniform':
				X[:,i] = uniform.ppf(Xnorm[:,i],loc=d['min'],scale=d['max']-d['min'])
			elif d['type'] == 'log-uniform':
				X[:,i] = loguniform.ppf(Xnorm[:,i],a=d['min'],b=d['max'])
			elif d['type'] == 'norm':
				X[:,i] = norm.ppf(Xnorm[:,i],loc=d['mean'],scale=d['std'])
			else:
				raise Exception('Not a recognised distribution for input dimension %s: %s' % (d['name'],d['type']))
		return X

	def X_to_Xnorm(self,X):
	# Convert from real space units X to normalised units, Xnorm, the cdf
		Xnorm = np.zeros_like(X)
		for i,d in enumerate(self.dims):
			if d['type'] == 'uniform':
				Xnorm[:,i] = uniform.cdf(X[:,i],loc=d['min'],scale=d['max']-d['min'])
			elif d['type'] == 'log-uniform':
				Xnorm[:,i] = loguniform.cdf(X[:,i],a=d['min'],b=d['max'])
			elif d['type'] == 'norm':
				Xnorm[:,i] = norm.cdf(X[:,i],loc=d['mean'],scale=d['std'])
			else:
				raise Exception('Not a recognised distribution for input dimension %s: %s' % (d['name'],d['type']))
		return Xnorm
		

	def initialise(self,Ninitial=10,n_restarts=10):
	# Initialises a Gaussian Process Regressor
	# 	Ninitial is the number of initial measurement points to build the regressor with
	#	n_restarts is the number of times to restart the GPR optimiser

		self.X,self.Xnorm = self.create_Xgrid(Ninitial)
		self.y = np.zeros((Ninitial))
		self.yerr = np.zeros((Ninitial))
				
		for n in range(Ninitial):
			self.y[n],self.yerr[n] = self.fun(self.X[n,:])
			
		self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=n_restarts, normalize_y=True, alpha=self.yerr**2)
		self.gaussian_process.fit(self.Xnorm, self.y)
		

	def acquisition_function(self,Xnorm_acq,explore=1.0,acq_fn='UCB'):
	# Returns the chosen acquisition function for finding the next place to sample
	# Thanks to Martin Krasser at krasserm.github.io
	#	Xnorm_acq is the place to calculate the acquisition function (in normalised units)
	#	explore describes the amount the algorithm should weight exploration over optimisation
	#	model describes what model to use (UCB for Upper Confidence Bound, EI for expected improvement)
	
		y_acq,sigma_acq = self.gaussian_process.predict(Xnorm_acq, return_std=True)
		
		if (acq_fn=='UCB'):	# Maximise upper confidence bound
			acq = y_acq + explore*sigma_acq
			
		elif (acq_fn=='EI'):	# Maximise expected improvement
			y_exp = self.gaussian_process.predict(self.Xnorm)
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
	
		_,Xnorm_start = self.create_Xgrid(Nacq)
		bounds = []
		for d in self.dims:
			bounds.append((d['min'],d['max']))
		
		def min_acq_fn(X_acq): # make acquisition function negative to use minimise
			return -self.acquisition_function(X_acq.reshape(1,-1),explore=explore,acq_fn=acq_fn)
		
		min_val = min_acq_fn(Xnorm_start[0,:])
		min_x = Xnorm_start[0,:]
		for x0 in Xnorm_start:
			res = minimize(min_acq_fn, x0=x0, bounds=bounds, method='L-BFGS-B')
			if res.fun < min_val:
				min_val = res.fun[0]
				min_x = res.x

		Xnorm_new = min_x.reshape(1,-1)

		return Xnorm_new

			
	def optimise(self,N,Nacq=10,explore=1,acq_fn='UCB'):
	# Iteratively improve the GPR using measurements in a place chosen by the acquisition function
	#	N gives the number of iterations to use
		
		sz = np.shape(self.X)
		self.X = np.resize(self.X,(sz[0]+N,sz[1]))
		self.Xnorm = np.resize(self.Xnorm,(sz[0]+N,sz[1]))
		self.y = np.resize(self.y,(sz[0]+N))
		self.yerr = np.resize(self.yerr,(sz[0]+N))
		
		for n in range(N):
			Xnorm_new = self.next_acquisition(Nacq=Nacq,explore=1,acq_fn='UCB')
			X_new = self.Xnorm_to_X(Xnorm_new)
			y_new,yerr_new = self.fun(X_new)
			self.X[sz[0]+n,:] = X_new
			self.Xnorm[sz[0]+n,:] = Xnorm_new
			self.y[sz[0]+n] = y_new
			self.yerr[sz[0]+n] = yerr_new
			self.gaussian_process.alpha = self.yerr[:sz[0]+n+1]**2
			self.gaussian_process.fit(self.Xnorm[:sz[0]+n+1,:],self.y[:sz[0]+n+1])					

	def uniform_Xgrid(self,n):
	# Create a list of points uniformly spaced along each of dims with n points in each dimension
	# X is the real space point, Xnorm is the cdf, so uniform and normalised to 0<=Xnorm<=1
		l = len(self.dims)
		mesh = np.meshgrid(*[np.linspace(0,1,n) for i in range(l)],indexing='ij')
		Xnorm = np.transpose(np.reshape(mesh,(l,n**l)))
		X = self.Xnorm_to_X(Xnorm)

		return X

	def predict(self,X):
	# Give the GPR prediction for y and its error
		
		Xnorm = self.X_to_Xnorm(X)
		y,std = self.gaussian_process.predict(Xnorm, return_std=True)

		return y,std
