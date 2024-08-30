#!/usr/bin/python3
# Chris Arran, March 2024
# Bayesian Optimisation using Gaussian Process Regression

import numpy as np
from scipy.stats import uniform,loguniform,norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
import matplotlib.pyplot as plt

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
	#	N is the number of points to create

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
	#	Xnorm is the list of points in normalised dimensions

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
	#	X is the list of points in real space

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
	#	acq_fn describes what model to use (UCB for Upper Confidence Bound, EI for expected improvement)
	
		y_acq,sigma_acq = self.gaussian_process.predict(Xnorm_acq, return_std=True)
		
		if (acq_fn=='UCB'):	# Maximise upper confidence bound
			acq = y_acq + explore*sigma_acq
			
		elif (acq_fn=='EI'):	# Maximise expected improvement
			imp = y_acq - np.max(self.y)
			z = imp / sigma_acq - explore
			acq = (imp-explore*sigma_acq)*norm.cdf(z) + sigma_acq*norm.pdf(z)
			acq[sigma_acq==0] = 0
			
#		exclude = np.logical_or(np.isnan(acq), np.isinf(acq))
#		acq[np.logical_or(exclude,acq<0)] = 0
#		return np.log(acq+1e-10)
		return acq


	def next_acquisition(self,Nacq=10,explore=1.0,acq_fn='UCB',debug = False):
	# Finds the next place to acquire with an option for what acquisition function to use
	#	Nacq is the number of places the algorithm starts from to choose the best next place
	# 	explore describes the amount the algorithm should weight exploration over optimisation
	#	acq_fn describes what model to use (UCB for Upper Confidence Bound, EI for expected improvement)
	#	debug prints some extra information on the minimisation
	
		_,Xnorm_start = self.create_Xgrid(Nacq)
		bounds = [(0,1) for d in self.dims] # bounds are for the normalised units
		
		def min_acq_fn(X_acq): # make acquisition function negative to use minimise
			acq = self.acquisition_function(X_acq.reshape(1,-1),explore=explore,acq_fn=acq_fn)
			return -acq
#			exclude = np.logical_or(np.isnan(acq), np.isinf(acq))
#			acq[np.logical_or(exclude,acq<0)] = 0
#			return -np.log(acq+1e-10)
		
		min_val = min_acq_fn(Xnorm_start[0,:])
		min_x = Xnorm_start[0,:]
		for x0 in Xnorm_start:
			res = minimize(min_acq_fn, x0=x0, bounds=bounds, method='L-BFGS-B')
			if debug: print('DEBUG: Starting from ',x0,',ending at ',res.x, ' with ',res.fun[0])
			if (res.fun < min_val):
				if debug: print('DEBUG: Replacing previous minimum: ', res.fun[0],'<',min_val)
				min_val = res.fun[0]
				min_x = res.x

		Xnorm_new = min_x.reshape(1,-1)
		if debug: print('DEBUG: Next acquisition at: ', Xnorm_new,',',acq_fn,'=',-min_val)

		return Xnorm_new

			
	def optimise(self,N,Nacq=10,explore=1,acq_fn='UCB',debug=False):
	# Iteratively improve the GPR using measurements in a place chosen by the acquisition function
	#	N gives the number of iterations to use
	# 	explore describes the amount the algorithm should weight exploration over optimisation
	#	acq_fn describes what model to use (UCB for Upper Confidence Bound, EI for expected improvement)
		
		sz = np.shape(self.X)
		self.X = np.pad(self.X,((0,N),(0,0)))
		self.Xnorm = np.pad(self.Xnorm,((0,N),(0,0)))
		self.y = np.pad(self.y,(0,N))
		self.yerr = np.pad(self.yerr,(0,N))
		
		for n in range(N):
			Xnorm_new = self.next_acquisition(Nacq=Nacq,explore=explore,acq_fn=acq_fn,debug=debug)
			X_new = self.Xnorm_to_X(Xnorm_new)
			y_new,yerr_new = self.fun(X_new[0,:])
			if debug: print('\nDEBUG: Calling function at Xnorm =',Xnorm_new,', X =',X_new,', giving y = ',y_new, ' +- ',yerr_new,'\n')

			self.X[sz[0]+n,:] = X_new
			self.Xnorm[sz[0]+n,:] = Xnorm_new
			self.y[sz[0]+n] = y_new
			self.yerr[sz[0]+n] = yerr_new
			self.gaussian_process.alpha = self.yerr[:sz[0]+n+1]**2
			self.gaussian_process.fit(self.Xnorm[:sz[0]+n+1,:],self.y[:sz[0]+n+1])

	def predict(self,X):
	# Give the GPR prediction for y and its error	
	#	X gives the list of points in real space to predict

		Xnorm = self.X_to_Xnorm(X)
		y,std = self.gaussian_process.predict(Xnorm, return_std=True)

		return y,std

	def uniform_Xgrid(self,n):
	# Create a list of points uniformly spaced along each of dims with n points in each dimension
	# X is the real space point, Xnorm is the cdf, so uniform and normalised to 0<=Xnorm<=1
	#	n gives the resolution of the grid in each dimension

		l = len(self.dims)
		mesh = np.meshgrid(*[np.linspace(0,1,n) for i in range(l)],indexing='ij')
		Xnorm = np.transpose(np.reshape(mesh,(l,n**l)))
		X = self.Xnorm_to_X(Xnorm)

		return X

	def mean_predict(self,ax,n,fun=None):
	# Take mean projections of the model predictions along the given axes with a resolution of n in each dimension
	# Optionally use a different function, such as the acquisition function
	#	ax gives the dimensions to grid against, averaging over all the others
	#	n gives the resolution of the grid in each dimension
	#	fun gives a different function on X that you want to calculate

		Xgrid = self.uniform_Xgrid(n)
		if fun is None:
			ygrid,yerrgrid = self.predict(Xgrid)
		else:
			ygrid = fun(Xgrid)
			yerrgrid = np.zeros_like(ygrid)

		newsz = tuple([n for d in self.dims]) # (n,n,...)
		arrays = tuple([ygrid,yerrgrid]) + tuple(np.transpose(Xgrid)) # (y,yerr,X[:,0],X[:,1],...)
		axmean = tuple( np.setdiff1d(range(len(self.dims)),ax) )
		ms = [np.mean(np.reshape(a,newsz),axis=axmean) for a in arrays]

		return ms # (y,yerr,X[:,0],X[:,1],...)

	def lineout_predict(self,ax,n,centre,fun=None):
	# Take a lineout of the model predictions around a given point along the given axes with a resolution of n in each dimension
	# Optionally use a different function, such as the acquisition function
	#	ax gives the dimension to take a lineout across
	#	n gives the resolution of the grid in each dimension
	#	centre gives the point to take lineouts through
	#	fun gives a different function on X that you want to calculate

		centrenorm = self.X_to_Xnorm(centre.reshape(1,-1))
		Xnormgrid = np.tile( centrenorm, (n,1) )
		Xnormgrid[:,ax] = np.linspace(0,1,n)
		Xgrid = self.Xnorm_to_X(Xnormgrid)
		
		if fun is None:
			ygrid,yerrgrid = self.predict(Xgrid)
		else:
			ygrid = fun(Xgrid)
			yerrgrid = np.zeros_like(ygrid)

		ms = tuple([ygrid,yerrgrid]) + tuple(np.transpose(Xgrid)) # (y,yerr,X[:,0],X[:,1],...)

		return ms # (y,yerr,X[:,0],X[:,1],...)

	def mean_slices_plot(self,n,centrepoint=None,figsize=None,figname=None,fun=None):
	# Plot a grid of projections of the mean model predictions against the data, with a resolution n in each dimension
	# 	n gives the resolution of the grid in each dimension
	# 	centrepoint specifies a point to plot lineouts through
	#	figsize gives the figure size in inches
	# 	figname gives the option of saving the figure with a given name
	#	fun gives a different function on X that you want to calculate

		l = len(self.dims)
		if figsize is None:
			figsize = (4*l,4*l)
		fig = plt.figure(figsize=figsize)

		axes = [(a,b) for a in range(l) for b in range(l)]
		axs = np.empty((l,l),dtype=plt.Axes)
		for (a,b) in axes:
			if (a<=b):
				axs[a,b] = fig.add_subplot(l,l,1+a+b*l) # a is column, b is row
				if self.dims[a]['type'] == 'log-uniform':
					plt.xscale('log')
				if self.dims[b]['type'] == 'log-uniform':
					plt.yscale('log')
				if (a==0 and b>0):
					plt.ylabel(self.dims[b]['name'])
				if (b==l-1):
					plt.xlabel(self.dims[a]['name'])

			if (a==b):		
				ms = self.mean_predict(a,n,fun=fun) # average to a 1D line

				plt.plot(ms[a+2],ms[0],color='tab:orange')
				plt.fill_between(ms[a+2],ms[0]-2*ms[1],ms[0]+2*ms[1],alpha=0.5,color='tab:orange')
				plt.errorbar(self.X[:,a],self.y,self.yerr,marker='o',linestyle='',markersize=4)
				plt.yscale('linear')

				if centrepoint is not None:
					ms = self.lineout_predict(a,n,centrepoint,fun=fun) # lineouts arond the maxpoint
					plt.plot(ms[a+2],ms[0],color='tab:red')
					plt.fill_between(ms[a+2],ms[0]-2*ms[1],ms[0]+2*ms[1],alpha=0.5,color='tab:red')
			elif (a<b):
				ms = self.mean_predict((a,b),n,fun=fun) # average to a 2D slice

				plt.contourf(ms[a+2],ms[b+2],ms[0],levels=int(n/2))
				plt.scatter(self.X[:,a],self.X[:,b],c=self.y,marker='o',edgecolors='black',s=20)

				if centrepoint is not None:
					ms = self.lineout_predict(a,n,centrepoint,fun=fun)
					plt.plot(ms[a+2],ms[b+2],linestyle='--',color='tab:red')
					ms = self.lineout_predict(b,n,centrepoint,fun=fun)
					plt.plot(ms[a+2],ms[b+2],linestyle='--',color='tab:red')

		if figname is not None:
			plt.savefig(figname)
		return axs
			

