#!/usr/bin/python3
# Chris Arran, March 2024
# Bayesian Optimisation using Gaussian Process Regression

import numpy as np
from scipy.stats import uniform,loguniform,norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
				rs = uniform.rvs(loc=0,scale=(d['max']-d['min'])/N,size=N)
				X[:,i] = rs + np.random.choice(np.linspace(d['min'],d['max'],N+1)[0:-1],N,replace=False)
				Xnorm[:,i] = uniform.cdf(X[:,i],loc=d['min'],scale=d['max']-d['min'])
			elif d['type'] == 'log-uniform':
				rs = loguniform.rvs(a=1,b=(d['max']/d['min'])**(1/N),size=N)
				X[:,i] = rs * np.random.choice(np.logspace(np.log10(d['min']),np.log10(d['max']),N+1)[0:-1],N,replace=False)
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

		mu_acq,sigma_acq = self.gaussian_process.predict(Xnorm_acq, return_std=True)

		if (acq_fn=='UCB'):	# Maximise upper confidence bound
			acq = mu_acq + explore*sigma_acq

		elif (acq_fn=='EI'):	# Maximise expected improvement
			imp = mu_acq + explore*sigma_acq - np.max(self.y)
			z = imp / sigma_acq
			acq = imp*norm.cdf(z) + sigma_acq*norm.pdf(z)

			# Handling sigma->0:
			pos = np.logical_and(sigma_acq==0,imp>0)
			neg = np.logical_and(sigma_acq==0,imp<=0)
			acq[pos] = mu_acq[pos] - np.max(self.y)
			acq[neg] = 0

		return acq


	def next_acquisition(self,Nacq=10,explore=1.0,acq_fn='UCB',debug = False):
	# Finds the next place to acquire with an option for what acquisition function to use
	#	Nacq is the number of places the algorithm starts from to choose the best next place
	# 	explore describes the amount the algorithm should weight exploration over optimisation
	#	acq_fn describes what model to use (UCB for Upper Confidence Bound, EI for expected improvement)
	#	debug prints some extra information on the minimisation

		if debug:
			print('DEBUG: Finding next acquisition point using Nacq=%i, explore=%0.2f, acq_fn=%s' % (Nacq, explore, acq_fn))
			X_grid = self.uniform_Xgrid(Nacq)
			Xnorm_grid = self.X_to_Xnorm(X_grid)
			acq_grid = self.acquisition_function(Xnorm_grid,explore=explore,acq_fn=acq_fn)
			imin = np.argmax(acq_grid)
			print('DEBUG: Gridded acquisition function has maximum at ',Xnorm_grid[imin],', (', X_grid[imin],' in real space), ',acq_fn,'=',acq_grid[imin])

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
			if debug: print('DEBUG: Starting from ',x0,',ending at ',res.x, ' with ',res.fun)
			if (res.fun < min_val):
				if debug: print('DEBUG: Replacing previous minimum: ', res.fun,'<',min_val)
				min_val = res.fun
				min_x = res.x

		Xnorm_new = min_x.reshape(1,-1)
		if debug: print('DEBUG: Next acquisition at: ', Xnorm_new,',',acq_fn,'=',-min_val,'\n')

		return Xnorm_new


	def optimise(self,N,Nacq=10,explore=1,acq_fn='UCB',debug=False):
	# Iteratively improve the GPR using measurements in a place chosen by the acquisition function
	#	N gives the number of iterations to use
	# 	explore describes the amount the algorithm should weight exploration over optimisation
	#	acq_fn describes what model to use (UCB for Upper Confidence Bound, EI for expected improvement)

		sz = np.shape(self.X)
		self.X = np.pad(self.X,((0,N),(0,0)),mode='edge')
		self.Xnorm = np.pad(self.Xnorm,((0,N),(0,0)),mode='edge')
		self.y = np.pad(self.y,(0,N),mode='edge')
		self.yerr = np.pad(self.yerr,(0,N),mode='edge')

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

				plt.errorbar(self.X[:,a],self.y,self.yerr,marker='o',linestyle='',markersize=4)
				plt.yscale('linear')

				if centrepoint is None:
					plt.plot(ms[a+2],ms[0],color='tab:orange')
					plt.fill_between(ms[a+2],ms[0]-ms[1],ms[0]+ms[1],alpha=0.5,color='tab:orange')

				if centrepoint is not None:
					ms = self.lineout_predict(a,n,centrepoint,fun=fun) # lineouts arond the maxpoint
					plt.plot(ms[a+2],ms[0],color='tab:red')
					plt.fill_between(ms[a+2],ms[0]-ms[1],ms[0]+ms[1],alpha=0.5,color='tab:red')
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
		
	def mean_plot_3d(self,n=20,figsize=None,figname=None,nlevels=11,clim=[None,None]):
	# Designed for 3 variables, plotting a 3D scatter plot along with the lineouts along each dimension
	
		def log_tick_formatter(val, pos=None):
		    return f"$10^{{{val:g}}}$" 

		fig,axs = plt.subplots(nrows=3,ncols=len(self.dims), sharey=True, figsize=figsize)
		ax = plt.subplot(3,3,(1,6),projection='3d')
		
		# average to 2D slices
		ms0 = self.mean_predict((1,2),n)
		ms1 = self.mean_predict((0,2),n)
		ms2 = self.mean_predict((0,1),n)
		
		X = self.X.copy()
		minmax = np.zeros((len(self.dims),2))
		for a,dim in enumerate(self.dims):
			minmax[a,:] = [dim['min'],dim['max']]
			if dim['type'] == 'log-uniform':
				X[:,a] = np.log10(X[:,a])
				ms0[2+a] = np.log10(ms0[2+a])
				ms1[2+a] = np.log10(ms1[2+a])
				ms2[2+a] = np.log10(ms2[2+a])
				minmax[a,:] = np.log10(minmax[a,:])
			
		# Plot the 3D scatter plot
		p = ax.scatter(X[:,0],X[:,1],X[:,2],c=self.y,edgecolors= "black",vmin=clim[0],vmax=clim[1])
		cbar = plt.colorbar(p,location='left',fraction=0.1,shrink=0.7,anchor=(2,0.5))

		ax.set_xlim(minmax[0,:])
		ax.set_ylim(minmax[1,:])
		ax.set_zlim(minmax[2,:])

		i_max = np.argmax(self.y)
		X_max = X[i_max,:]

		# Lines through the maximum
		o = np.array([1,1])
		ax.plot(X_max[0]*o,X_max[1]*o,minmax[2,:],'r--')
		ax.plot(X_max[0]*o,minmax[1,:],X_max[2]*o,'r--')
		ax.plot(minmax[0,:],X_max[1]*o,X_max[2]*o,'r--')

		# Projections onto the bottom z slice
		ax.plot(X_max[0]*o,minmax[1,:],minmax[2,0]*o,'r:')
		ax.plot(minmax[0,:],X_max[1]*o,minmax[2,0]*o,'r:')

		# Projections onto the back y slice
		ax.plot(X_max[0]*o,minmax[1,1]*o,minmax[2,:],'r:')
		ax.plot(minmax[0,:],minmax[1,1]*o,X_max[2]*o,'r:')
		
		# Projections onto the left x slice
		ax.plot(minmax[0,0]*o,X_max[1]*o,minmax[2,:],'r:')
		ax.plot(minmax[0,0]*o,minmax[1,:],X_max[2]*o,'r:')
		
		# Contours projected onto the bottom, back, and left
		lvls = np.linspace(np.min(self.y),np.max(self.y),nlevels)
		ax.contour(ms0[0],ms0[3],ms0[4], zdir='x', offset=minmax[0,0], levels=lvls, alpha = 0.5)
		ax.contour(ms1[2],ms1[0],ms1[4], zdir='y', offset=minmax[1,1], levels=lvls, alpha = 0.5)
		ax.contour(ms2[2],ms2[3],ms2[0], zdir='z', offset=minmax[2,0], levels=lvls, alpha = 0.5)

		# Labels
		ax.set_xlabel(self.dims[0]['name'], labelpad=-3)
		ax.set_ylabel(self.dims[1]['name'], labelpad=-3)
		ax.set_zlabel(self.dims[2]['name'], labelpad=-3)

		plt.setp( ax.xaxis.get_majorticklabels(), va="bottom" )
		plt.setp( ax.yaxis.get_majorticklabels(), va="bottom" )
		plt.setp( ax.zaxis.get_majorticklabels(), ha="right" )

		ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
		ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

		# Lineout plots below
		# Define a distance measure
		X_max = self.X[i_max,:]
		Xnorm_max = self.Xnorm[i_max,:]
		normcpoint = np.tile(Xnorm_max,(len(self.Xnorm),1))
		dist = np.sqrt(np.sum((self.Xnorm-normcpoint)**2,axis=1))

		for a,dim in enumerate(self.dims):
 			# Wipe first two rows
			axs[0,a].set_axis_off()
			axs[1,a].set_axis_off()
			
			# Scatter with transparency by distance
			subdist = np.sqrt(dist**2 - (self.Xnorm[:,a]-normcpoint[:,a])**2) / np.sqrt(len(self.dims)-1)
			colors = [('tab:blue', 1-d) for d in subdist]
			axs[2,a].scatter(self.X[:,a],self.y, s=20, c=colors, marker='o', linestyle='')
			
			# Errorbars
			markers,caps,bars  = axs[2,a].errorbar(self.X[:,a],self.y,self.yerr, marker='o', linestyle='')
			markers.set_markerfacecolor('none')
			markers.set_markeredgecolor('none')
			for bar in bars:
				bar.colors = colors
			
			# Model plot
			ms = self.lineout_predict(a,n,X_max) # lineouts around the maxpoint
			axs[2,a].plot(ms[a+2],ms[0],color='tab:red')
			axs[2,a].fill_between(ms[a+2],ms[0]-2*ms[1],ms[0]+2*ms[1],alpha=0.5,color='tab:red')
			
			axs[2,a].set_xlim([dim['min'],dim['max']])
			axs[2,a].minorticks_on()
			axs[2,a].grid(True)
			axs[2,a].grid(True, which='minor', linestyle='--')
			axs[2,a].set_xlabel(dim['name'])
			if dim['type'] == 'log-uniform':
				axs[2,a].set_xscale('log')

		plt.tight_layout()
		if figname is not None:
			plt.savefig(figname)
			
		axs[0,0] = ax
		axs[0,1] = cbar
		return axs

