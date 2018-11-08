# coding=utf-8

'''

Finding the parameters of a 2D multivariate normal distribution.

Trying to maximize the Fisher information by finding a non-linear summary
of the data

Recommended to run in a seperate environment where keras / tensorflow is 
installed. (see miniconda)

Code adapted from
arXiv:1802.03537.

'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# change to the path where the IMNN git clone is located
sys.path.insert(-1,'../../information_maximiser')
import IMNN # make sure the path to the IMNN is given
import tqdm

tf.reset_default_graph() # start fresh

def generate_data_ABC(theta, train=False):
	'''
	Fiducial parameter is passed as a list so many simulations can be made
	at once

	Returns: array of shape (n_s*n_train,input_shape) draws from norm dist
	'''
	return 'TODO'
	if train:
		return np.moveaxis(np.random.normal(0., np.sqrt(theta)
			, [1] + input_shape + [len(theta)]), -1, 0)
	else: 
		return np.moveaxis(np.random.normal(0., np.sqrt(theta)
			, input_shape + [len(theta)]), -1, 0)

def generate_data(theta1, theta2, n_samples, train=False):
	'''
	Fiducial parameter is passed as a list so many simulations can be made
	at once

	Returns: array of shape (n_samples,input_shape,N) draws 
	from multivariate normal dist, where N is the dimension of variables
	(i.e., N = len(mean) )
	'''

	mean = [0,0] 
	cov = [[theta1,0],  # x_0 has variance 1, 
	       [0,theta2]] # x_1 has variance 2, no covariance  

	# e.g: input_shape = [10,10,1], n_samples = 1000, len(mean) = 2
	# multivariate_normal returns shape (10,10,1,1000,2)
	# therefore we move the second to last axis to the first
	# so we get shape (1000,10,10,1,2)

	# but also, since the multivariate Gaussian returns ((shape,2)) always
	# we take all but the last argument of the input shape,
	# which is then required to be 2
	assert input_shape[-1] == 2, "Multivariate Gaussian generates (,2) shaped data"
	if train:
		return np.moveaxis(
					np.random.multivariate_normal(mean, cov,
						size = [1] + input_shape[:-1] + [n_samples]
					)
				, -2 , 0)
	else:
		return np.moveaxis(
					np.random.multivariate_normal(mean, cov,
						size = input_shape[:-1] + [n_samples]
					)
				, -2 , 0)


# Multivariate Gaussians are specified by their mean and covariance matrix
# We will just consider N=2 dimensions

# each entry is an N dimensional value
# Covariance indicates the level to which two variables vary together
# The covariance matrix element Cij is the covariance
# of x_i and x_j, The element c_ii is the variance of x_i 

# generate 10x10 data points, or just 100 data points?
input_shape = [100,1,2] # This is 100 2D data points with a dummy axis

# Fiducial parameter and perturbed values just above and below
theta1_fid = 1. # variance of 1st dimension (x)
theta2_fid = 2. # variance of 2nd dimension (y)
delta_theta = 0.1

''' Generate train data '''
n_s = 1000 # number of simulations
n_train = 1 # splits

t = generate_data(theta1_fid,theta2_fid,n_s*n_train, train = False)

# use less simulations for numerical derivative
derivative_fraction = 0.05
n_p = int(n_s * derivative_fraction)

# set a seed to surpress the sample variance
seed = np.random.randint(1e6)
np.random.seed(seed)

# t_m must be an array of shape (num_params,input_shape)
# thus in this case, we need to build an array with in the first position
# t_m_-theta1 and in the second position t_m-theta2
t_m1 = generate_data(theta1_fid - delta_theta, theta2_fid
						,n_train*n_p, train = True) 
t_m2 = generate_data(theta1_fid, theta2_fid - delta_theta
						,n_train*n_p, train= True)
t_m = np.concatenate( (t_m1, t_m2), axis=1) # I think this is the correct way
np.random.seed(seed)
t_p1 = generate_data(theta1_fid + delta_theta, theta2_fid
						,n_train*n_p, train = True) 
t_p2 = generate_data(theta1_fid, theta2_fid + delta_theta
						,n_train*n_p, train= True)
t_p = np.concatenate( (t_p1, t_p2), axis=1) # I think this is the correct way

np.random.seed()

# denominator of the derivative 
derivative_denominator = 1. / (2. * delta_theta)
# needs to be stored in an array of shape [number of parameters]
der_den = np.array([derivative_denominator, derivative_denominator]) 

data = {"x_central": t, "x_m": t_m, "x_p":t_p}

''' Generate test data '''
tt = generate_data(theta1_fid,theta2_fid, n_s, train=False)
seed = np.random.randint(1e6)
np.random.seed(seed)
tt_m1 = generate_data(theta1_fid - delta_theta, theta2_fid
	, n_p, train=True)
tt_m2 = generate_data(theta1_fid, theta2_fid - delta_theta
	, n_p, train=True)
tt_m = np.concatenate( (tt_m1, tt_m2), axis=1)
np.random.seed(seed)
tt_p1 = generate_data(theta1_fid + delta_theta, theta2_fid
	, n_p, train=True)
tt_p2 = generate_data(theta1_fid, theta2_fid + delta_theta
	, n_p, train=True)
tt_p = np.concatenate( (tt_p1, tt_p2), axis=1)
np.random.seed()
data["x_central_test"] = tt
data["x_m_test"] = tt_m
data["x_p_test"] = tt_p

def plot_data():
	""" 
	plot the data, since it is a 2D multivariate gaussian we plot the distribution
	of the datapoints in 2D space
	"""

	# fig, ax = plt.subplots(1, 2, figsize = (10, 6))
	# # plot one random row from 1000 rows of the simulated data images 
	# ax[0].imshow(data['x_central'][np.random.randint(n_train * n_s),:,:,0]
	# 	, label = "training data")
	# ax[0].set_xticks([])
	# ax[0].set_yticks([])
	# ax[0].set_xlabel('Training image')
	# ax[1].imshow(data['x_central_test'][np.random.randint(n_s),:,:,0]
	# 	, label = "test data")
	# ax[1].set_xticks([])
	# ax[1].set_yticks([])
	# ax[1].set_xlabel("Test image")
	# plt.savefig('./multivariate_gaussian/data_visualization')
	# plt.show()

	fig, ax = plt.subplots(1, 2, figsize = (10, 6))
	# plot one random row from 1000 rows of the simulated data images 
	# x, y = data['x_central'][np.random.randint(n_train * n_s)].T
													# [:,0] removes dummy axis
	ax[0].plot(*data['x_central'][np.random.randint(n_train * n_s)].T[:,0],'x')
	ax[0].axis('equal')
	ax[0].set_title('Training data')
	x, y = data['x_central_test'][np.random.randint(n_s)].T[:,0]
	ax[1].plot(x,y,'x')
	ax[1].axis('equal')
	ax[1].set_title('Test data')
	plt.savefig('./Figures/multivariate_gaussian/data_visualization.png')
	plt.show()
	plt.close()

# plot_data()

def plot_derivatives():
	'''
	plot the upper and lower derivatives to check the sample variance is being 
	surpressed. This needs to be done or the network learns very slowly
	'''

	fig, ax = plt.subplots(3, 2, figsize = (15, 10))
	# plt.subplots_adjust(wspace = 0, hspace = 0.1)
	training_index = np.random.randint(n_train * n_p)
	
	x, y = data['x_p'][training_index].T[:,0]
	
	# dont have to split, but for labelling purposes we split the first 
	x_theta1, x_theta2 = x.T
	y_theta1, y_theta2 = y.T

	ax[0, 0].plot(x_theta1,y_theta1,'x',label='$θ_1$')
	ax[0, 0].plot(x_theta2,y_theta2,'x',label='$θ_2$')
	ax[0, 0].legend(frameon=True)
	ax[0, 0].set_title('Upper training image')
	ax[0, 0].set_xlim(0-3*theta1_fid,0+3*theta1_fid)
	ax[0, 0].set_ylim(0-3*theta2_fid,0+3*theta2_fid)

	ax[1, 0].plot(*data['x_m'][training_index].T[:,0],'x')
	ax[1, 0].set_title('Lower training image')
	ax[1, 0].set_xlim(0-3*theta1_fid,0+3*theta1_fid)
	ax[1, 0].set_ylim(0-3*theta2_fid,0+3*theta2_fid)
	
	xm, ym = data["x_m"][training_index].T[:,0]
	xp, yp = data["x_p"][training_index].T[:,0]
	ax[2, 0].plot(xp-xm,yp-ym,'x')
	ax[2, 0].set_title('Difference between upper and lower training images');
	ax[2, 0].axhline(xmin = 0., xmax = 1., y = 0.
		, linestyle = 'dashed', color = 'black')
	test_index = np.random.randint(n_p)
	ax[0, 1].plot(*data['x_p_test'][test_index].T[:,0],'x')
	ax[0, 1].set_title('Upper test image')
	ax[1, 1].plot(*data['x_m_test'][training_index].T[:,0],'x')
	ax[1, 1].set_title('Lower test image')
	
	xm, ym = data["x_m_test"][test_index].T[:,0]
	xp, yp = data["x_p_test"][test_index].T[:,0]
	ax[2, 1].plot(xp-xm,yp-ym,'x')
	ax[2, 1].axhline(xmin = 0., xmax = 1., y = 0.
		, linestyle = 'dashed', color = 'black')
	ax[2, 1].set_title('Difference between upper and lower test images');	
	plt.savefig('./Figures/multivariate_gaussian/derivatives.png')
	plt.show()
	plt.close()

# plot_derivatives()

''' Initialize the Neural Net '''
parameters = {
    'verbose': True,
    'number of simulations': n_s,
    'fiducial θ': np.array([theta1_fid,theta2_fid]), # I think it works like this
    'derivative denominator': der_den,
    'differentiation fraction': derivative_fraction,
    'number of summaries': 1,
    'calculate MLE': True,
    'prebuild': True,
    'input shape': input_shape,
    'preload data': data,
    'save file': "data/model",
    'wv': 0.,
    'bb': 0.1,
    'activation': tf.nn.leaky_relu,
    'α': 0.01,
    'hidden layers': [256, 256, 128] # Dense layers only doesnt work well for 2D data
    # Convolutional layers:
    # 'hidden layers': [[10, [5, 5], [2, 2], 'SAME'], [6, [3, 3], [1, 1], 'SAME'], 100, 100],

}

# np.randfsdf()
data = {"x_central": t.flatten(), "x_m": t_m.flatten(), "x_p":t_p.flatten()}
data["x_central_test"] = tt.flatten()
data["x_m_test"] = tt_m.flatten()
data["x_p_test"] = tt_p.flatten()

# Initialize the IMNN
n = IMNN.IMNN(parameters=parameters)
eta = 1e-3
# Initialize input tensors, build network and define optimalization scheme
n.setup(η = eta)

# can change the optimization scheme (although adam was found to be unstable)
# n.backpropagate = tf.train.AdamOptimizer(eta).minimize(n.Λ) apparently doesnt work

num_epochs = 800
keep_rate = 0.8

n.train(num_epochs = num_epochs, n_train = n_train, keep_rate = keep_rate
	, data = data, history = True)

def plot_variables():
	fig, ax = plt.subplots(5, 1, sharex = True, figsize = (8, 14))
	plt.subplots_adjust(hspace = 0)
	end = len(n.history["det(F)"])
	epochs = np.arange(end)
	a, = ax[0].plot(epochs, n.history["det(F)"], label = 'Training data')
	b, = ax[0].plot(epochs, n.history["det(test F)"], label = 'Test data')
	# ax[0].axhline(y=5,ls='--',color='k')
	ax[0].legend(frameon = False)
	ax[0].set_ylabel(r'$|{\bf F}_{\alpha\beta}|$')
	ax[1].plot(epochs, n.history["Λ"])
	ax[1].plot(epochs, n.history["test Λ"])
	ax[1].set_xlabel('Number of epochs')
	ax[1].set_ylabel(r'$\Lambda$')
	ax[1].set_xlim([0, len(epochs)]);
	ax[2].plot(epochs, n.history["det(C)"])
	ax[2].plot(epochs, n.history["det(test C)"])
	ax[2].set_xlabel('Number of epochs')
	ax[2].set_ylabel(r'$|{\bf C}|$')
	ax[2].set_xlim([0, len(epochs)]);
	# Derivative wrt to theta1                   theta1 is column 0
	ax[3].plot(epochs, np.array(n.history["dμdθ"])[:,0].reshape((np.prod(np.array(n.history["dμdθ"]).shape)//2))
		, color = 'C0', label='theta1')
	# Derivative wrt to theta2                   theta2 is column 1
	ax[3].plot(epochs, np.array(n.history["dμdθ"])[:,1].reshape((np.prod(np.array(n.history["dμdθ"]).shape)//2))
		, color = 'C0', ls='dashed', label='theta2')

	ax[3].plot(epochs, np.array(n.history["test dμdθ"])[:,0].reshape((np.prod(np.array(n.history["test dμdθ"]).shape)//2))
		, color = 'C1', label='theta1')
	ax[3].plot(epochs, np.array(n.history["test dμdθ"])[:,1].reshape((np.prod(np.array(n.history["test dμdθ"]).shape)//2))
		, color = 'C1', ls='dashed', label='theta2')
	ax[3].legend(frameon=False)

	ax[3].set_ylabel(r'$\partial\mu/\partial\theta$')
	ax[3].set_xlabel('Number of epochs')
	ax[3].set_xlim([0, len(epochs)])
	ax[4].plot(epochs, np.array(n.history["μ"]).reshape((np.prod(np.array(n.history["μ"]).shape))))
	ax[4].plot(epochs, np.array(n.history["test μ"]).reshape((np.prod(np.array(n.history["test μ"]).shape))))
	ax[4].set_ylabel('μ')
	ax[4].set_xlabel('Number of epochs')
	ax[4].set_xlim([0, len(epochs)])
	plt.savefig('./Figures/multivariate_gaussian/variables_vs_epochs_flatten.png')
	plt.show()

plot_variables()
# ===============================================================
# Approximate Bayesian computation with the calculated summary:


# First calculate the real data
real_data = generate_data(theta1_fid, theta2_fid, 1, train = False)

def ABC():


	def show_real_data():
		fig, ax = plt.subplots(1, 1, figsize = (10, 6))
		ax.plot(real_data[0], label = "real data")
		ax.legend(frameon = False)
		ax.set_xlim([0, 9])
		ax.set_xticks([])
		ax.set_ylabel("Data amplitude")
		plt.show()
	# show_real_data()

	# Perform ABC by drawing 100,000 random samples from the prior.

	# Define the upper and lower bounds of a uniform prior to be 0 and 10
	# Only a uniform prior is implemented at the moment.
	# From the samples we create simulations at each parameter value and feed each
	# simulation through the network to get summaries. The summaries are compared
	# to the summary of the real data to find the distances which can be used to
	# accept or reject points. Because the simulations are created within the ABC
	# function, the generation function must be passed. This is why we have put the
	# data_gen func in its form, which takes a list of param valeus and returns
	# a simulation at each parameter.

	# If the data is not preloaded as a tensorflow constant then the data can be
	# passed to the function as

	# sampled parameter values, summary of real data, summaries of generated data
	# distances of generated data to real data, Fisher info of real data
	theta, summary, s, ro, F = n.ABC(real_data = real_data, prior = [0, 10]
		, draws = 100000, generate_simulation = generate_data
		, at_once = True, data = data)
	#at_once = False will create only one simulation at a time


	# Draws are accepted if the distance between the simulation summary and the 
	# simulation of real data are close (i.e., smaller than some value epsilon)
	epsilon = 1
	accept_indices = np.argwhere(ro < epsilon)[:, 0]
	reject_indices = np.argwhere(ro >= epsilon)[:, 0]

	# plot output samples and histogram of the accepted samples
	# which should peak around theta=1
	def plot_samples():
		fig, ax = plt.subplots(2, 1, sharex = True, figsize = (10, 10))
		plt.subplots_adjust(hspace = 0)
		ax[0].scatter(theta[accept_indices] , s[accept_indices, 0], s = 1)
		ax[0].scatter(theta[reject_indices], s[reject_indices, 0], s = 1, alpha = 0.1)
		ax[0].plot([0, 10], [summary[0], summary[0]], color = 'black', linestyle = 'dashed')
		ax[0].set_ylabel('Network output', labelpad = 0)
		ax[0].set_xlim([0, 10])
		ax[1].hist(theta[accept_indices], np.linspace(0, 10, 100)
			, histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
		ax[1].set_xlabel('$\\theta$')
		ax[1].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
		ax[1].set_yticks([])
		# plt.savefig('./Figures/approximate_bayesian_computation.png')
		plt.show()

	# plot_samples()
	# There can be a lot of theta draws which are unconstrained by the network
	# because no similar structures were seen in the data, which is indicative of
	# using too small of a small training set

	return theta, accept_indices

theta, accept_indices = ABC()

def PMC_ABC():
	# A way of reducing the number of draws by first sampling from a prior,
	# accepting the closest 75% of the samples and weighting all the rest of
	# the samples to create a new proposal distribution
	# The furthest 25% of the original samples are redrawn from the new proposal
	# distribution. The furthest 25% of the simulation summaries are continually
	# rejected and the proposal distribution updated until the number of draws 
	# needed (to?) accept all the 25% of the samples is much greater than
	# this number of samples. This ratio is called the criterion.
	# The inputs work in a very similar way to the ABC function above. If we want
	# 1000 samples from the approximate distribution and the end of the PMC we 
	# need to set num_keep = 1000. The initial random draw initialised with
	# num_draws, the larger this is the better proposal distr will be on 1st iter

	# W = weighting of samples, total_draws = total num draws so far
	theta_, summary_, ro_, s_, W, total_draws, F = n.PMC(real_data = real_data
		, prior = [0, 10], num_draws = 1000, num_keep = 1000
		, generate_simulation = generate_data, criterion = 0.1, at_once = True
		, samples = None, data = data)

	def plot():
		fig, ax = plt.subplots(2, 1, sharex = True, figsize = (10, 10))
		plt.subplots_adjust(hspace = 0)
		ax[0].scatter(theta_ , s_, s = 1)
		ax[0].plot([0, 10], [summary_[0], summary_[0]], color = 'black', linestyle = 'dashed')
		ax[0].set_ylabel('Network output', labelpad = 0)
		ax[0].set_xlim([0, 10])
		ax[0].set_ylim([np.min(s_), np.max(s_)])
		ax[1].hist(theta_, np.linspace(0, 10, 100), histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
		ax[1].set_xlabel('θ')
		ax[1].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
		ax[1].set_yticks([]);
		# plt.savefig('./Figures/PMC_ABC.png')
		plt.show()
	# plot()

	return theta_

theta_ =  PMC_ABC()


# =============================================
def first_order_Gaussian_MLE():
	# Can also calculate the first-order Gaussian approximation of the posterior on
	# the parameter and find a maximum likelihood estimate.

	asymptotic_likelihood = n.asymptotic_likelihood(real_data = real_data
		, prior = np.linspace(0, 10, 1000).reshape((1, 1, 1000)), data = data)

	MLE = n.θ_MLE(real_data = real_data, data = data)

	def plot():
		fig, ax = plt.subplots(1, 1, figsize = (10, 6))
		ax.plot(np.linspace(0, 10, 1000), asymptotic_likelihood[0, 0], linewidth = 1.5)
		ax.axvline(x = MLE[0, 0], ymin = 0., ymax = 1., linestyle = 'dashed', color = 'black')
		ax.set_xlabel("θ")
		ax.set_xlim([0, 10])
		ax.set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
		ax.set_yticks([]);
		plt.show()

	return MLE, asymptotic_likelihood

MLE, asymptotic_likelihood = first_order_Gaussian_MLE()
# Plot all kinds of different likelihoods
θ_grid = np.linspace(0.001, 10, 1000)
analytic_posterior = np.exp(-0.5 * np.sum(real_data**2.) / θ_grid
	) / np.sqrt(2. * np.pi * θ_grid)**10.
analytic_posterior = analytic_posterior / np.sum(analytic_posterior 
	* (θ_grid[1] - θ_grid[0]))

fig, ax = plt.subplots(1, 1, figsize = (10, 6))
ax.plot(θ_grid, analytic_posterior, linewidth = 1.5, color = 'C1'
	, label = "Analytic posterior")
ax.hist(theta_, np.linspace(0, 10, 100), histtype = u'step', density = True
	, linewidth = 1.5, color = '#9467bd', label = "PMC posterior");
ax.hist(theta[accept_indices], np.linspace(0, 10, 100), histtype = u'step'
	, density = True, linewidth = 1.5, color = 'C2', label = "ABC posterior")
ax.plot(np.linspace(0, 10, 1000), asymptotic_likelihood[0, 0], color = 'C0'
	, linewidth = 1.5, label = "Asymptotic Gaussian likelihood")
ax.axvline(x = MLE[0, 0], ymin = 0., ymax = 1., linestyle = 'dashed'
	, color = 'black', label = "Maximum likelihood estimate")
ax.legend(frameon = False)
ax.set_xlim([0, 10])
ax.set_xlabel('θ')
ax.set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
ax.set_yticks([])
# plt.savefig('./Figures/likelihoods.png')
plt.show()


