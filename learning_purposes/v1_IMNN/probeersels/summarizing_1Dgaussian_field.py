# coding=utf-8

'''
Reproducing the Summarising Gaussian Signals in the IMNN paper
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

np.random.seed(123) # for reproducibility

def generate_data_test(theta, train=False):
	'''
	Fiducial parameter is NOT passed as a list for testing purposes

	Returns: array of shape (n_s*n_train,input_shape) draws from norm dist
	'''
	if train:
		return np.moveaxis(np.random.normal(0., np.sqrt(theta[0])
			, [1] + input_shape + [len(theta)]), -1, 0)
	else: 
		return np.moveaxis(np.random.normal(0., np.sqrt(theta[0])
			, input_shape + [len(theta)]), -1, 0)

def generate_data(theta, train=False):
	'''
	Fiducial parameter is passed as a list so many simulations can be made
	at once

	Returns: array of shape (n_s*n_train,input_shape) draws from norm dist
	'''
	if train:
		return np.moveaxis(np.random.normal(0., np.sqrt(theta)
			, [1] + input_shape + [len(theta)]), -1, 0)
	else: 
		return np.moveaxis(np.random.normal(0., np.sqrt(theta)
			, input_shape + [len(theta)]), -1, 0)

# 10x20 data points of Gaussian zero-mean noise
input_shape = [10,20,1] # added another axis because we have to
# Fiducial parameter and perturbed values just above and below
theta_fid = 1.
delta_theta = 0.1

''' Generate train data '''

# number of simulations
n_s = 1000
n_train = 1 # splits

t = generate_data([theta_fid for i in range(n_train * n_s)], train = False)

# use less simulations for numerical derivative
derivative_fraction = 0.05
n_p = int(n_s * derivative_fraction)

# set a seed to surpress the sample variance
seed = np.random.randint(1e6)
np.random.seed(seed)
t_m = generate_data([theta_fid - delta_theta for i in range(n_train * n_p)]
					, train = True)
np.random.seed(seed)
t_p = generate_data([theta_fid + delta_theta for i in range(n_train * n_p)]
					, train = True)
np.random.seed()

# denominator of the derivative 
derivative_denominator = 1. / (2. * delta_theta)
der_den = np.array([derivative_denominator]) 

data = {"x_central": t, "x_m": t_m, "x_p":t_p}


''' Generate test data '''
tt = generate_data([theta_fid for i in range(n_s)], train=False)
seed = np.random.randint(1e6)
np.random.seed(seed)
tt_m = generate_data([theta_fid - delta_theta for i in range(n_p)], train=True)
np.random.seed(seed)
tt_p = generate_data([theta_fid + delta_theta for i in range(n_p)], train=True)
np.random.seed()
data["x_central_test"] = tt
data["x_m_test"] = tt_m
data["x_p_test"] = tt_p

def plot_data():
	''' plot the data '''
	fig, ax = plt.subplots(1, 2, figsize = (20, 12))
	plt.subplots_adjust(wspace = 0)
	ax[0].imshow(data["x_central"][np.random.randint(n_train * n_s), :, :, 0])
	ax[0].set_xticks([])
	ax[0].set_yticks([])
	ax[0].set_xlabel('Training image')
	ax[1].imshow(data["x_central_test"][np.random.randint(n_s), :, :, 0])
	ax[1].set_xticks([])
	ax[1].set_yticks([])
	ax[1].set_xlabel('Test image');

	plt.savefig('./Figures/1d_gaussian_field/data_visualization_dense.png')
	plt.close()

plot_data()

def plot_derivatives():
	'''
	plot the upper and lower derivatives to check the sample variance is being 
	surpressed. This needs to be done or the network learns very slowly
	'''
	fig, ax = plt.subplots(3, 2, figsize = (15, 10))
	plt.subplots_adjust(wspace = 0, hspace = 0.1)
	# select a random training example
	training_index = np.random.randint(n_train * n_p)
	# for color scale purposes
	training_min = min(np.min(data["x_m"][training_index, 0, :, :, 0]), np.min(data["x_p"][training_index, 0, :, :, 0]))
	training_max = min(np.max(data["x_m"][training_index, 0, :, :, 0]), np.max(data["x_p"][training_index, 0, :, :, 0]))
	ax[0, 0].imshow(data["x_m"][training_index, 0, :, :, 0], vmin = training_min, vmax = training_max)
	ax[0, 0].set_xticks([])
	ax[0, 0].set_yticks([])
	ax[0, 0].set_xlabel('Upper training image')
	ax[1, 0].imshow(data["x_p"][training_index, 0, :, :, 0])
	ax[1, 0].set_xticks([])
	ax[1, 0].set_yticks([])
	ax[1, 0].set_xlabel('Lower training image')
	ax[2, 0].imshow(data["x_m"][training_index, 0, :, :, 0] - data["x_p"][training_index, 0, :, :, 0], vmin = training_min, vmax = training_max)
	ax[2, 0].set_xticks([])
	ax[2, 0].set_yticks([])
	ax[2, 0].set_xlabel('Difference between upper and lower training images');
	test_index = np.random.randint(n_p)
	test_min = min(np.min(data["x_m_test"][test_index, 0, :, :, 0]), np.min(data["x_p_test"][test_index, 0, :, :, 0]))
	test_max = min(np.max(data["x_m_test"][test_index, 0, :, :, 0]), np.max(data["x_p_test"][test_index, 0, :, :, 0]))
	ax[0, 1].imshow(data["x_p_test"][test_index, 0, :, :, 0], vmin = test_min, vmax = test_max)
	ax[0, 1].set_xticks([])
	ax[0, 1].set_yticks([])
	ax[0, 1].set_xlabel('Upper test image');
	ax[1, 1].imshow(data["x_m_test"][test_index, 0, :, :, 0], vmin = test_min, vmax = test_max)
	ax[1, 1].set_xticks([])
	ax[1, 1].set_yticks([])
	ax[1, 1].set_xlabel('Lower test image');
	ax[2, 1].imshow(data["x_m_test"][test_index, 0, :, :, 0] - data["x_p_test"][test_index, 0, :, :, 0], vmin = test_min, vmax = test_max)
	ax[2, 1].set_xticks([])
	ax[2, 1].set_yticks([])
	ax[2, 1].set_xlabel('Difference between upper and lower test images');

	plt.savefig('./Figures/1d_gaussian_field/derivatives_visualization_dense.png')
	plt.close()

plot_derivatives()

parameters = {
    'verbose': True,
    'number of simulations': n_s,
    'fiducial θ': np.array([theta_fid]),
    'derivative denominator': der_den,
    'differentiation fraction': derivative_fraction,
    'number of summaries': 1,
    'calculate MLE': True,
    'prebuild': True,
    'input shape': input_shape,
    'preload data': data,
    'save file': "data/saved_model",
    'wv': 0.,
    'bb': 0.1,
    'activation': tf.nn.leaky_relu,
    'α': 0.01,
    # works
    # 'hidden layers': [[10, [5, 5], [2, 2], 'SAME'], [6, [3, 3], [1, 1], 'SAME'], 100, 100],
    # doesnt work
    # 'hidden layers': [256,256,256,256,256,256,256]
    # ..
	'hidden layers': [[5, [3, 3], [1, 1], 'SAME'], [5, [3, 3], [1, 1], 'SAME'], 128, 128],


}

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
	ax[3].plot(epochs, np.array(n.history["dμdθ"]).reshape((np.prod(np.array(n.history["dμdθ"]).shape))))
	ax[3].plot(epochs, np.array(n.history["test dμdθ"]).reshape((np.prod(np.array(n.history["test dμdθ"]).shape))))
	ax[3].set_ylabel(r'$\partial\mu/\partial\theta$')
	ax[3].set_xlabel('Number of epochs')
	ax[3].set_xlim([0, len(epochs)])
	ax[4].plot(epochs, np.array(n.history["μ"]).reshape((np.prod(np.array(n.history["μ"]).shape))))
	ax[4].plot(epochs, np.array(n.history["test μ"]).reshape((np.prod(np.array(n.history["test μ"]).shape))))
	ax[4].set_ylabel('μ')
	ax[4].set_xlabel('Number of epochs')
	ax[4].set_xlim([0, len(epochs)])
	plt.savefig('./Figures/1d_gaussian_field/variables_vs_epochs_dense.png')
	plt.close()
	# plt.show()

plot_variables()
# ===============================================================
# Approximate Bayesian computation with the calculated summary:

# First calculate the real data
real_data = generate_data([1.], train = False)

def ABC():


	def show_real_data():
		fig, ax = plt.subplots(1, 1, figsize = (10, 12))
		ax.imshow(real_data[0, :, :, 0])
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlabel('Simulated real image');
		plt.savefig('./Figures/1d_gaussian_field/real_data_dense.png')
		plt.close()

	show_real_data()

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
	ϵ = 20000
	accept_indices = np.argwhere(ro < ϵ)[:, 0]
	reject_indices = np.argwhere(ro >= ϵ)[:, 0]

	# plot output samples and histogram of the accepted samples
	# which should peak around theta=1
	def plot_samples():
		fig, ax = plt.subplots(2, 1, sharex = True, figsize = (10, 10))
		plt.subplots_adjust(hspace = 0)
		ax[0].scatter(θ[accept_indices] , s[accept_indices, 0], s = 1)
		ax[0].scatter(θ[reject_indices], s[reject_indices, 0], s = 1, alpha = 0.1)
		ax[0].plot([0, 10], [summary[0], summary[0]], color = 'black', linestyle = 'dashed')
		ax[0].set_ylabel('Network output', labelpad = 0)
		ax[0].set_xlim([0, 10])
		ax[1].hist(θ[accept_indices], np.linspace(0, 10, 100), histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
		ax[1].set_xlabel('$\\theta$')
		ax[1].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
		ax[1].set_yticks([])
		plt.savefig('./Figures/1d_gaussian_field/ABC_dense.png')
		plt.close()

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
		ax[0].scatter(θ_ , s_, s = 1)
		ax[0].plot([0, 10], [summary[0], summary[0]], color = 'black', linestyle = 'dashed')
		ax[0].set_ylabel('Network output', labelpad = 0)
		ax[0].set_xlim([0, 10])
		ax[0].set_ylim([np.min(s_), np.max(s_)])
		ax[1].hist(θ_, np.linspace(0, 10, 100), histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
		ax[1].set_xlabel('θ')
		ax[1].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
		ax[1].set_yticks([]);
		plt.savefig('./Figures/1d_gaussian_field/PMC_ABC_dense.png')
		plt.close()
		# plt.show()
	# plot()

	return theta_

theta_ =  PMC_ABC()


# =============================================
def first_order_Gaussian_MLE():
	# Can also calculate the first-order Gaussian approximation of the posterior on
	# the parameter and find a maximum likelihood estimate.

	asymptotic_likelihood = n.asymptotic_likelihood(real_data = real_data, prior = np.linspace(0, 10, 1000).reshape((1, 1, 1000)))
	MLE = n.θ_MLE(real_data = real_data)

	def plot():
		fig, ax = plt.subplots(1, 1, figsize = (10, 6))
		ax.plot(np.linspace(0, 10, 1000), asymptotic_likelihood[0, 0], linewidth = 1.5)
		ax.axvline(x = MLE[0, 0], ymin = 0., ymax = 1., linestyle = 'dashed', color = 'black')
		ax.set_xlabel("θ")
		ax.set_xlim([0, 10])
		ax.set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
		ax.set_yticks([])
		plt.savefig('./Figures/1d_gaussian_field/Gaussian_MLE_dense.png')
		plt.close()
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
ax.axvline(x = 1., ymin = 0., ymax = 1., linestyle = 'dashed'
	, color = 'red', label = "Actual Fiducial Parameter")
ax.legend(frameon = False)
ax.set_xlim([0, 10])
ax.set_xlabel('θ')
ax.set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
ax.set_yticks([])
plt.savefig('./Figures/1d_gaussian_field/likelihoods_dense.png')
# plt.show()
plt.close()