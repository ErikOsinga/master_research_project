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

def generate_data(theta1, theta2, n_samples = 1, train=False):
	'''

	Returns: array of shape (n_samples,input_shape) draws 
	from normal dist with mean=theta1 and std = theta2
	'''

	mean = theta1
	std = np.sqrt(theta2)
	
	# e.g: input_shape = [10,10,1], n_samples = 1000, len(mean) = 2
	# multivariate_normal returns shape (10,10,1,1000,2)
	# therefore we move the second to last axis to the first
	# so we get shape (1000,10,10,1,2)

	# but also, since the multivariate Gaussian returns ((shape,2)) always
	# we take all but the last argument of the input shape,
	# which is then required to be 2
	if train:
		return np.moveaxis(
					np.random.normal(mean,std, 
							size = [1] + input_shape + [n_samples]
					)
				, -1, 0)
	else:
		return np.moveaxis(
					np.random.normal(mean,std, 
							size = input_shape + [n_samples]
					)
				, -1, 0)


# 10 data points of Gaussian zero-mean noise
input_shape = [100] 
# Fiducial parameter and perturbed values just above and below
theta_fid = 0. # mean
theta2_fid = 1. # variance
delta_theta = 0.1

''' Generate train data '''

# number of simulations
n_s = 1000
n_train = 1 # splits, for if it doesnt fit into memory

t = generate_data(theta_fid,theta2_fid,(n_train*n_s), train = False)

# use less simulations for numerical derivative
derivative_fraction = 0.05
n_p = int(n_s * derivative_fraction)

# set a seed to surpress the sample variance
seed = np.random.randint(1e6)
np.random.seed(seed)
# Perturb theta1 minus delta theta
t_m1 = generate_data(theta_fid - delta_theta, theta2_fid,(n_train*n_p), train = True) 
# Perturb theta2 minus delta theta
t_m2 = generate_data(theta_fid, theta2_fid - delta_theta,(n_train*n_p), train= True)
# Concatenate these into a vector
t_m = np.concatenate( (t_m1, t_m2), axis=1) # I think this is the correct way

# set a seed to surpress the sample variance
np.random.seed(seed)
# Perturb theta1 plus delta theta
t_p1 = generate_data(theta_fid + delta_theta,theta2_fid,(n_train*n_p), train = True) 
# Perturb theta2 plus delta theta
t_p2 = generate_data(theta_fid, theta2_fid + delta_theta,(n_train*n_p), train= True)
t_p = np.concatenate( (t_p1, t_p2), axis=1) # I think this is the correct way

np.random.seed()

# denominator of the derivative 
derivative_denominator = 1. / (2. * delta_theta)
# needs to be stored in an array of shape [number of parameters]
der_den = np.array([derivative_denominator, derivative_denominator]) 


data = {"x_central": t, "x_m": t_m, "x_p":t_p}

# Repeat the same story to generate training data
tt = generate_data(theta_fid,theta2_fid, n_s, train=False)
seed = np.random.randint(1e6)
np.random.seed(seed)
tt_m1 = generate_data(theta_fid - delta_theta,theta2_fid,n_p, train=True)
tt_m2 = generate_data(theta_fid, theta2_fid - delta_theta,n_p, train=True)
tt_m = np.concatenate( (tt_m1, tt_m2), axis=1)

np.random.seed(seed)
tt_p1 = generate_data(theta_fid + delta_theta, theta2_fid,n_p, train=True)
tt_p2 = generate_data(theta_fid, theta2_fid + delta_theta, n_p, train=True)
tt_p = np.concatenate( (tt_p1, tt_p2), axis=1)
np.random.seed()
data["x_central_test"] = tt
data["x_m_test"] = tt_m
data["x_p_test"] = tt_p

def plot_data():
	''' plot the data '''
	fig, ax = plt.subplots(1, 1, figsize = (10, 6))
	# plot one random row from the simulated data 
	ax.plot(data['x_central'][np.random.randint(n_train * n_s)], label = "training data")
	ax.plot(data['x_central_test'][np.random.randint(n_s)], label = "test data")
	ax.legend(frameon = False)
	# ax.set_xlim([0, 9])
	ax.set_xticks([])
	ax.set_ylabel("Data amplitude");
	plt.savefig('./Figures/1d_gaussian2params/data_visualization_2summaries.png')
	plt.close()

plot_data()

def plot_derivatives():
	'''
	plot the upper and lower derivatives to check the sample variance is being 
	surpressed. This needs to be done or the network learns very slowly
	'''

	fig, ax = plt.subplots(2, 2, figsize = (15, 10))
	# plt.subplots_adjust(wspace = 0, hspace = 0.1)
	training_index = np.random.randint(n_train * n_p)
	
	xp_theta1, xp_theta2 = data['x_p'][training_index]

	# Theta 1 upper simulation
	ax[0, 0].plot(data['x_p'][training_index, 0], label = "upper training data"
		, color = 'C0', linestyle='dashed')
	# Theta 2 upper simulation
	ax[0, 1].plot(data['x_p'][training_index, 1], label = "upper training data"
		, color = 'C0', linestyle='dashed')

	# Theta 1 lower simulation
	ax[0, 0].plot(data['x_m'][training_index, 0], label = "lower training data"
				, color = 'C0')
	# Theta 2 lower simulation
	ax[0, 1].plot(data['x_m'][training_index, 1], label = "lower training data"
				, color = 'C0')

	test_index = np.random.randint(n_p)
	# Theta1  upper and lower
	ax[0, 0].plot(data['x_m_test'][test_index, 0], label = "lower test data"
				, color = 'C1')
	ax[0, 0].plot(data['x_p_test'][test_index, 0], label = "upper test data"
				, color = 'C1', linestyle='dashed')

	# Theta2  upper and lower
	ax[0, 1].plot(data['x_m_test'][test_index, 1], label = "lower test data"
				, color = 'C1')
	ax[0, 1].plot(data['x_p_test'][test_index, 1], label = "upper test data"
				, color = 'C1', linestyle='dashed')


	ax[0, 0].legend(frameon = False)
	ax[0, 1].legend(frameon = False)
	ax[0, 0].set_xticks([])
	ax[0, 1].set_xticks([])
	ax[0, 0].set_ylabel("Data amplitude")
	ax[0, 1].set_ylabel("Data amplitude")
	ax[0, 0].set_title("Theta 1 (the mean)")
	ax[0, 1].set_title("Theta 2 (the std)")

	# Theta 1
	ax[1, 0].axhline(xmin = 0., xmax = 1., y = 0., linestyle = 'dashed'
				, color = 'black')
	ax[1, 0].plot(data['x_p'][training_index, 0] - data['x_m'][training_index, 0]
				, color = 'C0',alpha=0.5)
	ax[1, 0].plot(data['x_p_test'][test_index, 0] - data['x_m_test'][test_index, 0]
				, color = 'C1',alpha=0.5)
	ax[1, 0].set_xticks([])
	ax[1, 0].set_ylabel("Difference between derivative data amplitudes")

	# Theta 2
	ax[1, 1].axhline(xmin = 0., xmax = 1., y = 0., linestyle = 'dashed'
				, color = 'black')
	ax[1, 1].plot(data['x_p'][training_index, 1] - data['x_m'][training_index, 1]
				, color = 'C0',alpha=0.5)
	ax[1, 1].plot(data['x_p_test'][test_index, 1] - data['x_m_test'][test_index, 1]
				, color = 'C1',alpha=0.5)
	ax[1, 1].set_xticks([])
	ax[1, 1].set_ylabel("Difference between derivative data amplitudes");

	plt.savefig('./Figures/1d_gaussian2params/derivatives_visualization_2summaries.png')
	plt.close()

plot_derivatives()

parameters = {
    'verbose': True,
    'number of simulations': n_s,
    'fiducial θ': np.array([theta_fid,theta2_fid]), # I think it works like this
    'derivative denominator': der_den,
    'differentiation fraction': derivative_fraction,
    'number of summaries': 2,
    'calculate MLE': True,
    'prebuild': True,
    'input shape': input_shape,
    'preload data': data,
    'save file': "data_oneD/saved_model",
    'wv': 0.,
    'bb': 0.1,
    'activation': tf.nn.leaky_relu,
    'α': 0.01,
    'hidden layers': [256,256,256]
}

# Initialize the IMNN
n = IMNN.IMNN(parameters=parameters)
eta = 1e-3
# Initialize input tensors, build network and define optimalization scheme
n.setup(η = eta)
# can change the optimization scheme (although adam was found to be unstable)
# n.backpropagate = tf.train.AdamOptimizer(eta).minimize(n.Λ) apparently doesnt work

num_epochs = 700
keep_rate = 0.8

n.train(num_epochs = num_epochs, n_train = n_train, keep_rate = keep_rate
	, data = data, history = True)

def plot_variables():
	fig, ax = plt.subplots(7, 1, sharex = True, figsize = (8, 14))
	plt.subplots_adjust(hspace = 0)
	end = len(n.history["det(F)"])
	epochs = np.arange(end)
	a, = ax[0].plot(epochs, n.history["det(F)"], label = 'Training data')
	b, = ax[0].plot(epochs, n.history["det(test F)"], label = 'Test data')
	ax[0].axhline(y=5,ls='--',color='k')
	ax[0].legend(frameon = False)
	ax[0].set_ylabel(r'$|{\bf F}_{\alpha\beta}|$')
	ax[0].set_title('Final Fisher info on test data: %.3f'%n.history["det(test F)"][-1])
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

	# Derivative first summary wrt to theta1                   theta1 is column 0
	ax[3].plot(epochs, np.array(n.history["dμdθ"])[:,0,0].flatten()
		, color = 'C0', label='theta1')
	# Derivative first summary wrt to theta2                   theta1 is column 1
	ax[3].plot(epochs, np.array(n.history["dμdθ"])[:,1,0].flatten()
		, color = 'C0', ls='dashed', label='theta2')


	ax[3].plot(epochs, np.array(n.history["test dμdθ"])[:,0, 0].flatten()
		, color = 'C1', label='theta1')
	ax[3].plot(epochs, np.array(n.history["test dμdθ"])[:,1, 0].flatten()
		, color = 'C1', ls='dashed', label='theta2')
	ax[3].legend(frameon=False)

	ax[3].set_ylabel(r'$\partial\mu 1/\partial\theta$')
	ax[3].set_xlabel('Number of epochs')
	ax[3].set_xlim([0, len(epochs)])

	# Derivative second summary wrt to theta1                   theta1 is column 0
	ax[4].plot(epochs, np.array(n.history["dμdθ"])[:,0,1].flatten()
		, color = 'C0', label='theta1')
	# Derivative second summary wrt to theta2                   theta1 is column 1
	ax[4].plot(epochs, np.array(n.history["dμdθ"])[:,1,1].flatten()
		, color = 'C0', ls='dashed', label='theta2')


	ax[4].plot(epochs, np.array(n.history["test dμdθ"])[:,0, 1].flatten()
		, color = 'C1', label='theta1')
	ax[4].plot(epochs, np.array(n.history["test dμdθ"])[:,1, 1].flatten()
		, color = 'C1', ls='dashed', label='theta2')
	ax[4].legend(frameon=False)

	ax[4].set_ylabel(r'$\partial\mu 2 /\partial\theta$')
	ax[4].set_xlabel('Number of epochs')
	ax[4].set_xlim([0, len(epochs)])
	
	ax[5].plot(epochs, np.array(n.history["μ"])[:,0])
	ax[5].plot(epochs, np.array(n.history["test μ"])[:,0])
	ax[5].set_ylabel('μ 1')
	ax[5].set_xlabel('Number of epochs')
	ax[5].set_xlim([0, len(epochs)])

	ax[6].plot(epochs, np.array(n.history["μ"])[:,1])
	ax[6].plot(epochs, np.array(n.history["test μ"])[:,1])
	ax[6].set_ylabel('μ 2')
	ax[6].set_xlabel('Number of epochs')
	ax[6].set_xlim([0, len(epochs)])

	plt.savefig('./Figures/1d_gaussian2params/variables_vs_epochs_2summaries.png')
	plt.close()

	print ('Maximum Fisher info on train data:',np.max(n.history["det(F)"]))
	print ('Final Fisher info on train data:',(n.history["det(F)"][-1]))
	
	print ('Maximum Fisher info on test data:',np.max(n.history["det(test F)"]))
	print ('Final Fisher info on test data:',(n.history["det(test F)"][-1]))


plot_variables()

# ===============================================================
# Approximate Bayesian computation with the calculated summary:

# First calculate the real data
real_data = generate_data(theta_fid,theta2_fid, 1, train = False)

def ABC():

	def show_real_data():
		fig, ax = plt.subplots(1, 1, figsize = (10, 12))
		ax.imshow(real_data[0, :, :, 0])
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlabel('Simulated real image');
		plt.savefig('./Figures/1d_gaussian2params/real_data_2summaries.png')
		plt.close()

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
	# passed to the function as data = data

	# sampled parameter values, summary of real data, summaries of generated data
	# distances of generated data to real data, Fisher info of real data
	theta, summary, s, ro, F = n.ABC(real_data = real_data, prior = [0, 10]
		, draws = 100000, generate_simulation = generate_data
		, at_once = False, data = data)
	#at_once = False will create only one simulation at a time


	# Draws are accepted if the distance between the simulation summary and the 
	# simulation of real data are close (i.e., smaller than some value epsilon)
	
	# For the first summary
	epsilon1 = abs(summary[0]/10) # chosen quite arbitrarily
	accept_indices1 = np.argwhere(ro < epsilon1)[:, 0]
	reject_indices1 = np.argwhere(ro >= epsilon1)[:, 0]

	# For the second summary
	epsilon2 = abs(summary[1]/10) # chosen quite arbitrarily
	accept_indices2 = np.argwhere(ro < epsilon2)[:, 0]
	reject_indices2 = np.argwhere(ro >= epsilon2)[:, 0]

	# plot output samples and histogram of the accepted samples
	# accept_indices_total = np.bitwise_and( (ro < epsilon1), ro < epsilon2 )


	def plot_samples():
		fig, ax = plt.subplots(2, 2, sharex = 'col', figsize = (10, 10))
		# plt.subplots_adjust(hspace = 0)
		theta1 = theta[:,0]
		theta2 = theta[:,1]

		ax[0, 0].set_title('Epsilon is chosen to be %.2f'%epsilon1)
		ax[0, 0].scatter(theta1[accept_indices1] , s[accept_indices1, 0], s = 1)
		ax[0, 0].scatter(theta1[reject_indices1], s[reject_indices1, 0], s = 1, alpha = 0.1)
		ax[0, 0].plot([0, 10], [summary[0], summary[0]], color = 'black', linestyle = 'dashed')
		ax[0, 0].set_ylabel('Network output', labelpad = 0)
		ax[0, 0].set_xlim([0, 10])
		ax[1, 0].hist(theta1[accept_indices1], np.linspace(0, 10, 100)
			, histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
		ax[1, 0].set_xlabel('$\\theta_1$')
		ax[1, 0].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
		ax[1, 0].set_yticks([])

		ax[0, 1].scatter(theta2[accept_indices1] , s[accept_indices1, 0], s = 1)
		ax[0, 1].scatter(theta2[reject_indices1], s[reject_indices1, 0], s = 1, alpha = 0.1)
		ax[0, 1].plot([0, 10], [summary[0], summary[0]], color = 'black', linestyle = 'dashed')
		ax[0, 1].set_ylabel('Network output', labelpad = 0)
		ax[0, 1].set_xlim([0, 10])
		ax[1, 1].hist(theta2[accept_indices1], np.linspace(0, 10, 100)
			, histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
		ax[1, 1].set_xlabel('$\\theta_2$')
		ax[1, 1].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
		ax[1, 1].set_yticks([])

		#  This doesnt make sense to do
		# ax[2, 0].set_title('Epsilon is chosen to be %.2f'%epsilon2)
		# ax[2, 0].scatter(theta1[accept_indices2] , s[accept_indices2, 0], s = 1)
		# ax[2, 0].scatter(theta1[reject_indices2], s[reject_indices2, 0], s = 1, alpha = 0.1)
		# ax[2, 0].plot([0, 10], [summary[1], summary[1]], color = 'black', linestyle = 'dashed')
		# ax[2, 0].set_ylabel('Network output', labelpad = 0)
		# ax[2, 0].set_xlim([0, 10])
		# ax[3, 0].hist(theta1[accept_indices2], np.linspace(0, 10, 100)
		# 	, histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
		# ax[3, 0].set_xlabel('$\\theta_1$')
		# ax[3, 0].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
		# ax[3, 0].set_yticks([])

		# ax[2, 1].scatter(theta2[accept_indices2] , s[accept_indices2, 0], s = 1)
		# ax[2, 1].scatter(theta2[reject_indices2], s[reject_indices2, 0], s = 1, alpha = 0.1)
		# ax[2, 1].plot([0, 10], [summary[1], summary[1]], color = 'black', linestyle = 'dashed')
		# ax[2, 1].set_ylabel('Network output', labelpad = 0)
		# ax[2, 1].set_xlim([0, 10])
		# ax[3, 1].hist(theta2[accept_indices2], np.linspace(0, 10, 100)
		# 	, histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
		# ax[3, 1].set_xlabel('$\\theta_2$')
		# ax[3, 1].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
		# ax[3, 1].set_yticks([])

		plt.savefig('./Figures/1d_gaussian2params/ABC_2summaries.png')
		plt.show()

	plot_samples()

	# There can be a lot of theta draws which are unconstrained by the network
	# because no similar structures were seen in the data, which is indicative of
	# using too small of a small training set

	return theta, accept_indices1, accept_indices2

theta, accept_indices1, accept_indices2 = ABC()

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
		plt.savefig('./Figures/1d_gaussian2params/PMC_ABC_2summaries.png')
		plt.close()
		# plt.show()
	plot()

	return theta_

np.wait_here()
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
		plt.savefig('./Figures/1d_gaussian2params/Gaussian_MLE_dense_2summaries.png')
		plt.close()
	return MLE, asymptotic_likelihood

# MLE, asymptotic_likelihood = first_order_Gaussian_MLE()
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
# ax.plot(np.linspace(0, 10, 1000), asymptotic_likelihood[0, 0], color = 'C0'
# 	, linewidth = 1.5, label = "Asymptotic Gaussian likelihood")
# ax.axvline(x = MLE[0, 0], ymin = 0., ymax = 1., linestyle = 'dashed'
	# , color = 'black', label = "Maximum likelihood estimate")
ax.axvline(x = 1., ymin = 0., ymax = 1., linestyle = 'dashed'
	, color = 'red', label = "Actual Fiducial Parameter")
ax.legend(frameon = False)
ax.set_xlim([0, 10])
ax.set_xlabel('θ')
ax.set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
ax.set_yticks([])
plt.savefig('./Figures/1d_gaussian2params/likelihoods_dense_2summaries.png')
# plt.show()
plt.close()