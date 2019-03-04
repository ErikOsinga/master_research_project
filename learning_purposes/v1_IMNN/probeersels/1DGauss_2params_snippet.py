# coding=utf-8

'''
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

def generate_data(theta1, theta2, n_samples = 1, train=False):
	'''

	Returns: array of shape (n_samples,input_shape) draws 
	from normal dist with mean=theta1 and var = theta2
	'''
	mean = theta1
	std = np.sqrt(theta2)
	
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


# 10000 data points of Gaussian zero-mean noise
input_shape = [10000] 
# Fiducial parameter and perturbed values just above and below
theta_fid = 3. # mean
theta2_fid = 1. # variance
delta_theta1 = 0.1
delta_theta2 = 0.1

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
t_m1 = generate_data(theta_fid - delta_theta1, theta2_fid,(n_train*n_p), train = True) 
# Perturb theta2 minus delta theta
t_m2 = generate_data(theta_fid, theta2_fid - delta_theta2,(n_train*n_p), train= True)
# Concatenate these into a vector
t_m = np.concatenate( (t_m1, t_m2), axis=1) # I think this is the correct way

# set same seed to surpress the sample variance
np.random.seed(seed)
# Perturb theta1 plus delta theta
t_p1 = generate_data(theta_fid + delta_theta1,theta2_fid,(n_train*n_p), train = True) 
# Perturb theta2 plus delta theta
t_p2 = generate_data(theta_fid, theta2_fid + delta_theta2,(n_train*n_p), train= True)
t_p = np.concatenate( (t_p1, t_p2), axis=1) # I think this is the correct way

np.random.seed()

# denominator of the derivative 
# needs to be stored in an array of shape [number of parameters]
der_den = np.array([1. / (2. * delta_theta1), 1. / (2. * delta_theta2)]) 

data = {"x_central": t, "x_m": t_m, "x_p":t_p}

# Repeat the same story to generate training data
tt = generate_data(theta_fid,theta2_fid, n_s, train=False)
seed = np.random.randint(1e6)
np.random.seed(seed)
# Perturb minus delta theta
tt_m1 = generate_data(theta_fid - delta_theta1,theta2_fid,n_p, train=True)
tt_m2 = generate_data(theta_fid, theta2_fid - delta_theta2,n_p, train=True)
tt_m = np.concatenate( (tt_m1, tt_m2), axis=1)

np.random.seed(seed)
# Perturb plus delta theta
tt_p1 = generate_data(theta_fid + delta_theta1, theta2_fid,n_p, train=True)
tt_p2 = generate_data(theta_fid, theta2_fid + delta_theta2, n_p, train=True)
tt_p = np.concatenate( (tt_p1, tt_p2), axis=1)
np.random.seed()
data["x_central_test"] = tt
data["x_m_test"] = tt_m
data["x_p_test"] = tt_p

parameters = {
    'verbose': True,
    'number of simulations': n_s,
    'fiducial θ': np.array([theta_fid,theta2_fid]), # I think it works like this
    'derivative denominator': der_den,
    'differentiation fraction': derivative_fraction,
    'number of summaries': 1,
    'calculate MLE': True,
    'prebuild': True,
    'input shape': input_shape,
    'preload data': data,
    'save file': "data_oneD/saved_model",
    'wv': 0.,
    'bb': 0.1,
    'activation': tf.nn.leaky_relu,
    'α': 0.01,
    # 'hidden layers': [256,256]
    'hidden layers': [512,256,256,256]
}

# Initialize the IMNN
n = IMNN.IMNN(parameters=parameters)
eta = 1e-3
# Initialize input tensors, build network and define optimalization scheme
n.setup(η = eta)
# can change the optimization scheme (although adam was found to be unstable)
# n.backpropagate = tf.train.AdamOptimizer(eta).minimize(n.Λ)  # apparently doesnt work

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
	
	'''
	# Derivative wrt to theta1                   theta1 is column 0
	ax[3].plot(epochs, np.array(n.history["dμdθ"])[:,0].flatten()
		, color = 'C0', label='theta1',alpha=0.5)
	# Derivative wrt to theta2                   theta1 is column 1
	ax[3].plot(epochs, np.array(n.history["dμdθ"])[:,1].flatten()
		, color = 'C0', ls='dashed', label='theta2',alpha=0.5)

	ax[3].plot(epochs, np.array(n.history["test dμdθ"])[:,0].flatten()
		, color = 'C1', label='theta1',alpha=0.5)
	ax[3].plot(epochs, np.array(n.history["test dμdθ"])[:,1].flatten()
		, color = 'C1', ls='dashed', label='theta2',alpha=0.5)
	ax[3].legend(frameon=False)

	ax[3].set_ylabel(r'$\partial\mu/\partial\theta$')
	ax[3].set_xlabel('Number of epochs')
	ax[3].set_xlim([0, len(epochs)])
	ax[4].plot(epochs, np.array(n.history["μ"]).reshape((np.prod(np.array(n.history["μ"]).shape))),alpha=0.5)
	ax[4].plot(epochs, np.array(n.history["test μ"]).reshape((np.prod(np.array(n.history["test μ"]).shape))),alpha=0.5)
	ax[4].set_ylabel('μ')
	ax[4].set_xlabel('Number of epochs')
	ax[4].set_xlim([0, len(epochs)])
	'''

	# plt.savefig('./Figures/1d_gaussian2params/variables_vs_epochs.png')
	plt.show()
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
# ABC code is omitted