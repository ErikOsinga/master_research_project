"""
Folllowing the example of pmcabc on a 2D Gaussian 
from https://github.com/jakeret/abcpmc

This specific example:
http://nbviewer.jupyter.org/github/jakeret/abcpmc/blob/master/notebooks/2d_gauss.ipynb

Goal: Trying to extend the example by using the IMNN as summary statistic

"""
import sys
import numpy as np
import tensorflow as tf
import keras
# change to the path where the IMNN git clone is located
sys.path.insert(-1,'../../information_maximiser')
import IMNN # make sure the path to the IMNN is given
import tqdm
import abcpmc # find citation at https://github.com/jakeret/abcpmc
import corner # find citation at https://github.com/dfm/corner.py

import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

np.random.seed(10)

plt.rc('text', usetex=True)
plt.rc('axes', labelsize=15, titlesize=15)

def generate_data(theta1, theta2, n_samples = 1, train=False):
	'''

	Returns: array of shape (n_samples,input_shape,N) draws 
	from multivariate normal dist, where N is the dimension of variables
	(i.e., N = len(mean) )
	'''

	mean = [theta1,theta2] 
	cov = [[0.25,0],  # x_0 has variance 1, 
	       [0,0.25]] # x_1 has variance 2, no covariance  

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

# Same parameters as were used for training the network.
n_s = 1000
theta1_fid, theta2_fid = 1., 2.
delta_theta = 0.1
derivative_denominator = 1. / (2. * delta_theta)
der_den = np.array([derivative_denominator, derivative_denominator]) 
derivative_fraction = 0.05
input_shape = [100,1,2] # This is 100 2D data points with a dummy axis

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
    'preload data': data, # No idea why we should be giving this when only restoring the network
    'save file': "data_mean/model",
    'wv': 0.,
    'bb': 0.1,
    'activation': tf.nn.leaky_relu,
    'α': 0.01,
    # 'hidden layers': [128, 128] # Dense layers only doesnt work well for 2D data
    # Convolutional layers:
    'hidden layers': [[10, [5, 5], [2, 2], 'SAME'], [6, [3, 3], [1, 1], 'SAME'], 100, 100],

}

# Loading the trained network
n = IMNN.IMNN(parameters=parameters)
n.restore_network()

# The example assumes we know sigma and want to estimate the mean

samples_size = 100
sigma = np.eye(2) * 0.25
means = [1.0, 2.0]
data = np.random.multivariate_normal(means, sigma, samples_size)
# plt.matshow(sigma) # display array as matrix
# plt.title('Covariance matrix sigma')
# plt.colorbar()
# plt.show()

# generate data function
def generate_data_quick(theta):
	"""
	Equivalent function to generate_data but one that it optimized to be
	used with the abcpmc module
	"""
	theta1, theta2 = theta 
	# if (theta1 < 0)  or (theta2 < 0): 
	# 	# If theta1 or theta2 is negative return garbage
	# 	return np.asarray([1e9,1e9]*100).reshape(100,1,2)
	#print (theta1,theta2)
	mean = [theta1,theta2] 
	cov = np.eye(2)*0.25
	assert input_shape[-1] == 2, "Multivariate Gaussian generates (,2) shaped data"
	return np.random.multivariate_normal(mean,cov,size=input_shape[:-1],check_valid='raise')


def dist_measure(x, y):
	"""Sum of the abs difference of the mean of the simulated and obs data"""
	return np.sum(np.abs(np.mean(x, axis=0) - np.mean(y, axis=0)))

def MSE(x, y):
	""" Mean squared error distance measure"""
	return np.mean(np.power(x - y,2))

def euclidian(x,y):
	return np.linalg.norm(x-y)

def std(x,y):
	return abs(np.std(x)-np.std(y))

def summary_statistic_distance(x, y):
	"""
	Calculates the difference between the summary statistic of the simulated
	data X and the real data y
	"""
	x = x.reshape(1,*x.shape)
	y = y.reshape(1,*y.shape)
	summary_X = n.sess.run(n.output, feed_dict = {n.x: x, n.dropout: 1.})[0]
	summary_Y = n.sess.run(n.output, feed_dict = {n.x: y, n.dropout: 1.})[0]
	print ('Summary X: ', summary_X, 'Summary Y:',summary_X)
	return abs(summary_X-summary_Y)

''' Setup '''
# 'Best' guess about the distribution
# prior = abcpmc.GaussianPrior(mu=[1.0, 1.0], sigma=np.eye(2) * 0.5)

# 'Best' guess about the distribution, uniform distribution
prior = abcpmc.TophatPrior([0.0,0.0], [5.0,5.0])

# As threshold for accepting draws from the prior we use the alpha-th percentile
# of the sorted distances of the particles of the current iteration
alpha = 75
T = 20 # sample for T iterations
eps_start = 1.0 # sufficiently high starting threshold
eps = abcpmc.ConstEps(T, eps_start)


''' Sampling function '''
def launch(threads):
	eps = abcpmc.ConstEps(T, eps_start)

	pools = []
	# pool is a namedtuple representing the values of one iteration
	print ('Starting sampling now..')
	for pool in sampler.sample(prior, eps):
		print ("T: {0}, eps: {1:>.4f}, ratio: {2:>.4f}".format(
				pool.t, eps(pool.eps), pool.ratio))

		for i, (mean,std) in enumerate(zip(*abcpmc.weighted_avg_and_std(
												pool.thetas, pool.ws, axis=0))):
			print(u"    theta[{0}]: {1:>.4f} \u00B1 {2:>.4f}".format(i, mean,std))
		# reduce the eps value to the alpha-th percentile of the sorted distances
		eps.eps = np.percentile(pool.dists, alpha) 
		pools.append(pool)
	sampler.close()
	return pools


''' Postprocessing '''
def postprocessing(pools):

	"""Check the estimated value of theta1 and theta2 """
	for i in range(len(means)):
		moments = np.array([abcpmc.weighted_avg_and_std(
			pool.thetas[:,i], pool.ws, axis=0) for pool in pools])
		plt.errorbar(range(T),moments[:,0], moments[:,1], label='Theta %i'%i)
	plt.hlines(means, 0, T, linestyle='dotted', linewidth=0.7)
	plt.xlim([-.5,T])
	plt.xlabel("Iteration")
	plt.ylabel("Value")
	plt.legend()
	plt.savefig('./Figures/abcpmc/estimated_thetas_vs_iterations.png')
	# plt.show()
	plt.close()

	"""Check the distribution of distances for the posterior"""
	distances = np.array([pool.dists for pool in pools]).flatten()
	# If we are close to the true posterior, we expect to have a very high
	# bin count around the values we have found in the earlier distribution plot
	sns.distplot(distances, axlabel='distance')
	plt.ylabel('Counts')
	plt.savefig('./Figures/abcpmc/distance_distribution_posterior.png')
	# plt.show()
	plt.close()

	"""Check the epsilon value as function of iteration """
	eps_values = np.array([pool.eps for pool in pools])
	plt.plot(eps_values, label=r'$\epsilon$ values')
	plt.xlabel('Iteration')
	plt.ylabel(r'$\epsilon$')
	plt.legend()
	plt.savefig('./Figures/abcpmc/epsilon_vs_iterations.png')
	# plt.show()
	plt.close()

	""" Check the acceptance ratio"""
	acc_ratios = np.array([pool.ratio for pool in pools])
	plt.plot(acc_ratios, label='Acceptance ratio')
	plt.ylim([0,1])
	plt.xlabel('Iteration')
	plt.ylabel('Acceptance ratio')
	plt.legend()
	plt.savefig('./Figures/abcpmc/acceptance_vs_iterations.png')
	# plt.show()
	plt.close()

	"""Plot the posterior, visualize with 'corner' package """
	samples = np.vstack([pool.thetas for pool in pools])
	fig = corner.corner(samples, truths= means)
	plt.savefig('./Figures/abcpmc/posterior_all_iterations.png')
	# plt.show()
	plt.close()

	""" Plot the posterior, omitting the first iterations """
	idx = -1
	samples = pools[idx].thetas
	fig = corner.corner(samples, weights=pools[idx].ws, truths= means)
	for mean, std in zip(*abcpmc.weighted_avg_and_std(samples, pools[idx].ws, axis=0)):
		print(u"mean: {0:>.4f} \u00B1 {1:>.4f}".format(mean,std))
	plt.savefig('./Figures/abcpmc/posterior.png')
	# plt.show()


# Pools might fail if we dont execute this if statement
if __name__ == '__main__':
	threads = 10 # for some reason threads>1 gives a Broken pipe error

	# check the variability of the distances at the correct parameters
	# distances = [std(data, create_new_sample(means)) for _ in range(1000)]
	# sns.distplot(distances, axlabel="distances", )
	# plt.title("Variablility of distance from simulations")
	# plt.savefig('./Figures/abcpmc/variability_of_distances.png')
	# plt.show()
	# plt.close()
	# time.sleep(5)


	# Create an instance of the sampler. 5000 particles
	# The sampler HAS to be created in the __main__ thread else multiprocessing
	# does not work 
	sampler = abcpmc.Sampler(N=5000, Y=data, postfn=generate_data_quick
				, dist=summary_statistic_distance, threads=threads)

	# Optional: customize the proposal creation. 
	# Here we use Optimal Local Covariance Matrix - kernel. (Filipi et al. 2012)
	# sampler.particle_proposal_cls = abcpmc.OLCMParticleProposal

	# sampler = create_sampler(threads)
	t0 = time.time()
	pools = launch(threads)
	print ("took %.2f seconds"%(time.time() - t0))

	postprocessing(pools)
