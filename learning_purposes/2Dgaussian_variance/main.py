import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# change to the path where the IMNN git clone is located
sys.path.insert(-1,'../../../information_maximiser')
import IMNN # make sure the path to the IMNN is given
import tqdm
sys.path.insert(-1,'../../') # change to path where utils_mrp is located
import utils_mrp

"""
Summarizing a 2D gaussian with known mean and unkown variance of the axes
both axes are uncorrelated, and have the same variance
"""


class nholder(object):
	"""
	Class to hold and store all parameters/data for an IMNN 
	"""
	def __init__(self, input_shape, generate_data, theta_fid, delta_theta, n_s, n_train, 
		derivative_fraction, eta, parameters, num_epochs, keep_rate, verbose, 
		version):
		"""
		INITIALIZE PARAMETERS
		#______________________________________________________________
		input_shape					list	shape of the data that is generated
		generate_data				list	function to generate the data
		theta_fid					list	fiducial parameter values
		delta_theta					list 	perturbation values for fiducial param
		n_s 						int 	number of simulations
		n_train						int 	number of splits, to make more simulations
		derivative_fraction 		float	fraction of n_s to use for derivatives
		eta 						float	learning rate
		parameters 					dict 	dict of parameters to feed IMNN
		num_epochs 					int 	amount of epochs
		keep_rate 					float 	(1-dropout rate), amount of nodes to keep every batch
		verbose 					int 	TODO
		version 					float	version ID of this particular network
		"""

		tf.reset_default_graph()
		self.input_shape = input_shape
		self.generate_data = generate_data
		self.theta_fid = theta_fid
		self.delta_theta = delta_theta
		self.n_s = n_s
		self.n_train = n_train
		self.n_p = int(n_s * derivative_fraction)
		self.derivative_fraction = derivative_fraction
		self.eta = eta
		self.num_epochs = num_epochs
		self.keep_rate = keep_rate
		self.verbose = verbose

		self.data, self.der_den = self.create_data()
		# Make parameters dictionary of params that are always the same or defined
		# by other parameters	
		self.parameters = { 'number of simulations': self.n_s,
							'preload data': self.data,
							'derivative denominator': self.der_den,
							'number of simulations': self.n_s,
							'fiducial θ': self.theta_fid,
							'differentiation fraction': self.derivative_fraction,
							'input shape': self.input_shape,
						}
		# Add user parameters to this dictionary
		for key, value in parameters.items():
			self.parameters[key] = value

		# For saving the settings
		self.modelversion = version 
		self.modelloc = 'Models/' #location where the models (networks) are saved
		
		#the file in which the network settings will be saved
		self.modelsettings_name = 'modelsettings2.csv' 

		self.modelsettings = {'Version' : str(self.modelversion),
						'Learning rate': str(self.eta),
						'Keep rate': str(self.keep_rate),
						'num_epochs': str(self.num_epochs),
						'n_train': str(self.n_train),
						'delta_theta': str(self.delta_theta)
						}
		# Add user parameters to modelsettings
		# except these from the parameters dictionary
		not_save = ['preload data', 'derivative denominator', 'verbose']
		for key, value in self.parameters.items():
			if key == 'activation':
				# e.g., save only the string 'leaky relu'
				value = str(value).split(' ')[1]
			elif key in not_save:
				continue
			self.modelsettings[key] = str(value) # parse everything to string

		# Holders for the Final F train and Final F test after training network
		self.modelsettings['Final detF train'] = ''
		self.modelsettings['Final detF test'] = ''

		# For saving the figures
		self.figuredir = 'Figures/'

		# For saving the network history
		self.historydir = 'History/'

		# Check if folders exist, create directory if necessary
		utils_mrp.checkFolders([self.modelloc, self.figuredir, self.historydir])

		# Check if modelsettings.csv file exists, create if necessary
		utils_mrp.checkFiles([self.modelsettings_name])

		# Save settings for this model
		utils_mrp.save_model_settings(self, self.modelsettings)

	def create_data(self):
		"""
		Generate the training and test data for the network
		Called as soon as the network is initialized

		RETURNS
		#______________________________________________________________

		data 					dict 	dict containing training and test data
		der_den					array	array containing the derivative denominator

		"""

		# Number of upper and lower simulations
		n_p = int(self.n_s * self.derivative_fraction)

		# set a seed to surpress the sample variance
		seed = np.random.randint(1e6)
		np.random.seed(seed)
		# Perturb lower 
		t_m = self.generate_data(np.array([self.theta_fid for i in 
					range(self.n_train * self.n_p)]), train = -self.delta_theta)
		np.random.seed(seed)
		# Perturb higher 
		t_p = self.generate_data(np.array([theta_fid for i in 
					range(self.n_train * self.n_p)]), train = self.delta_theta)
		np.random.seed()

		t = self.generate_data(np.array([self.theta_fid for i in 
					range(self.n_train * self.n_s)]), train = None)
		np.random.seed()

		der_den = 1. / (2. * self.delta_theta)

		data = {"x_central": t, "x_m": t_m, "x_p":t_p}

		# Repeat the same story to generate training data
		seed = np.random.randint(1e6)
		np.random.seed(seed)
		# Perturb lower 
		tt_m = self.generate_data(np.array([self.theta_fid for i in 
					range(self.n_train * self.n_p)]), train = -self.delta_theta)
		np.random.seed(seed)
		# Perturb higher 
		tt_p = self.generate_data(np.array([self.theta_fid for i in 
					range(self.n_train * self.n_p)]), train = self.delta_theta)
		np.random.seed()

		tt = self.generate_data(np.array([self.theta_fid for i in 
					range(self.n_train * self.n_s)]), train = None)
		np.random.seed()
		data["x_central_test"] = tt
		data["x_m_test"] = tt_m
		data["x_p_test"] = tt_p

		return data, der_den

	def plot_data(self, show=False):
		""" 
		Plot the data 

		Since it is a 2D multivariate gaussian we plot the distribution
		of the datapoints in 2D space

		VARIABLES
		#______________________________________________________________
		show 					bool	whether or not plt.show() is called
		data 					dict 	dict containing training and test data

		"""

		fig, ax = plt.subplots(1, 2, figsize = (10, 6))
		# plot one random row from the simulated data 
		ax[0].plot(*self.data['x_central'][np.random.randint(self.n_train * self.n_s)].T[:,0],'x')
		ax[0].axis('equal')
		ax[0].set_title('Training data')
		
		x, y = self.data['x_central_test'][np.random.randint(self.n_s)].T[:,0]
		ax[1].plot(x,y,'x')
		ax[1].axis('equal')
		ax[1].set_title('Test data')

		plt.savefig(f'{self.figuredir}data_visualization_{self.modelversion}.png')
		if show: plt.show()
		plt.close()

	def plot_derivatives(self, show=False):
		""" 
		Plot the upper and lower perturbed data as data amplitude
		Good to check if the sample variance is being 
		surpressed. This needs to be done or the network learns very slowly

		VARIABLES
		#______________________________________________________________
		show 					bool	whether or not plt.show() is called
		data 					dict 	dict containing training and test data

		"""

		fig, ax = plt.subplots(3, 2, figsize = (15, 10))
		# plt.subplots_adjust(wspace = 0, hspace = 0.1)
		training_index = np.random.randint(self.n_train * self.n_p)
		
		x, y = self.data['x_p'][training_index].T[:,0]
		
		ax[0, 0].plot(x,y,'x',label='$θ_1$')
		ax[0, 0].set_title('Upper training image')
		ax[0, 0].set_xlim(-3,3)
		ax[0, 0].set_ylim(-3,3)

		ax[1, 0].plot(*self.data['x_m'][training_index].T[:,0],'x')
		ax[1, 0].set_title('Lower training image')
		ax[1, 0].set_xlim(-3,3)
		ax[1, 0].set_ylim(-3,3)
		
		xm, ym = self.data["x_m"][training_index].T[:,0]
		xp, yp = self.data["x_p"][training_index].T[:,0]
		ax[2, 0].plot(xp-xm,yp-ym,'x')
		ax[2, 0].set_title('Difference between upper and lower training images');
		ax[2, 0].axhline(xmin = 0., xmax = 1., y = 0.
			, linestyle = 'dashed', color = 'black')
		test_index = np.random.randint(self.n_p)
		ax[0, 1].plot(*self.data['x_p_test'][test_index].T[:,0],'x')
		ax[0, 1].set_title('Upper test image')
		ax[1, 1].plot(*self.data['x_m_test'][training_index].T[:,0],'x')
		ax[1, 1].set_title('Lower test image')
		
		xm, ym = self.data["x_m_test"][test_index].T[:,0]
		xp, yp = self.data["x_p_test"][test_index].T[:,0]
		ax[2, 1].plot(xp-xm,yp-ym,'x')
		ax[2, 1].axhline(xmin = 0., xmax = 1., y = 0.
			, linestyle = 'dashed', color = 'black')
		ax[2, 1].set_title('Difference between upper and lower test images')

		plt.savefig(f'{self.figuredir}derivatives_visualization_{self.modelversion}.png')
		if show: plt.show()
		plt.close()

	def create_network(self):
		""" 
		Create the network with the given data and parameters

		INPUTS
		#______________________________________________________________
		data 					dict 	dict containing training and test data
		
		RETURNS
		#_____________________________________________________________
		n 						class 	IMNN class as defined in IMNN.py

		"""
		n = IMNN.IMNN(parameters=self.parameters)
		tf.reset_default_graph()
		n.setup(η = eta)
		
		return n

	def train_network(self, n):
		""" 
		Train the created network with the given data and parameters
		Saves the history and the determinant of the final fisher info

		INPUTS
		#______________________________________________________________
		n 						class 	IMNN class as defined in IMNN.py
		
		"""

		n.train(num_epochs = self.num_epochs, n_train = self.n_train
			, keep_rate = self.keep_rate, data = self.data, history = True)

		# save the network history to a file
		utils_mrp.save_history(self, n)

		# save the det(Final Fisher info) in the modelsettings.csv file
		utils_mrp.save_final_fisher_info(self, n)

	def plot_variables(self, n, show=False):
		""" 
		Plot variables vs epochs

		INPUTS
		#______________________________________________________________
		n 						class 	IMNN class as defined in IMNN.py
		
		"""
		fig, ax = plt.subplots(5, 1, sharex = True, figsize = (8, 14))
		plt.subplots_adjust(hspace = 0)
		end = len(n.history["det(F)"])
		epochs = np.arange(end)
		a, = ax[0].plot(epochs, n.history["det(F)"], label = 'Training data')
		b, = ax[0].plot(epochs, n.history["det(test F)"], label = 'Test data')
		# ax[0].axhline(y=5,ls='--',color='k')
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

		print ('Maximum Fisher info on train data:',np.max(n.history["det(F)"]))
		print ('Final Fisher info on train data:',(n.history["det(F)"][-1]))
		
		print ('Maximum Fisher info on test data:',np.max(n.history["det(test F)"]))
		print ('Final Fisher info on test data:',(n.history["det(test F)"][-1]))

		if np.max(n.history["det(test F)"]) == n.history["det(test F)"][-1]:
			print ('Promising network found, possibly more epochs needed')

		plt.savefig(f'{self.figuredir}variables_vs_epochs_{self.modelversion}.png')
		if show: plt.show()
		plt.close()

	def ABC(self, n, real_data, prior, draws, show=False):
		""" 
		Perform ABC
		Only a uniform prior is implemented at the moment.

		INPUTS
		#______________________________________________________________
		n 						class 	IMNN class as defined in IMNN.py
		real_data 				array 	array containing true data
		prior 					list 	lower and upper bound for uniform prior
		draws 					int 	amount of draws from prior
		show 					bool	whether or not plt.show() is called
		

		RETURNS
		#_____________________________________________________________
		theta					list 	sampled parameter values			
		accept_indices 			list 	indices of theta that satisfy (ro < epsilon)
	
		"""

		# If the data is not preloaded as a tensorflow constant then the data can be
		# passed to the function as data = data

		# sampled parameter values, summary of real data, summaries of generated data
		# distances of generated data to real data, Fisher info of real data
		theta, summary, s, ro, F = n.ABC(real_data = real_data, prior = prior
			, draws = draws, generate_simulation = self.generate_data
			, at_once = True, data = self.data)
		#at_once = False will create only one simulation at a time

		# Draws are accepted if the distance between the simulation summary and the 
		# simulation of real data are close (i.e., smaller than some value epsilon)
		epsilon = abs(summary/10) # chosen quite arbitrarily
		accept_indices = np.argwhere(ro < epsilon)[:, 0]
		reject_indices = np.argwhere(ro >= epsilon)[:, 0]


		# plot output samples and histogram of the accepted samples
		def plot_samples():
			fig, ax = plt.subplots(2, 2, sharex = 'col', figsize = (10, 10))
			plt.subplots_adjust(hspace = 0)
			theta1 = theta[:,0]
			theta2 = theta[:,1]

			ax[0, 0].set_title('Epsilon is chosen to be %.2f'%epsilon)
			ax[0, 0].scatter(theta1[accept_indices] , s[accept_indices, 0], s = 1)
			ax[0, 0].scatter(theta1[reject_indices], s[reject_indices, 0], s = 1, alpha = 0.1)
			ax[0, 0].plot([0, 10], [summary[0], summary[0]], color = 'black', linestyle = 'dashed')
			ax[0, 0].set_ylabel('Network output', labelpad = 0)
			ax[0, 0].set_xlim([0, 10])
			ax[1, 0].hist(theta1[accept_indices], np.linspace(0, 10, 100)
				, histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
			ax[1, 0].set_xlabel('$\\theta_1$')
			ax[1, 0].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
			ax[1, 0].set_yticks([])

			ax[0, 1].scatter(theta2[accept_indices] , s[accept_indices, 0], s = 1)
			ax[0, 1].scatter(theta2[reject_indices], s[reject_indices, 0], s = 1, alpha = 0.1)
			ax[0, 1].plot([0, 10], [summary[0], summary[0]], color = 'black', linestyle = 'dashed')
			ax[0, 1].set_ylabel('Network output', labelpad = 0)
			ax[0, 1].set_xlim([0, 10])
			ax[1, 1].hist(theta2[accept_indices], np.linspace(0, 10, 100)
				, histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
			ax[1, 1].set_xlabel('$\\theta_2$')
			ax[1, 1].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
			ax[1, 1].set_yticks([])

			plt.savefig(f'{self.figuredir}ABC_{self.modelversion}.png')
			if show: plt.show()
			plt.close()

		plot_samples()

		# There can be a lot of theta draws which are unconstrained by the network
		# because no similar structures were seen in the data, which is indicative of
		# using too small of a small training set

		return theta, accept_indices

def gauss_2D(mean, variance):
	"""
	If the two axes are uncorrelated we can just generate 2 one-D Gaussians

	mean -- list of 2 numbers, mean along X and Y dimension
	sigma -- list of variances (theta)       (cov = np.eye(2)*sigma)

	RETURNS
	np array of shape  (len(sigma), 2)

	"""
	assert input_shape[-1] == 2, "2D Gaussian, input shape must be (,2)"

	sigma = np.sqrt(variance)										
	# input_shape[0] for just the amount of datapints
	x = np.random.normal(mean[0],sigma, [variance.shape[0]] + [input_shape[0]] )
	y = np.random.normal(mean[1],sigma, [variance.shape[0]] + [input_shape[0]] )
	# x and y are of shape (num_sim, input_shape[0])
	# by stacking them we get (2, num_sim, input_shape[0])
	# so we move the first axis to the last, (num,sim, input_shape[0], 2)
	# and add dummy axis at one before last position
	return np.expand_dims(np.moveaxis(np.asarray([x,y]),0, -1),-2) 

def generate_data(θ, train = None):
	"""
	Fiducial parameter is passed as a list so many simulations can be made
	at once

	θ -- the variance of dimension X and Y (diagonal)
	Train -- whether the upper and lower derivatives are calculated

	Returns array of shape
	(num_simulations, input_shape)
	"""
	mean = [0,0] # known
	
	if train is not None:
		if len(θ.shape) == 1: # to be able to generalize and also do 1 parameter
			num_params = 1
		else:
			num_params = θ.shape[1]

		holder = np.zeros([θ.shape[0]] + [θ.shape[1]] + input_shape)
		# shape (amount of simulations, amount of parameters, input shape)
		for i in range(num_params): # for i in range (num params)
			params = np.copy(θ)
			params[:, i] += train[i] # add delta theta for this parameter
			holder[:, i, :] = gauss_2D(mean, params)
		return holder
	else:
		return gauss_2D(mean,θ)


# ALL PARAMETERS
#_______________________________________________________
input_shape = [10,1,2] # This is <input_shape[0]> 2D data points with a dummy axis

theta_fid = np.array([1.]) # variance
delta_theta = np.array([0.1]) # perturbation values
n_s = 1000 # number of simulations
n_train = 1 # splits, for if it doesnt fit into memory
# use less simulations for numerical derivative
derivative_fraction = 0.20
eta = 1e-4
num_epochs = int(2e4)
keep_rate = 0.6
verbose = 0
# CNN
hidden_layers = [[10, [5, 5], [2, 2], 'SAME'], [6, [3, 3], [1, 1], 'SAME'], 100, 100]

initial_version = 1006

version = initial_version

parameters = {
	'verbose': False,
	'number of summaries': 1,
	'calculate MLE': True,
	'prebuild': True, # allow IMNN to build network
	'save file': "Models/data/model"+str(version),
	'wv': 0., # the variance with which to initialise the weights
	'bb': 0.1, # the constant value with which to initialise the biases
	'activation': tf.nn.leaky_relu,
	'α': 0.01, # negative gradient parameter 
	'hidden layers': hidden_layers
}
#______________________________________________________

# Network holder
nholder1 = nholder(input_shape, generate_data, theta_fid, delta_theta, n_s,
		n_train, derivative_fraction, eta, parameters, num_epochs, keep_rate,
		verbose, version)

# # IMNN network
n = nholder1.create_network()
# plot data
nholder1.plot_data(show=False)
# plot derivatives
nholder1.plot_derivatives(show=False)
# # Train network
nholder1.train_network(n)
# # Plot the output
nholder1.plot_variables(n,show=True)
# # Perform ABC
# real_data = generate_data(np.array([theta_fid]), train = None)
# prior = [0, 6]
# draws = 100000
# nholder1.ABC(n, real_data, prior, draws, show=False)
