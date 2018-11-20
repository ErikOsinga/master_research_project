import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# change to the path where the IMNN git clone is located
sys.path.insert(-1,'../../information_maximiser')
import IMNN # make sure the path to the IMNN is given
import tqdm
sys.path.insert(-1,'../') # change to path where utils_mrp is located
import utils_mrp

"""
Summarizing a 1D gaussian with unkown mean and variance

"""


class nholder(object):
	"""
	Class to hold and store all parameters for an IMNN 
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
		self.modelsettings_name = 'modelsettings.csv' 

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
		Plot the data as data amplitude

		VARIABLES
		#______________________________________________________________
		show 					bool	whether or not plt.show() is called
		data 					dict 	dict containing training and test data

		"""

		fig, ax = plt.subplots(1, 1, figsize = (10, 6))
		# plot one random row from the simulated data 
		ax.plot(self.data['x_central'][np.random.randint(self.n_train * self.n_s)]
			, label = "training data")

		ax.plot(self.data['x_central_test'][np.random.randint(self.n_s)]
			, label = "test data")

		ax.legend(frameon = False)
		# ax.set_xlim([0, 9])
		ax.set_xticks([])
		ax.set_ylabel("Data amplitude")
		plt.savefig(f'{self.figuredir}data_visualization_{self.modelversion}.png')
		if show: plt.show()
		plt.close()

	def plot_data_hist(self, show=False):
		""" 
		Plot the data as histogram 

		INPUTS
		#______________________________________________________________
		show 					bool	whether or not plt.show() is called
		data 					dict 	dict containing training and test data

		"""
		fig, ax = plt.subplots(2, 1 ,figsize= (15,10))

		ax[0].hist(self.data['x_central'][np.random.randint(self.n_train*self.n_s)]
			,label='training data',alpha=0.5)

		ax[0].legend(frameon = False)
		ax[0].set_xlabel("Data amplitude")
		ax[0].set_ylabel('Counts')
		ax[0].set_title('%i data points'%self.input_shape[0])
		ax[0].set_xlim(0,6)

		ax[1].hist(self.data['x_central_test'][np.random.randint(self.n_s)]
			,label='test data',alpha=0.5)

		ax[1].legend(frameon = False)
		ax[0].set_title('%i data points'%self.input_shape[0])
		ax[1].set_xlabel("Data amplitude")
		ax[1].set_ylabel('Counts')
		ax[1].set_xlim(0,6)
		
		plt.savefig(f'{self.figuredir}data_visualization_hist_{self.modelversion}.png')
		if show: plt.show()
		plt.close()

	def plot_derivatives(self, show=False):
		""" 
		Plot the upper and lower perturbed data as data amplitude

		VARIABLES
		#______________________________________________________________
		show 					bool	whether or not plt.show() is called
		data 					dict 	dict containing training and test data

		"""

		fig, ax = plt.subplots(2, 2, figsize = (15, 10))
		# plt.subplots_adjust(wspace = 0, hspace = 0.1)
		training_index = np.random.randint(self.n_train * self.n_p)
		
		xp_theta1, xp_theta2 = self.data['x_p'][training_index]

		# Theta 1 upper simulation
		ax[0, 0].plot(self.data['x_p'][training_index, 0], label = "upper training data"
			, color = 'C0', linestyle='dashed')
		# Theta 2 upper simulation
		ax[0, 1].plot(self.data['x_p'][training_index, 1], label = "upper training data"
			, color = 'C0', linestyle='dashed')

		# Theta 1 lower simulation
		ax[0, 0].plot(self.data['x_m'][training_index, 0], label = "lower training data"
					, color = 'C0')
		# Theta 2 lower simulation
		ax[0, 1].plot(self.data['x_m'][training_index, 1], label = "lower training data"
					, color = 'C0')

		test_index = np.random.randint(self.n_p)
		# Theta1  upper and lower
		ax[0, 0].plot(self.data['x_m_test'][test_index, 0], label = "lower test data"
					, color = 'C1')
		ax[0, 0].plot(self.data['x_p_test'][test_index, 0], label = "upper test data"
					, color = 'C1', linestyle='dashed')

		# Theta2  upper and lower
		ax[0, 1].plot(self.data['x_m_test'][test_index, 1], label = "lower test data"
					, color = 'C1')
		ax[0, 1].plot(self.data['x_p_test'][test_index, 1], label = "upper test data"
					, color = 'C1', linestyle='dashed')


		ax[0, 0].legend(frameon = False)
		ax[0, 1].legend(frameon = False)
		ax[0, 0].set_xticks([])
		ax[0, 1].set_xticks([])
		ax[0, 0].set_ylabel("Data amplitude")
		ax[0, 1].set_ylabel("Data amplitude")
		ax[0, 0].set_title("Theta 1 (the mean)")
		ax[0, 1].set_title("Theta 2 (the std)")

		for i in range(2):
			for j in range(2):
				ax[i, j].set_xlim(0,9)
		fig.suptitle('Showing only first 10 datapoints out of %i'%self.data['x_m'].shape[1])

		# Theta 1
		ax[1, 0].axhline(xmin = 0., xmax = 1., y = 0., linestyle = 'dashed'
					, color = 'black')
		ax[1, 0].plot(self.data['x_p'][training_index, 0] - self.data['x_m'][training_index, 0]
					, color = 'C0',alpha=0.5)
		ax[1, 0].plot(self.data['x_p_test'][test_index, 0] - self.data['x_m_test'][test_index, 0]
					, color = 'C1',alpha=0.5)
		ax[1, 0].set_xticks([])
		ax[1, 0].set_ylabel("Difference between derivative data amplitudes")

		# Theta 2
		ax[1, 1].axhline(xmin = 0., xmax = 1., y = 0., linestyle = 'dashed'
					, color = 'black')
		ax[1, 1].plot(self.data['x_p'][training_index, 1] - self.data['x_m'][training_index, 1]
					, color = 'C0',alpha=0.5)
		ax[1, 1].plot(self.data['x_p_test'][test_index, 1] - self.data['x_m_test'][test_index, 1]
					, color = 'C1',alpha=0.5)
		ax[1, 1].set_xticks([])
		ax[1, 1].set_ylabel("Difference between derivative data amplitudes");

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
		
		'''
		# Derivative wrt to theta1				   theta1 is column 0
		ax[3].plot(epochs, np.array(n.history["dμdθ"])[:,0].flatten()
			, color = 'C0', label='theta1',alpha=0.5)
		# Derivative wrt to theta2				   theta1 is column 1
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

def generate_data(θ, train = None):
	'''Train is whether the upper and lower derivatives are calculated '''
	if train is not None:
		holder = np.zeros([θ.shape[0]] + [θ.shape[1]] + input_shape)
		for i in range(θ.shape[1]):
			params = np.copy(θ)
			params[:, i] += train[i]
			holder[:, i, :] = np.moveaxis(np.random.normal(params[:, 0], np.sqrt(params[:, 1]), input_shape + [θ.shape[0]]), -1, 0)
		return holder																							  
	else:
		return np.moveaxis(np.random.normal(θ[:, 0], np.sqrt(θ[:, 1]), input_shape + [θ.shape[0]]), -1, 0)


# ALL PARAMETERS
#_______________________________________________________
# input_shape = [10]
theta_fid = np.array([0.,1.]) # mean and variance
# delta_theta = np.array([0.1,0.1]) # perturbation values
n_s = 1000 # number of simulations
n_train = 1 # splits, for if it doesnt fit into memory
# use less simulations for numerical derivative
# derivative_fraction = 0.20
# eta = 1e-5
num_epochs = 10000
keep_rate = 0.6
verbose = 0
# hidden_layers = [256,256,256]
initial_version = 1

run_three_times = True


def main():
	parameters = {
		'verbose': False,
		'number of summaries': 2,
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
	# # Train network
	nholder1.train_network(n)
	# # Plot the output
	nholder1.plot_variables(n,show=False)


i = 0
for input_shape in [[10],[100],[1000]]:
	# for theta_fid in [ np.array([0., 1.]), np.array([3., 1.]), np.array([3., 3.])]:
	for delta_theta in [ np.array([0.1,0.1]), np.array([0.1, 0.05])]:
		for derivative_fraction in [0.20, 0.05]:
			for eta in [1e-4, 1e-5, 1e-6]:
				for hidden_layers in [ [256,256,256], [128,128], [256,128] ]:
					initial_version += 1
					i += 1
					print ('Now doing i =', i, 'out of i = ', 108)
					
					if run_three_times:
						# Run every network 3 times
						for init in range(3):
							version = initial_version + init/10
							print (version)
							print ('Running version: ', version)
							main()
					else:
						version = initial_version
						print ('Running just one network, version: ', version)
						main()
