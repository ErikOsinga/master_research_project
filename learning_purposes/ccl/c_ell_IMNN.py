import sys
import numpy as np
import matplotlib.pyplot as plt
# for running notebooks, plotting inline
# %pylab inline

import tensorflow as tf
import keras
# change to the path where the IMNN git clone is located
sys.path.insert(-1,'../../../information_maximiser')
import IMNN # make sure the path to the IMNN is given
import tqdm
sys.path.insert(-1,'../../') # change to path where utils_mrp is located
import utils_mrp
# For making corner plots of the posterior
import corner # Reference: https://github.com/dfm/corner.py/

import pyccl as ccl # for generating weak lensing cross power spectra

"""
Summarizing a weak lensing data vector. Generated with the pyccl module.
The goal is to predict Omega_m and Sigma_8 for a Euclid-like survey.

We assume a redshift distribution given by
    z^alpha * exp(z/z0)^beta
    with alpha=1.3, beta = 1.5 and z0 = 0.65

And assume the photo-z error is Gaussian with a bias is 0.05(1+z)

The weak lensing data vectors are of shape 
    (nbins*nbins/2 + nbins, len(ell))
    i.e., the number of cross/auto correlation spectra and the number of
    ell that is simulated

We flatten these data vectors and feed them to the IMNN, producing 2 summary statistics

"""


class nholder(object):
    """
    Class to hold and store all parameters for an IMNN 
    """
    def __init__(self, input_shape, generate_data, theta_fid, delta_theta, n_s, n_train, 
        derivative_fraction, eta, parameters, num_epochs, keep_rate, verbose, 
        version, flatten):
        """
        INITIALIZE PARAMETERS
        #______________________________________________________________
        input_shape                 list    shape of the data that is generated
        generate_data               list    function to generate the data
        theta_fid                   list    fiducial parameter values
        delta_theta                 list    perturbation values for fiducial param
        n_s                         int     number of simulations
        n_train                     int     number of splits, to make more simulations
        derivative_fraction         float   fraction of n_s to use for derivatives
        eta                         float   learning rate
        parameters                  dict    dict of parameters to feed IMNN
        num_epochs                  int     amount of epochs
        keep_rate                   float   (1-dropout rate), amount of nodes to keep every batch
        verbose                     int     TODO
        version                     float   version ID of this particular network
        flatten                     bool    whether to flatten the train/test data
        """

        tf.reset_default_graph()

        self.unflattened_shape = input_shape
        if flatten:
            self.input_shape = [int(np.prod(input_shape))] # must be a list of an integer
        else:
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
        self.flatten = flatten

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

        data                    dict    dict containing training and test data
        der_den                 array   array containing the derivative denominator

        """

        # Number of upper and lower simulations
        n_p = int(self.n_s * self.derivative_fraction)

        # set a seed to surpress the sample variance
        seed = np.random.randint(1e6)
        np.random.seed(seed)
        # Perturb lower 
        t_m = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_p)])
                    ,train = -self.delta_theta, flatten = self.flatten)
        np.random.seed(seed)
        # Perturb higher 
        t_p = self.generate_data(np.array([theta_fid for i in 
                    range(self.n_train * self.n_p)])
                    ,train = self.delta_theta, flatten = self.flatten)
        np.random.seed()

        t = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_s)])
                    ,train = None, flatten = self.flatten)
        np.random.seed()

        der_den = 1. / (2. * self.delta_theta)

        data = {"x_central": t, "x_m": t_m, "x_p":t_p}

        # Repeat the same story to generate training data
        seed = np.random.randint(1e6)
        np.random.seed(seed)
        # Perturb lower 
        tt_m = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_p)])
                    , train = -self.delta_theta, flatten = self.flatten)
        np.random.seed(seed)
        # Perturb higher 
        tt_p = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_p)])
                    , train = self.delta_theta, flatten = self.flatten)
        np.random.seed()

        tt = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_s)])
                    , train = None, flatten = self.flatten)
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
        show                    bool    whether or not plt.show() is called

        """

        fig, ax = plt.subplots(figsize = (10, 6))

        # plot one random row from the simulated train data 
        if self.flatten:
            print ('Plotting data... reshaping the flattened data to %s'%str(input_shape))
            temp = self.data['x_central'][np.random.randint(self.n_train * self.n_s)].reshape(input_shape)
            x, y = temp.T[:,0]
        else:
            print ('Plotting data...')
            temp = self.data['x_central'][np.random.randint(self.n_train * self.n_s)].reshape(55,4900)
            Cl = temp[0] # plot the (0,0) autocorrelation bin

        ax.loglog(ell, ell*(ell+1)*Cl)
        ax.set_title('Training data')
        ax.set_xlabel('$\ell$')
        ax.set_ylabel('$\ell(\ell+1) C_\ell$')
        
        # plot one random row from the simulated test data 
        if self.flatten:
            temp = self.data['x_central_test'][np.random.randint(self.n_s)].reshape(input_shape)
            x, y = temp.T[:,0]
        else:
            x, y = self.data['x_central_test'][np.random.randint(self.n_s)].T[:,0]
        ax[1].plot(x,y,'x',label='data')
        ax[1].axis('equal')
        ax[1].set_title('Test data')
        ax[1].plot(*theta_fid,'x',color='r',label='Fiducial mean')
        ax[1].set_xlim(theta_fid[1]-3,theta_fid[1]+3)

        plt.legend()

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
        show                    bool    whether or not plt.show() is called

        """

        fig, ax = plt.subplots(3, 2, figsize = (15, 10))
        # plt.subplots_adjust(wspace = 0, hspace = 0.1)
        training_index = np.random.randint(self.n_train * self.n_p)
        
        if self.flatten:
            print ('Plotting derivatives... reshaping the flattened data to %s'%str(input_shape))
            temp = self.data['x_p'][training_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            x, y = self.data['x_p'][training_index].T[:,0]
        
        # x, y have shape (10,2) since they are the x and y of the  
        # upper training image for both params
        labels =['$θ_1$','$θ_2$']

        # we loop over them in this plot only, to assign labels
        for i in range(x.shape[1]):
            ax[0, 0].plot(x[:,i],y[:,i],'x',label=labels[i])
        ax[0, 0].set_title('Upper training image')
        ax[0, 0].set_xlim(-3,3)
        ax[0, 0].set_ylim(-3,3)
        ax[0, 0].legend(frameon=False)

        if self.flatten:
            temp = self.data['x_m'][training_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            x, y = self.data['x_m'][training_index].T[:,0]

        ax[1, 0].plot(x, y, 'x')
        ax[1, 0].set_title('Lower training image')
        ax[1, 0].set_xlim(-3,3)
        ax[1, 0].set_ylim(-3,3)

        if self.flatten:
            temp = self.data["x_m"][training_index].reshape(len(theta_fid),*input_shape)
            xm, ym = temp.T[:,0]

            temp = self.data["x_p"][training_index].reshape(len(theta_fid),*input_shape)
            xp, yp = temp.T[:,0]
        else:
            xm, ym = self.data["x_m"][training_index].T[:,0]
            xp, yp = self.data["x_p"][training_index].T[:,0]

        ax[2, 0].plot(xp-xm,yp-ym,'x')
        ax[2, 0].set_title('Difference between upper and lower training images');
        ax[2, 0].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')

        test_index = np.random.randint(self.n_p)

        if self.flatten:
            temp = self.data['x_p_test'][test_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            x, y = self.data['x_p_test'][test_index].T[:,0]
        
        ax[0, 1].plot(x, y, 'x')
        ax[0, 1].set_title('Upper test image')

        if self.flatten:
            temp = self.data['x_m_test'][training_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            x, y = self.data['x_m_test'][training_index].T[:,0]

        ax[1, 1].plot(x, y, 'x')
        ax[1, 1].set_title('Lower test image')

        if self.flatten:
            temp = self.data["x_m_test"][test_index].reshape(len(theta_fid),*input_shape)
            xm, ym = temp.T[:,0]

            temp = self.data["x_p_test"][test_index].reshape(len(theta_fid),*input_shape)
            xp, yp = temp.T[:,0]
        else:
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
        data                    dict    dict containing training and test data
        
        RETURNS
        #_____________________________________________________________
        n                       class   IMNN class as defined in IMNN.py

        """
        n = IMNN.IMNN(parameters=self.parameters)
        tf.reset_default_graph()
        n.setup(η = eta)
        
        return n

    def train_network(self, n, to_continue=False):
        """ 
        Train the created network with the given data and parameters
        Saves the history and the determinant of the final fisher info

        INPUTS
        #______________________________________________________________
        n                       class   IMNN class as defined in IMNN.py
        
        """

        n.train(num_epochs = self.num_epochs, n_train = self.n_train
            , keep_rate = self.keep_rate, data = self.data, history = True
            , to_continue= to_continue)

        # save the network history to a file
        utils_mrp.save_history(self, n)

        # save the det(Final Fisher info) in the modelsettings.csv file
        utils_mrp.save_final_fisher_info(self, n)

    def plot_variables(self, n, show=False):
        """ 
        Plot variables vs epochs

        INPUTS
        #______________________________________________________________
        n                       class   IMNN class as defined in IMNN.py
        
        """
        fig, ax = plt.subplots(3, 1, sharex = True, figsize = (8, 14))
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
        # Derivative wrt to theta1                 theta1 is column 0
        ax[3].plot(epochs, np.array(n.history["dμdθ"])[:,0].flatten()
            , color = 'C0', label='theta1',alpha=0.5)
        # Derivative wrt to theta2                 theta1 is column 1
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

    def ABC(self, n, real_data, prior, draws, show=False, epsilon=None, oneD='both'):
        """ 
        Perform ABC
        Only a uniform prior is implemented at the moment.

        INPUTS
        #______________________________________________________________
        n                       class   IMNN class as defined in IMNN.py
        real_data               array   array containing true data
        prior                   list    lower and upper bounds for uniform priors
        draws                   int     amount of draws from prior
        show                    bool    whether or not plt.show() is called
        oneD                    bool    whether to plot one dimensional posteriors
                                        or two dimensional with the corner module

        RETURNS
        #_____________________________________________________________
        theta                   list    sampled parameter values            
        accept_indices          list    indices of theta that satisfy (ro < epsilon)
    
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
        if epsilon is None: epsilon = np.linalg.norm(summary)/8. # chosen quite arbitrarily
        accept_indices = np.argwhere(ro < epsilon)[:, 0]
        reject_indices = np.argwhere(ro >= epsilon)[:, 0]

        print ('Epsilon is chosen to be %.2f'%epsilon)

        # plot output samples and histogram of the accepted samples in 1D
        def plot_samples_oneD():
            fig, ax = plt.subplots(2, 2, sharex = 'col', figsize = (10, 10))
            plt.subplots_adjust(hspace = 0)
            theta1 = theta[:,0]
            theta2 = theta[:,1]

            ax[0, 0].set_title('Epsilon is chosen to be %.2f'%epsilon)
            ax[0, 0].scatter(theta1[reject_indices], s[reject_indices, 0], s = 1, alpha = 0.1)
            ax[0, 0].scatter(theta1[accept_indices] , s[accept_indices, 0], s = 1)
            ax[0, 0].plot(prior[0], [summary[0], summary[0]], color = 'black', linestyle = 'dashed')
            ax[0, 0].set_ylabel('Network output', labelpad = 0)
            ax[0, 0].set_xlim(prior[0])
            ax[1, 0].hist(theta1[accept_indices], bins=np.linspace(*prior[0], 100)
                , histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
            ax[1, 0].set_xlabel('$\\theta_1$ (mean1)')
            ax[1, 0].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
            ax[1, 0].set_yticks([])

            ax[0, 1].scatter(theta2[reject_indices], s[reject_indices, 0], s = 1, alpha = 0.1)
            ax[0, 1].scatter(theta2[accept_indices] , s[accept_indices, 0], s = 1)
            ax[0, 1].plot(prior[1], [summary[0], summary[0]], color = 'black', linestyle = 'dashed')
            ax[0, 1].set_ylabel('Network output', labelpad = 0)
            ax[0, 1].set_xlim(prior[1])
            ax[1, 1].hist(theta2[accept_indices], np.linspace(*prior[1], 100)
                , histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
            ax[1, 1].set_xlabel('$\\theta_2$ (mean2)')
            ax[1, 1].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
            ax[1, 1].set_yticks([])

            fig.suptitle("Only showing 1st network output summary out of %i \n Full network output on real data: %s"%(s.shape[1],str(summary)))

            plt.savefig(f'{self.figuredir}ABC_{self.modelversion}_1D.png')
            if show: plt.show()
            plt.close()

        # plot approximate posterior of the accepted samples in 2D
        def plot_samples_twoD():
            hist_kwargs = {} # add kwargs to give to matplotlib hist funct
            fig, ax = plt.subplots(2, 2, figsize = (10, 10))
            fig = corner.corner(theta[accept_indices], bins=100, fig=fig, truths = theta_fid
                , labels=['$\\theta_1$ (mean1)','$\\theta_2$ (mean2)']
                , plot_contours=True, range=prior, hist_kwargs=hist_kwargs)
            fig.suptitle('Approximate posterior after ABC for %i draws'%draws)
            plt.savefig(f'{self.figuredir}ABC_{self.modelversion}_2D.png')
            if show: plt.show()
            plt.close()

        if oneD == 'both':
            plot_samples_oneD()
            plot_samples_twoD()
        elif type(oneD) == bool:
            if oneD: 
                plot_samples_oneD()
            else: 
                plot_samples_twoD()
        else: 
            raise ValueError('Allowed values for oneD are "both", True or False')

        # There can be a lot of theta draws which are unconstrained by the network
        # because no similar structures were seen in the data, which is indicative of
        # using too small of a small training set

        return theta, accept_indices

    def PMC_ABC(self, n, real_data, prior, draws, num_keep, criterion = 0.1, show=False, oneD='both'):
        """ 
        Perform PMC ABC, which is a way of reducing the number of draws
        The inputs work in a very similar way to the ABC function above. If we 
        want 1000 samples from the approximate distribution at the end of the
        PMC we need to set num_keep = 1000. The initial random draw is initialised
        with num_draws, the larger this is the better proposal distr will be on
        the 1st iteration.


        Only a uniform prior is implemented at the moment.

        INPUTS
        #______________________________________________________________
        n                       class   IMNN class as defined in IMNN.py
        real_data               array   array containing true data
        prior                   list    lower and upper bounds for uniform priors
        draws                   int     number of initial draws from the prior
        num_keep                int     number of samples in the approximate posterior
        criterion               float   ratio of number of draws wanted over number of draws needed
        show                    bool    whether or not plt.show() is called
        oneD                    bool    whether to plot one dimensional posteriors
                                        or two dimensional with the corner module       

        RETURNS
        #_____________________________________________________________
        theta                   list    sampled parameter values in the approximate posterior           
        all_epsilon             list    progression of epsilon during PMC       
    
        """

        # W = weighting of samples, total_draws = total num draws so far
        theta_, summary_, ro_, s_, W, total_draws, F, all_epsilon = n.PMC(real_data = real_data
            , prior = prior, num_draws = draws, num_keep = num_keep
            , generate_simulation = self.generate_data, criterion = criterion
            , at_once = True, samples = None, data = self.data)

        # plot output samples and histogram of approximate posterior
        def plot_samples_oneD():

            theta1 = theta_[:,0]
            theta2 = theta_[:,1]
            
            fig, ax = plt.subplots(2, 2, sharex = 'col', figsize = (10, 10))
            plt.subplots_adjust(hspace = 0)
            ax[0,0].scatter(theta1 , s_[:,0], s = 1)
            ax[0,0].plot(prior[0], [summary_[0], summary_[0]], color = 'black', linestyle = 'dashed')
            ax[0,0].set_ylabel('Network output', labelpad = 0)
            ax[0,0].set_ylim([np.min(s_[:,0]), np.max(s_[:,0])])
            ax[0,0].set_xlim(prior[0])
            ax[1,0].hist(theta1, bins= np.linspace(*prior[0], 100), histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
            ax[1,0].set_xlabel('$\\theta_1$ (mean1)')
            ax[1,0].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
            ax[1,0].set_yticks([])

            ax[0,1].scatter(theta2 , s_[:,0], s = 1)
            ax[0,1].plot(prior[1], [summary_[0], summary_[0]], color = 'black', linestyle = 'dashed')
            ax[0,1].set_ylabel('Network output', labelpad = 0)
            ax[0,1].set_xlim(prior[1])
            ax[0,1].set_ylim([np.min(s_[:,0]), np.max(s_[:,0])])
            ax[1,1].hist(theta2, bins = np.linspace(*prior[1], 100), histtype = u'step', density = True, linewidth = 1.5, color = '#9467bd');
            ax[1,1].set_xlabel('$\\theta_2$ (mean2)')
            ax[1,1].set_ylabel('$\\mathcal{P}(\\theta|{\\bf d})$')
            ax[1,1].set_yticks([])

            fig.suptitle("Only showing 1st network output summary out of %i \n Full network output on real data: %s"%(s_.shape[1],str(summary_)))

            plt.savefig(f'{self.figuredir}PMC_ABC_{self.modelversion}_1D.png')
            if show: plt.show()
            plt.close()

        # plot output samples and histogram of the accepted samples
        def plot_samples_twoD():
            hist_kwargs = {} # add kwargs to give to matplotlib hist funct
            fig, ax = plt.subplots(2, 2, figsize = (10, 10))
            fig = corner.corner(theta_, fig=fig, truths = theta_fid
                , labels=['$\\theta_1$ (mean1)','$\\theta_2$ (mean2)']
                , plot_contours=True, range=prior, hist_kwargs=hist_kwargs)
            fig.suptitle("Approximate posterior after PMC ABC, num_keep = %i"%num_keep)
            
            plt.savefig(f'{self.figuredir}PMC_ABC_{self.modelversion}_2D.png')
            if show: plt.show()
            plt.close()

        # Plot epsilon vs iterations
        def plot_epsilon():
            fig, ax = plt.subplots()
            ax.plot(all_epsilon,color='k',label='$\epsilon$ values')
            plt.xlabel('Iteration')
            plt.ylabel('$\epsilon$')
            plt.legend()
            
            plt.savefig(f'{self.figuredir}PMC_ABC_{self.modelversion}_epsilon.png')
            if show: plt.show()
            plt.close()

        if oneD == 'both':
            plot_samples_oneD()
            plot_samples_twoD()
        elif type(oneD) == bool:
            if oneD:
                plot_samples_oneD()
            else:
                plot_samples_twoD()
        else: 
            raise ValueError('Allowed values for oneD are "both", True or False')

        plot_epsilon()

        return theta_, all_epsilon


# #######################################
# PYCCL PARAMETERS AND GENERATE DATA FUNCTION
nz = 1000 # resolution in the redshift
z = np.linspace(0.0, 2.0, nz) # samples of z
ell=np.arange(100,5000) # samples of ell

"""
Assume a redshift distribution given by
    z^alpha * exp(z/z0)^beta
    with alpha=1.3, beta = 1.5 and z0 = 0.65
"""
dNdz_true = ccl.dNdzSmail(alpha = 1.3, beta = 1.5, z0=0.65)
# Assumes photo-z error is Gaussian with a bias is 0.05(1+z)
pz = ccl.PhotoZGaussian(sigma_z0=0.05)

def euclid_ccl(Omega_c, sigma8):
    """
    Generate C_ell as function of ell for a given Omega_c and Sigma8

    Inputs
        Omega_c -- float: CDM density 
        Sigma_8 -- float: sigma_8

    Assumed global variables
        z -- np.array: samples of z
        ell -- np.array: samples of ell
        dNdz_true -- ccl.dNdzSmail: dNdz distribution
        pz -- ccl.PhotoZGaussian: PhotoZ error
        bins -- the amount of tomographic redshift bins

    Outputs
        Cls -- np.array, shape (nbins*(nbins-1)/2 + nbins, len(ell)): 
                Cross/Auto correlation shear spectra for the tomographic bins
        dNdzs -- np.array, shape (nbins,len(z):
                dNdz per redshift bin, for all redshifts

    """

    cosmo_fid = ccl.Cosmology(Omega_c=Omega_c, Omega_b=0.045, h=0.71, sigma8=sigma8, n_s=0.963)

    dNdzs = np.zeros((10, z.size))
    shears = []
    
    for i in range(10):
        # edges of 10 equal width redshift bins, between 0 and 2
        zmin, zmax = i*0.2, (i+1)*.2
        # generate dNdz per bin
        dNdzs[i,:] = ccl.dNdz_tomog(z=z, zmin=zmin, zmax=zmax, pz_func=pz, dNdz_func = dNdz_true)
        # calculate the shear per bin
        gal_shapes = ccl.WeakLensingTracer(cosmo_fid, dndz=(z, dNdzs[i,:]))
        shears.append(gal_shapes)
        
    # calculate 10*9/2 (cross corr) + 10 (autocorr) spectra from the shears
    Cls = []
    for i in range(10):
        for j in range(0,i+1):
            Cls.append(ccl.angular_cl(cosmo_fid, shears[i], shears[j], ell))
     
    return np.array(Cls), dNdzs

def generate_data(θ, train=None):
    """
    Holder function for the generation of the Cls
    
    θ = vector of lists of [Omega_c, Sigma8]'s to produce a Cl for

    Returns the weak lensing data vector flattened to use as input for the IMNN
            shape (num_simulations=len(θ), length of Cl vector)
    """

    # Think about how to generate multiple at once
    # Omega_c, sigma8 = θ

    all_Cls = []
    for Omega_c, sigma8 in θ:
        Cls, dNdzs = euclid_ccl(Omega_c, sigma8)
        all_Cls.append(Cls.flatten()) # flatten the Cl data

    return all_Cls



t = self.generate_data(np.array([self.theta_fid for i in 
                range(self.n_train * self.n_s)])
                ,train = None, flatten = self.flatten)

Cls, dNdzs = euclid_ccl(0.27, 0.82)

data = generate_data(0.27,0.82)






# #######################################
# ALL PARAMETERS FOR IMNN
#_______________________________________________________
# The input shape is a 1D vector of length 55*4900
input_shape = [55*4900] 

theta_fid = np.array([0.27, 0.82]) # Omega_m and Sigma_8 

delta_theta = np.array([0.05,0.05]) # perturbation values
n_s = 1000 # number of simulations
n_train = 1 # splits, for if it doesnt fit into memory
# use less simulations for numerical derivative
derivative_fraction = 0.05
eta = 1e-5 # learning rate
num_epochs = int(10e3)
keep_rate = 0.5 # 1 minus the dropout
verbose = 0

# MLP
hidden_layers = [256,256,256]

flatten = False # data is already flat, don't have to flatten it again

initial_version = 1

version = initial_version

parameters = {
    'verbose': False,
    'number of summaries': 2,
    'calculate MLE': True,
    'prebuild': True, # allow IMNN to build network
    'save file': "Models/data/model"+str(version),
    'wv': 0., # the variance with which to initialise the weights
    'bb': 0.1, # the constant value with which to initialise the biases
    'activation': tf.nn.leaky_relu,
    'α': 0.01, # negative gradient parameter, only needed for certain activation func
    'hidden layers': hidden_layers,
    'flatten': flatten
}
#______________________________________________________

# Network holder, creates the data as well
nholder1 = nholder(input_shape, generate_data, theta_fid, delta_theta, n_s,
        n_train, derivative_fraction, eta, parameters, num_epochs, keep_rate,
        verbose, version, flatten)

# # IMNN network
n = nholder1.create_network()
# # Plot data
nholder1.plot_data(show=False)
# # plot derivatives
nholder1.plot_derivatives(show=False)
# # Train network
nholder1.train_network(n)
# # Plot the output
nholder1.plot_variables(n,show=True)

# Generate actual data with mean = [1., 2.]
real_data = generate_data(np.array([theta_fid]), train = None, flatten=flatten)
print ('Means of the "real data" generated at fiducial parameters')
print (np.mean(real_data.reshape(input_shape),axis=0))

# # Perform ABC
prior = [[-2, 3], [0,4]] # [ [prior1], [prior2], etc.. ]
draws = 1000000
print ('Running ABC for %i draws'%draws)
nholder1.ABC(n, real_data, prior, draws, show=True, epsilon=None,oneD='both')

# # Perform PMC
num_keep = int(1e3)
inital_draws = int(1e4)

theta_, all_epsilon = nholder1.PMC_ABC(n, real_data, prior, inital_draws
        , num_keep, criterion = 0.1, show=True,oneD='both')



"""
nholder1.train_network(n,to_continue=True)
nholder1.plot_variables(n,show=True)
"""


























# plot the 55 spectra
fig = plt.figure(figsize=(20,20))
counter = 0
for i in range(10):
        for j in range(0,i+1):
            ax = plt.subplot2grid((10,10), (i,j))
            ax.loglog(ells, ells*(ells+1)*Cls[counter])
            counter += 1

plt.show()


# plot the dNdzs distribution
plt.figure(figsize=(10,10))
for i in range(10):
    plt.plot(z, dNdzs[i,:], lw = 3)

plt.show()