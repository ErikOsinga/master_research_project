import sys
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
# change to the path where the IMNN git clone is located
# new version of IMNN by Tom
sys.path.append('../../../../IMNNv2/IMNN/')
import IMNN.IMNN as IMNN # make sure the path to the IMNN is given
import IMNN.ABC.ABC as ABC
import IMNN.ABC.priors as priors

sys.path.append('../../../cosmosis_wrappers/') # change to correct path
from generate_cells_cosmosis import generate_cells, load_cells


import tqdm
sys.path.append('../../../') # change to path where utils_mrp is located
import utils_mrp_v2 as utils_mrp
import set_plot_sizes # set font sizes

# For making corner plots of the posterior
import corner # Reference: https://github.com/dfm/corner.py/


# for some reason pyccl doesnt work on eemmeer
import pyccl as ccl # for generating weak lensing cross power spectra

"""
Summarizing a weak lensing data vector. Generated with the cosmosis module.

Make sure that this command is executed before or cosmosis will not be found
source /net/reusel/data1/osinga/cosmosis_installation/cosmosis/stup-my-cosmosis 

The goal is to predict Omega_M and Sigma_8 for a Euclid-like survey.

We assume a redshift distribution given by
    z^alpha * exp(z/z0)^beta
    with alpha=1.3, beta = 1.5 and z0 = 0.65

And assume the photo-z error is Gaussian with a bias is 0.05(1+z)

The weak lensing data vectors are of shape 
    (nbin*nbin/2 + nbin, len(ell))
    i.e., the number of cross/auto correlation spectra and the number of
    ell that is simulated

We just use 1 tomographic bin, so we will have 1 Cl auto spectrum (0,0)
We sample it at 100 logarithmically equal spaced points, Thus our weak lensing
data vector will be of shape (1,100)

We flatten these (already flat) data vectors and feed them to the IMNN,
producing 2 summary statistics

!!!
Therefore, flatten = False in the nholder object, 
because the data does not have to be flattened again
!!!

"""
tf.reset_default_graph()


class nholder(object):
    """
    Class to hold and store all parameters for an IMNN 
    """
    def __init__(self, input_shape, generate_data, theta_fid, delta_theta, n_s, n_train, 
        derivative_fraction, n_train_val, derivative_fraction_val, eta, parameters,
        num_epochs, keep_rate, verbose, dtype, training_dictionary, validation_dictionary, 
        network, version, fromdisk, noiseless_deriv, flatten, rescale, 
        load_network=False):
        """
        INITIALIZE PARAMETERS
        #______________________________________________________________
        input_shape                  list    shape of the data that is generated
        generate_data                list    function to generate the data
        theta_fid                    list    fiducial parameter values
        delta_theta                  list    perturbation values for fiducial param
        n_s                          int     number of simulations used to calculate cov
                                             SAME FOR TRAIN AND TEST DATA 
        n_train                      int     number of splits, to make more simulations
        n_p                          int     number of simulations for derivatives
        n_train_val                  int     number of splits, to make more simulations
        n_p_val                      int     number of simulations for test data derivatives
        derivative_fraction          float   fraction of n_s to use for derivatives
        derivative_fraction_val      float   fraction of n_s to use for test derivatives
        eta                          float   learning rate
        parameters                   dict    dict of parameters to feed IMNN
        num_epochs                   int     amount of epochs
        keep_rate                    float   (1-dropout rate), fraction of nodes to keep every batch
        verbose                      int     TODO
        network                      func    function that builds the network
        training_dictionary          dict    training dictionary, contains variable values 
                                             for training 
        validation_dictionary        dict    test dictionary, contains variable values
        version                      float   version ID of this particular network
        fromdisk                     bool    whether to read train/test from disk
        noiseless_deriv              bool    whether to add noise to derivatives or not
        flatten                      bool    whether to flatten the train/test data
        rescale                      bool    whether to take -1*log() of the data
        load_network                 bool    whether to load a previous network version

        """

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
        self.derivative_fraction = derivative_fraction
        self.n_p = int(n_s * derivative_fraction)
        self.n_train_val = n_train_val
        self.derivative_fraction_val = derivative_fraction_val
        self.n_p_val = int(n_s * derivative_fraction_val)
        self.eta = eta
        self.num_epochs = num_epochs
        self.keep_rate = keep_rate
        self.verbose = verbose
        self.rescaled = False # is set to true when self.rescale_data is called
        self.noiseless_deriv = noiseless_deriv
        self.flatten = flatten
        self.dtype = dtype
        self.training_dictionary = training_dictionary
        self.validation_dictionary = validation_dictionary
        self.network = network
        self.rescale = rescale
        self.load_network = load_network

        print (f"Network version {version}")

        if fromdisk:
            self.data = self.load_data_from_disk()
        else:
            self.data = self.create_data()

        # # Theoretical Cl with no noise, looks like upper/lower but with +/- 0
        self.Cl_noiseless = generate_data([theta_fid], train=[0], noiseless_deriv=True)[0]
        # shape (1,100) for 1 tomographic bin

        # Covariance from theoretical Cl
        # Calculate the covariance for every ell
        self.covariance = calculate_covariance(self.Cl_noiseless)
        
        self.Cl_noiseless = self.Cl_noiseless[0] # remove redundant dimension afterwards

        # generate cross power spectra Cl
        # Cls, dNdzs = euclid_ccl(Omega_M)

        # Make parameters dictionary of params that are always the same or defined
        # by other parameters   
        self.parameters = { 'number of simulations': self.n_s,
                            'fiducial': list(self.theta_fid),
                            'number of derivative simulations': self.n_p,
                            'input shape': self.input_shape,
                            'dtype': self.dtype,
                            'noiseless deriv': self.noiseless_deriv
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

        if not self.load_network:
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

        print (f'Using {self.n_s} simulations for the training data to estimate cov')
        print (f'Using {self.n_p} simulations for the upper/lower  training data')
        print (f'Number of splits, to increase number simulations: {self.n_train}')
        print (f'Adding noise to the derivative: {np.invert(self.noiseless_deriv)}')

        # Number of upper and lower simulations
        n_p = int(self.n_s * self.derivative_fraction)

        # set a seed to surpress the sample variance (EVEN FOR CENTRAL SIMULATIONS)
        seed = np.random.randint(1e6) 
        # We should double-check to see if the sample variance if being surpressed

        # Perturb lower 
        np.random.seed(seed)
        t_m = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_p)])
                    ,train = -self.delta_theta, flatten = self.flatten
                    ,noiseless_deriv = self.noiseless_deriv) 
        # Perturb higher 
        np.random.seed(seed)
        t_p = self.generate_data(np.array([theta_fid for i in 
                    range(self.n_train * self.n_p)])
                    ,train = self.delta_theta, flatten = self.flatten
                    , noiseless_deriv = self.noiseless_deriv)

        # Central
        np.random.seed(seed)
        t = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_s)])
                    ,train = None, flatten = self.flatten)


        if self.rescale: # take -1*log
            print ("Rescaling data by taking -1*log() ")
            print ("Replacing NaN with 0 ")
            self.rescaled = True
            t_m = -1 * np.log(t_m) 
            # replace NaNs with 0
            np.nan_to_num(t_m,copy=False)
            
            t_p = -1 * np.log(t_p)
            np.nan_to_num(t_p,copy=False)
            
            t = -1 * np.log(t)
            np.nan_to_num(t,copy=False)

        # derivative data
        t_d = (t_p - t_m) / (2. * self.delta_theta)

        # Save in a dict that the network takes
        data = {"data": t, "data_d": t_d}
        # for plotting purposes we save the upper/lower separately as well
        data["x_m"], data["x_p"] = t_m, t_p 

        # Repeat the same story to generate test data
        print ('\n')
        print (f'Using {self.n_s} simulations for the test data to estimate cov')
        print (f'Using {self.n_p_val} simulations for the upper/lower test data')
        print (f'Number of splits, to increase number simulations: {self.n_train_val}')
        print (f'Adding noise to the derivative: {np.invert(self.noiseless_deriv)}')
        print ('\n')

        seed = np.random.randint(1e6)
        # Perturb lower 
        np.random.seed(seed)
        tt_m = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_p)])
                    , train = -self.delta_theta, flatten = self.flatten
                    , noiseless_deriv = self.noiseless_deriv)
        # Perturb higher 
        np.random.seed(seed)
        tt_p = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_p)])
                    , train = self.delta_theta, flatten = self.flatten
                    , noiseless_deriv = self.noiseless_deriv)
        # Central sim
        np.random.seed(seed)
        tt = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_s)])
                    , train = None, flatten = self.flatten)
        
        # np.random.seed()
        if self.rescale: # take -1*log
            print ("Rescaling data by taking -1*log() ")
            print ("Replacing NaN with 0 ")
            self.rescaled = True
            tt_m = -1 * np.log(tt_m) 
            np.nan_to_num(tt_m,copy=False)
            
            tt_p = -1 * np.log(tt_p)
            np.nan_to_num(tt_p,copy=False)
            
            tt = -1 * np.log(tt)
            np.nan_to_num(tt,copy=False)
        # derivative data
        tt_d = (tt_p - tt_m) / (2. * self.delta_theta)

        data["validation_data"] = tt 
        data["validation_data_d"] = tt_d

        # for plotting purposes we save the upper/lower separately
        data["x_m_test"], data["x_p_test"] = tt_m, tt_p 

        return data

    def plot_covariance(self, show=False):
        """
        Plot the unperturbed Cl data vector and it's covariance as error bars
        to see whether we have calculated good values, since the other plots don't
        really tell us much about the covariance values
        """

        Omega_M = self.theta_fid[0]

        # generate cross power spectra Cl
        Cls, dNdzs = euclid_ccl(Omega_M)

        # Calculate the covariance for every ell
        covariance = calculate_covariance(Cls)

        fig, ax = plt.subplots()
        # for the legend indicating the bin
        ax.plot(ells[0], ells[0]*(ells[0]+1)*Cls[0][0]
            ,color='white',label=f'0,0')
        ax.legend(frameon=False,loc='upper left')
        
        # Plot the standard deviation as error bars
        onesigma = np.sqrt(covariance)
        ax.errorbar(ells, ells*(ells+1)*Cls[0], yerr=ells*(ells+1)*onesigma,fmt='-',lw=1)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel(r'$\ell  (\ell + 1) C_\ell$')
        ax.set_xlabel(r'$\ell$')

        ax.set_title("Determinitistic Cl 0,0 with 1sigma error bars")
        
        plt.tight_layout()
        plt.savefig(f'{self.figuredir}errorbars_{self.modelversion}.png')
        if show: plt.show()
        plt.close()

    def plot_data(self, show=False):
        """ 
        Plot the data

        VARIABLES
        #______________________________________________________________
        show                    bool    whether or not plt.show() is called

        """

        fig, ax = plt.subplots(2, 1, figsize = (8, 6))
        plt.subplots_adjust(hspace=0.5)

        nrows = 10

        for _ in range(nrows):

            # plot nrows random examples from the simulated train data 
            if self.flatten:
                print ('Plotting data... reshaping the flattened data to %s'%str(input_shape))
                temp = self.data['data'][np.random.randint(self.n_train * self.n_s)].reshape(input_shape)
                x, y = temp.T[:,0]
            else:
                print ('Plotting data...')
                temp = self.data['data'][np.random.randint(self.n_train * self.n_s)].reshape(ncombinations,len(ells))
                Cl = temp[0] # plot the (0,0) autocorrelation bin

            if self.rescaled:
                ax[0].plot(ells, Cl)
            else:
                ax[0].loglog(ells, ells*(ells+1)*Cl)
            ax[0].set_title(f'{nrows} examples from training data, Cl (0,0)')
            ax[0].set_xlabel(r'$\ell$')
            ax[0].set_xscale('log')
            if self.rescaled:
                ax[0].set_ylabel(r'$C_\ell$')
            else:
                ax[0].set_ylabel(r'$\ell(\ell+1) C_\ell$')
                

            # plot nrows random examples from the simulated test data 
            if self.flatten:
                temp = self.data['validation_data'][np.random.randint(self.n_s)].reshape(input_shape)
                x, y = temp.T[:,0]
            else:
                temp = self.data['validation_data'][np.random.randint(self.n_train * self.n_s)].reshape(ncombinations,len(ells))
                Cl = temp[0] # plot the (0,0) autocorrelation bin

            if self.rescaled:
                ax[1].plot(ells, Cl)
            else:
                ax[1].loglog(ells, ells*(ells+1)*Cl)
            ax[1].set_title(f'{nrows} examples from test data, Cl (0,0)')
            ax[1].set_xlabel(r'$\ell$')
            ax[1].set_xscale('log')
            if self.rescaled:
                ax[1].set_ylabel(r'$C_\ell$')
            else:
                ax[1].set_ylabel(r'$\ell(\ell+1) C_\ell$')

        # plt.legend()

        plt.savefig(f'{self.figuredir}data_visualization_{self.modelversion}.png')
        if show: plt.show()
        plt.close()

    def plot_derivatives(self, show=False):
        """ 
        Plot the upper and lower perturbed data 
        Good to check if the sample variance is being 
        surpressed. This needs to be done or the network learns very slowly

        VARIABLES
        #______________________________________________________________
        show                    bool    whether or not plt.show() is called

        """

        fig, ax = plt.subplots(5, 2, figsize = (15, 10), sharex='col')
        # plt.subplots_adjust(wspace = 0, hspace = 0.1)
        plt.subplots_adjust(hspace=0.5)
        training_index = np.random.randint(0,self.n_train * self.n_p)
        
        if self.flatten:
            print ('Plotting derivatives... reshaping the flattened data to %s'%str(input_shape))
            # TODO
            temp = self.data['x_p'][training_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            print ('Plotting derivatives... reshaping the flattened data to power spectra')
            temp = self.data['x_p'][training_index].reshape(len(theta_fid),ncombinations,len(ells))
            # temp has shape (num_params, ncombinations, len(ells))
            Cl = temp[:,0,:] # plot the (0,0) autocorrelation bin
        
        # Cl has shape (1,10) since it is the data vector for the 
        # upper training image for both params
        labels =[r'$θ_1$ ($\Omega_M$)']

        # we loop over them in this plot to assign labels
        for i in range(Cl.shape[0]):
            if self.rescaled:
                ax[0, 0].plot(ells, Cl[i],label=labels[i])
            else:
                ax[0, 0].loglog(ells, ells*(ells+1)*Cl[i],label=labels[i])
        ax[0, 0].set_title('One upper training example, Cl 0,0')
        ax[0, 0].set_xlabel(r'$\ell$')
        if self.rescaled:
            ax[0, 0].set_ylabel(r'$C_\ell$')
        else:
            ax[0, 0].set_ylabel(r'$\ell(\ell+1) C_\ell$')

        ax[0, 0].set_xscale('log')

        ax[0, 0].legend(frameon=False)

        if self.flatten:
            # TODO
            temp = self.data['x_m'][training_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            temp = self.data['x_m'][training_index].reshape(len(theta_fid),ncombinations,len(ells))
            # temp has shape (num_params, ncombinations, len(ells))
            Cl = temp[:,0,:] # plot the (0,0) autocorrelation bin

        for i in range(Cl.shape[0]):
            if self.rescaled:
                ax[1, 0].plot(ells, Cl[i])
            else:
                ax[1, 0].loglog(ells, ells*(ells+1)*Cl[i])
        ax[1, 0].set_title('One lower training example, Cl 0,0')
        ax[1, 0].set_xlabel(r'$\ell$')
        if self.rescaled:
            ax[1, 0].set_ylabel(r'$C_\ell$')
        else:
            ax[1, 0].set_ylabel(r'$\ell(\ell+1) C_\ell$')

        ax[1, 0].set_xscale('log')

        if self.flatten:
            # TODO
            temp = self.data["x_m"][training_index].reshape(len(theta_fid),*input_shape)
            xm, ym = temp.T[:,0]

            temp = self.data["x_p"][training_index].reshape(len(theta_fid),*input_shape)
            xp, yp = temp.T[:,0]
        else:
            temp = self.data['x_m'][training_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl_lower = temp[:,0,:]
            temp = self.data['x_p'][training_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl_upper = temp[:,0,:]

        for i in range(Cl_lower.shape[0]):
            ax[2, 0].plot(ells, (Cl_upper[i]-Cl_lower[i]))
        ax[2, 0].set_title('Upper - lower input data: train sample');
        ax[2, 0].set_xlabel(r'$\ell$')
        ax[2, 0].set_ylabel(r'$C_\ell (u) - C_\ell (m) $')
        ax[2, 0].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[2, 0].set_xscale('log')

        for i in range(Cl_lower.shape[0]):
            ax[3, 0].plot(ells, (Cl_upper[i]-Cl_lower[i])/(2*delta_theta[i]))
        ax[3, 0].set_title('Numerical derivative: train sample');
        ax[3, 0].set_xlabel(r'$\ell$')
        ax[3, 0].set_ylabel(r'$\Delta C_\ell / 2\Delta \theta$')
        ax[3, 0].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[3, 0].set_xscale('log')

        for i in range(Cl_lower.shape[0]):
            ax[4, 0].plot(ells, self.data['data_d'][training_index].reshape(ncombinations,len(ells))[0])
        ax[4, 0].set_title('Data_d: train sample Cl 0,0');
        ax[4, 0].set_ylabel(r'$\Delta C_\ell / 2\Delta \theta$')
        ax[4, 0].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[4, 0].set_xscale('log')

        ax[-1, 0].set_xlabel(r'$\ell$')

        test_index = np.random.randint(self.n_p)

        if self.flatten:
            # TODO
            temp = self.data['x_p_test'][test_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            temp = self.data['x_p_test'][test_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl = temp[:,0,:] # plot the (0,0) autocorrelation bin
        
        for i in range(Cl.shape[0]):
            if self.rescaled:
                ax[0, 1].plot(ells, Cl[i])
            else:
                ax[0, 1].loglog(ells, ells*(ells+1)*Cl[i])
        ax[0, 1].set_title('One upper test example Cl 0,0')
        ax[0, 1].set_xlabel(r'$\ell$')
        if self.rescaled:
            ax[0, 1].set_ylabel(r'$C_\ell$')
        else:
            ax[0, 1].set_ylabel(r'$\ell(\ell+1) C_\ell$')

        if self.flatten:
            # TODO
            temp = self.data['x_m_test'][test_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            temp = self.data['x_m_test'][test_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl = temp[:,0,:] # plot the (0,0) autocorrelation bin

        ax[0, 1].set_xscale('log')

        for i in range(Cl.shape[0]):
            if self.rescaled:
                ax[1, 1].plot(ells, Cl[i])
            else:
                ax[1, 1].loglog(ells, ells*(ells+1)*Cl[i])
        ax[1, 1].set_title('One lower test example Cl 0,0')
        ax[1, 1].set_xlabel(r'$\ell$')
        if self.rescaled:
            ax[1, 1].set_ylabel(r'$C_\ell$')
        else:
            ax[1, 1].set_ylabel(r'$\ell(\ell+1) C_\ell$')

        if self.flatten:
            # TODO
            temp = self.data["x_m_test"][test_index].reshape(len(theta_fid),*input_shape)
            xm, ym = temp.T[:,0]

            temp = self.data["x_p_test"][test_index].reshape(len(theta_fid),*input_shape)
            xp, yp = temp.T[:,0]
        else:
            temp = self.data['x_m_test'][test_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl_lower = temp[:,0,:]
            temp = self.data['x_p_test'][test_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl_upper = temp[:,0,:]
        
        ax[1, 1].set_xscale('log')

        for i in range(Cl_lower.shape[0]):
            ax[2, 1].plot(ells, (Cl_upper[i]-Cl_lower[i]))
        ax[2, 1].set_title('Upper - lower input data: test sample');
        ax[2, 1].set_xlabel(r'$\ell$')
        ax[2, 1].set_ylabel(r'$C_\ell (u) - C_\ell (m) $')
        ax[2, 1].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[2, 1].set_xscale('log')


        for i in range(Cl_lower.shape[0]):
            ax[3, 1].plot(ells, (Cl_upper[i]-Cl_lower[i])/(2*delta_theta[i]))
        ax[3, 1].set_title('Numerical derivative: train sample');
        ax[3, 1].set_xlabel(r'$\ell$')
        ax[3, 1].set_ylabel(r'$\Delta C_\ell / \Delta \theta $')
        ax[3, 1].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[3, 1].set_xscale('log')

        for i in range(Cl_lower.shape[0]):
            ax[4, 1].plot(ells, self.data['validation_data_d'][test_index].reshape(ncombinations,len(ells))[0])
        ax[4, 1].set_title('Data_d: test sample Cl 0,0');
        ax[4, 1].set_ylabel(r'$\Delta C_\ell / 2\Delta \theta$')
        ax[4, 1].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[4, 1].set_xscale('log')

        ax[-1, 1].set_xlabel(r'$\ell$')

        plt.savefig(f'{self.figuredir}derivatives_visualization_{self.modelversion}.png')
        if show: plt.show()
        plt.close()

    def plot_derivatives_divided(self, show=False):
        """ 
        Plot the upper and lower perturbed data, divided by CL 
        see question by MJ: https://github.com/ErikOsinga/master_research_project/issues/6

        VARIABLES
        #______________________________________________________________
        show                    bool    whether or not plt.show() is called

        """

        fig, ax = plt.subplots(3, 2, figsize = (15, 10))
        # plt.subplots_adjust(wspace = 0, hspace = 0.1)
        plt.subplots_adjust(hspace=0.5)
        training_index = np.random.randint(self.n_train * self.n_p)
        
        if self.flatten:
            print ('Plotting derivatives... reshaping the flattened data to %s'%str(input_shape))
            # TODO
            temp = self.data['x_p'][training_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            print ('Plotting derivatives... reshaping the flattened data to power spectra')
            temp = self.data['x_p'][training_index].reshape(len(theta_fid),ncombinations,len(ells))
            # temp has shape (num_params, ncombinations, len(ells))
            Cl = temp[:,0,:] # plot the (0,0) autocorrelation bin
        
        # Cl has shape (1,10) since it is the data vector for the 
        # upper training image for both params
        labels =[r'$θ_1$ ($\Omega_M$)']

        # we loop over them in this plot to assign labels
        for i in range(Cl.shape[0]):
            if self.rescaled:
                ax[0, 0].plot(ells, Cl[i],label=labels[i])
            else:
                ax[0, 0].loglog(ells, ells*(ells+1)*Cl[i],label=labels[i])
        ax[0, 0].set_title('One upper training example, Cl 0,0')
        ax[0, 0].set_xlabel(r'$\ell$')
        if self.rescaled:
            ax[0, 0].set_ylabel(r'$C_\ell$')
        else:
            ax[0, 0].set_ylabel(r'$\ell(\ell+1) C_\ell$')

        ax[0, 0].legend(frameon=False)

        if self.flatten:
            # TODO
            temp = self.data['x_m'][training_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            temp = self.data['x_m'][training_index].reshape(len(theta_fid),ncombinations,len(ells))
            # temp has shape (num_params, ncombinations, len(ells))
            Cl = temp[:,0,:] # plot the (0,0) autocorrelation bin

        for i in range(Cl.shape[0]):
            if self.rescaled:
                ax[1, 0].plot(ells, Cl[i])
            else:
                ax[1, 0].loglog(ells, ells*(ells+1)*Cl[i])
        ax[1, 0].set_title('One lower training example, Cl 0,0')
        ax[1, 0].set_xlabel(r'$\ell$')
        if self.rescaled:
            ax[1, 0].set_ylabel(r'$C_\ell$')
        else:
            ax[1, 0].set_ylabel(r'$\ell(\ell+1) C_\ell$')

        if self.flatten:
            # TODO
            temp = self.data["x_m"][training_index].reshape(len(theta_fid),*input_shape)
            xm, ym = temp.T[:,0]

            temp = self.data["x_p"][training_index].reshape(len(theta_fid),*input_shape)
            xp, yp = temp.T[:,0]
        else:
            temp = self.data['x_m'][training_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl_lower = temp[:,0,:]
            temp = self.data['x_p'][training_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl_upper = temp[:,0,:]

        for i in range(Cl_lower.shape[0]):
            ax[2, 0].plot(ells, (Cl_upper[i]-Cl_lower[i])/self.Cl_noiseless)
        ax[2, 0].set_title('Difference between upper and lower training examples');
        ax[2, 0].set_xlabel(r'$\ell$')
        ax[2, 0].set_ylabel(r'$\Delta C_\ell$ / $C_{\ell,thr}$')
        ax[2, 0].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[2, 0].set_xscale('log')

        # also plot sigma_cl / CL
        sigma_cl = np.sqrt(self.covariance)
        ax[2, 0].plot(ells, sigma_cl/self.Cl_noiseless, label=r'$\sigma_{Cl} / C_{\ell,thr}$')
        ax[2, 0].legend(frameon=False)

        test_index = np.random.randint(self.n_p)

        if self.flatten:
            # TODO
            temp = self.data['x_p_test'][test_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            temp = self.data['x_p_test'][test_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl = temp[:,0,:] # plot the (0,0) autocorrelation bin
        
        for i in range(Cl.shape[0]):
            if self.rescaled:
                ax[0, 1].plot(ells, Cl[i])
            else:
                ax[0, 1].loglog(ells, ells*(ells+1)*Cl[i])
        ax[0, 1].set_title('One upper test example Cl 0,0')
        ax[0, 1].set_xlabel(r'$\ell$')
        if self.rescaled:
            ax[0, 1].set_ylabel(r'$C_\ell$')
        else:
            ax[0, 1].set_ylabel(r'$\ell(\ell+1) C_\ell$')

        if self.flatten:
            # TODO
            temp = self.data['x_m_test'][test_index].reshape(len(theta_fid),*input_shape)
            x, y = temp.T[:,0]
        else:
            temp = self.data['x_m_test'][test_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl = temp[:,0,:] # plot the (0,0) autocorrelation bin

        for i in range(Cl.shape[0]):
            if self.rescaled:
                ax[1, 1].plot(ells, Cl[i])
            else:
                ax[1, 1].loglog(ells, ells*(ells+1)*Cl[i])
        ax[1, 1].set_title('One lower test example Cl 0,0')
        ax[1, 1].set_xlabel(r'$\ell$')
        if self.rescaled:
            ax[1, 1].set_ylabel(r'$C_\ell$')
        else:
            ax[1, 1].set_ylabel(r'$\ell(\ell+1) C_\ell$')

        if self.flatten:
            # TODO
            temp = self.data["x_m_test"][test_index].reshape(len(theta_fid),*input_shape)
            xm, ym = temp.T[:,0]

            temp = self.data["x_p_test"][test_index].reshape(len(theta_fid),*input_shape)
            xp, yp = temp.T[:,0]
        else:
            temp = self.data['x_m_test'][test_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl_lower = temp[:,0,:]
            temp = self.data['x_p_test'][test_index].reshape(len(theta_fid),ncombinations,len(ells))
            Cl_upper = temp[:,0,:]
        
        for i in range(Cl_lower.shape[0]):
            ax[2, 1].plot(ells, (Cl_upper[i]-Cl_lower[i]) / self.Cl_noiseless)
        ax[2, 1].set_title('Difference between upper and lower test samples');
        ax[2, 1].set_xlabel(r'$\ell$')
        ax[2, 1].set_ylabel(r'$\Delta C_\ell$ / $C_{\ell,thr}$')
        ax[2, 1].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[2, 1].set_xscale('log')

        # also plot sigma_cl / CL
        sigma_cl = np.sqrt(self.covariance)
        ax[2, 1].plot(ells, sigma_cl/self.Cl_noiseless, label=r'$\sigma_{Cl} / C_{\ell,thr}$')

        plt.savefig(f'{self.figuredir}derivatives_visualization_divided_{self.modelversion}.png')
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

        print ('Creating network, changing data will have no effect beyond this point.')
        n = IMNN.IMNN(parameters=self.parameters)

        if self.load_network:
            n.restore_network()
        else:
            n.setup(network = self.network, load_data = self.data)

        return n

    def rescale_data(self):
        """
        Rescaling the input data. 


        """

        # Dividing every array of simulated data vectors by the mean of that array.
        '''# Didnt work
        for key in self.data.keys():
            self.data[key] /= np.mean(self.data[key])
        '''

        raise ValueError("Deprecated, see create_data")
        self.rescaled = True

        # Mean normalization
        """ didnt work
        for key in self.data.keys():
            self.data[key] -= np.mean(self.data[key])
            self.data[key] /= (np.max(self.data[key]) - np.min(self.data[key]))
        """

        # Median normalization
        """ didnt work, still dividing by large number 
        for key in self.data.keys():
            self.data[key] -= np.median(self.data[key])
            self.data[key] /= (np.max(self.data[key]) - np.min(self.data[key]))
        """

        # Divide by median
        """ didnt work
        for key in self.data.keys():
            self.data[key] -= np.median(self.data[key])
            self.data[key] /= (np.median(self.data[key]))
        """

        # Take logarithm of data
        """ didnt work
        for key in self.data.keys():
            self.data[key] = np.log10(self.data[key])
        """

        # Scale by length of vector
        """
        for key in self.data.keys():
            self.data[key] /= np.linalg.norm(self.Cl_noiseless)
        """

        
        # Scale by negative of the natural logarithm 
        for key in self.data.keys():
            self.data[key] = -1 * np.log(self.data[key]) 
        
        """
        # Scale by subtracting the mean and dividing by std
        std = np.nanstd(self.data['data'])
        mean = np.nanmean(self.data['data'])
        for key in self.data.keys():
            # self.data[key] -= np.log(self.Cl_noiseless) # -1* # scale this same way
            # self.data[key] -= self.Cl_noiseless # -1* # scale this same way
            self.data[key] -= mean 
            self.data[key] /= std
        """


    def save_data_to_disk(self):
        """
        Save current train/test data to disk. In the directory ./preloaded_data/

        See also load_data_from_disk

        """
        Omega_M = self.theta_fid[0]
        for key in self.data.keys():
            np.save(f'./preloaded_data/{Omega_M}_{self.delta_theta[0]}_{key}.npy', self.data[key])

    def load_data_from_disk(self):
        """
        Load data from disk. Looks in the directory ./preloaded_data/

        See also load_data_from_disk

        """
        data = dict()
        Omega_M = self.theta_fid[0]
        der_den = 1. / (2. * self.delta_theta)

        print ("Loading data from disk.. Omega_M = ", Omega_M, "delta_theta = ", self.delta_theta[0])

        for key in ['x_central', 'x_m', 'x_p', 'x_central_test', 'x_m_test', 'x_p_test']:
            data[key] = np.load(f'./preloaded_data/{Omega_M}_{self.delta_theta[0]}_{key}.npy')

        return data, der_den

    def train_network(self, n, restart=False, diagnostics=True):
        """ 
        Train the created network with the given data and parameters
        Saves the history and the determinant of the final fisher info

        INPUTS
        #______________________________________________________________
        n                       class   IMNN class as defined in IMNN.py
        restart                 bool    whether to restart from scratch or continue 
        diagnostics             bool    whether to collect diagnostics
        
        """

        # at_once is basically a kind of batch size, but we dont update every batch
        n.train(updates = self.num_epochs, at_once = 1000, learning_rate=self.eta
            , constraint_strength = 2., training_dictionary = self.training_dictionary
            , validation_dictionary = self.validation_dictionary
            , data = self.data, get_history = True
            , restart= restart, diagnostics = diagnostics)

        # Save the trained network
        n.save_network(filename=self.parameters["filename"])

        # save the network history to a file
        utils_mrp.save_history(self, n)

        # save the det(Final Fisher info) in the modelsettings.csv file
        utils_mrp.save_final_fisher_info(self, n)

    def plot_variables(self, n, show=False, diagnostics=False):
        """ 
        Plot variables vs epochs

        INPUTS
        #______________________________________________________________
        n                       class   IMNN class as defined in IMNN.py
        
        """

        if diagnostics:
            fig, ax = plt.subplots(5, 1, sharex = True, figsize = (10, 10))
        else:
            fig, ax = plt.subplots(2, 1, sharex = True, figsize = (10, 10))

        plt.subplots_adjust(hspace = 0)
        end = len(n.history["det F"])
        epochs = np.arange(end)
        a, = ax[0].plot(epochs, n.history["det F"], label = 'Training data')
        b, = ax[0].plot(epochs, n.history["det test F"], label = 'Test data')
        # ax[0].axhline(y=5,ls='--',color='k')
        ax[0].legend(frameon = False)
        ax[0].set_ylabel(r'$|{\bf F}_{\alpha\beta}|$')
        ax[0].set_title('Final Fisher info on test data: %.3f'%n.history["det test F"][-1])
        ax[1].plot(epochs, n.history["loss"])
        ax[1].plot(epochs, n.history["test loss"])
        # ax[1].set_xlabel('Number of epochs')
        ax[1].set_ylabel(r'$\Lambda$')
        ax[1].set_xlim([0, len(epochs)]);
        
        if diagnostics:
            ax[2].plot(epochs, n.history["det C"])
            ax[2].plot(epochs, n.history["det test C"])
            # ax[2].set_xlabel('Number of epochs')
            ax[2].set_ylabel(r'$|{\bf C}|$')
            ax[2].set_xlim([0, len(epochs)]);
            
            # Derivative of first summary wrt to theta1                theta1 is 3rd dimension index 0
            ax[3].plot(epochs, np.array(n.history["dμdθ"])[:,0,0]
                , color = 'C0', label=r'$\theta_1$',alpha=0.5)
            
            """
            # Derivative of first summary wrt to theta2                theta2 is 3rd dimension index 1
            ax[3].plot(epochs, np.array(n.history["dμdθ"])[:,0,1]
                , color = 'C0', ls='dashed', label=r'$\theta_2$',alpha=0.5)
            """

            # Test Derivative of first summary wrt to theta1                   theta1 is 3rd dimension index 0
            ax[3].plot(epochs, np.array(n.history["test dμdθ"])[:,0,0]
                , color = 'C1', label=r'$\theta_1$',alpha=0.5)
            
            """
            # Test Derivative of first summary wrt to theta2                   theta2 is 3rd dimension index 1
            ax[3].plot(epochs, np.array(n.history["test dμdθ"])[:,0,1]
                , color = 'C1', ls='dashed', label=r'$\theta_2$',alpha=0.5)
            ax[3].legend(frameon=False)
            """

            ax[3].set_ylabel(r'$\partial\mu/\partial\theta$')
            # ax[3].set_xlabel('Number of epochs')
            ax[3].set_xlim([0, len(epochs)])

            # Mean of network output summary 1
            ax[4].plot(epochs, np.array(n.history["μ"])[:,0],alpha=0.5)
            # Mean of test output network summary 1
            ax[4].plot(epochs, np.array(n.history["test μ"])[:,0],alpha=0.5)
            ax[4].set_ylabel('μ')
            ax[4].set_xlabel('Number of epochs')
            ax[4].set_xlim([0, len(epochs)])
        

        print ('Maximum Fisher info on train data:',np.max(n.history["det F"]))
        print ('Final Fisher info on train data:',(n.history["det F"][-1]))
        
        print ('Maximum Fisher info on test data:',np.max(n.history["det test F"]))
        print ('Final Fisher info on test data:',(n.history["det test F"][-1]))

        if np.max(n.history["det test F"]) == n.history["det test F"][-1]:
            print ('Promising network found, possibly more epochs needed')

        plt.tight_layout()
        plt.savefig(f'{self.figuredir}variables_vs_epochs_{self.modelversion}.png')
        if show: plt.show()
        plt.close()

    def plot_train_output(self, n, show=False, amount=10):
        """ 
        Plot network output on training set vs epochs, and show the mean as well

        INPUTS
        #______________________________________________________________
        n                       class   IMNN class as defined in IMNN.py
        amount                  int     how many outputs to plot
        
        """

        fig, axes = plt.subplots(3, 1, sharex = True, figsize = (8, 6))
        ax = axes[0]

        # track 'amount' random outputs of the 1000 input simulations
        random_indices = np.random.randint(0,self.n_s, amount)
        outputs = np.asarray(n.history['train output'])[:,random_indices,0]
        end = len(n.history['train output'])
        epochs = np.arange(end)

        # plot 'amount' random outputs vs epochs
        for i in range(amount):
            ax.plot(epochs,outputs[:,i],alpha=0.5,ls='dashed')

        # plot the network mean of all the input simulations
        ax.plot(epochs, np.array(n.history["μ"])[:,0],ls='solid',label='μ')
        ax.set_ylabel('Network output')
        ax.set_title(f'Output of {amount} random input simulations')
        ax.legend()

        # plot the network mean of all the input simulations, in a separate plot
        ax = axes[1]
        ax.plot(epochs, np.array(n.history["μ"])[:,0],ls='solid',label='μ')
        ax.set_title(f'Network mean')
        ax.set_ylabel('Network output')

        # plot the numpy mean of this subset
        ax = axes[2]
        ax.set_title('numpy mean/std of the random subset of simulations')
        ax.errorbar(epochs, np.mean(outputs,axis=1), yerr=np.std(outputs,axis=1),label=r'$1\sigma$',zorder=1)
        ax.plot(epochs, np.mean(outputs,axis=1),label='mean',zorder=2)
        ax.set_ylabel('Value')
        ax.legend()
        
        # print (np.std(outputs,axis=1))

        axes[-1].set_xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(f'{self.figuredir}output_vs_epochs_{self.modelversion}.png')
        if show: plt.show()
        plt.close()

    def calc_derivative_Omega_M(self):
        """
        Calculates the (worst) numerical derivative of the real data
        
        # We calculate the 'normal' derivative wrt Omega_M:
            d (C_ell)/ d (Omega_M)
        """

        Omega_M = self.theta_fid[0] # Fiducial param
        domega = 0.001
        # Calculate the noiseless Cl just above the fidicual param
        Cls_t, _ = euclid_ccl(Omega_M = Omega_M+domega)
        # Numerical derivative, give Cl noiseless its redundant axis back
        deriv_OM = (Cls_t - self.Cl_noiseless[np.newaxis, :]) / domega

        return deriv_OM # shape (1,100)

    def do_ABC(self, n, real_data, prior, draws, show=False, epsilon=None, oneD=True
        ,analytic_posterior = None, param_array = None, at_once=True,save_sims=None):
        """ 
        Perform ABC

        INPUTS
        #______________________________________________________________
        n                       class   IMNN class as defined in IMNN.py
        real_data               array   array containing true data
        prior                   dict    "mean", "variance", "lower" and "upper" 
                                            bounds for a truncated Gaussian prior
        draws                   int     amount of draws from prior
        show                    bool    whether or not plt.show() is called
        oneD                    bool    whether to plot one dimensional posteriors
                                        or two dimensional with the corner module
        analytic_posterior      array   normalized values of analytic posterior 
        param_array             array   parameter values of analytic posterior
        at_once                 bool    whether to run simulations all or one at a time

        RETURNS
        #_____________________________________________________________
        abc                     class   abc object (see ABC.py)         
    
        """

        Gaussprior = priors.TruncatedGaussian(prior["mean"],prior["variance"],prior["lower"]
                                        ,prior["upper"])

        abc = ABC.ABC(real_data = real_data, prior = Gaussprior, sess=n.sess
            , get_compressor=n.get_compressor, simulator=generate_data_no_interp, seed=None
            , simulator_args=None, dictionary = self.validation_dictionary)

        # actually perform ABC
        if save_sims is not None: 
            save_sims = f'/net/reusel/data1/osinga/master_research_project/saved_data/sims_{version}'
            print ("Saving simulations to :",save_sims)

        abc.ABC(draws=draws, at_once=at_once, save_sims=save_sims, MLE=True)

        # Draws are accepted if the distance between the simulation summary and the 
        # simulation of real data are close (i.e., smaller than some value epsilon)
        if epsilon is None: epsilon = np.linalg.norm(abc.summary)/2. # chosen quite arbitrarily
        accept_indices = np.argwhere(abc.ABC_dict["distances"] < epsilon)[:, 0]
        reject_indices = np.argwhere(abc.ABC_dict["distances"] >= epsilon)[:, 0]

        print ('Epsilon is chosen to be %.2f'%epsilon)
        print("Number of accepted samples = ", accept_indices.shape[0])

        truths = theta_fid

        # plot output samples and histogram of the accepted samples in 1D
        def plot_samples_oneD():
            fig, ax = plt.subplots(2, 1, sharex = 'col', figsize = (10, 10))
            plt.subplots_adjust(hspace = 0)
            
            # Plot the accepted/rejected samples
            ax[0].scatter(abc.ABC_dict["parameters"][reject_indices] # x
                , abc.ABC_dict["summaries"][reject_indices] # y
                , s = 1, alpha = 0.1, label = "Rejected samples", color = "C3") 
            
            ax[0].scatter(abc.ABC_dict["parameters"][accept_indices]
             , abc.ABC_dict["summaries"][accept_indices]
             , s = 1, label = "Accepted samples", color = "C6", alpha = 0.5)
            
            ax[0].axhline(abc.summary[0]
                , color = 'black', linestyle = 'dashed', label = "Summary of observed data")
            
            ax[0].legend(frameon=False)
            ax[0].set_ylabel('First network output', labelpad = 0)
            ax[0].set_xlim([prior["lower"][0], prior["upper"][0]])
            # ax[0].set_xticks([])
            ax[1].set_xlabel(r"$\theta_1 = \Omega_M$")

            # plot the posterior
            ax[1].hist(abc.ABC_dict["parameters"][accept_indices], np.linspace(prior["lower"][0], prior["upper"][0], 100), histtype = u'step', density = True, linewidth = 1.5, color = "C6", label = "ABC posterior");
            ax[1].axvline(abc.MLE[0], linestyle = "dashed", color = "black", label = "(Gaussian) MLE")
            ax[1].set_xlim([prior["lower"][0], prior["upper"][0]])
            ax[1].set_ylabel('$\\mathcal{P}(\\theta_1|{\\bf d})$')
            # ax[1].set_yticks([])
            # ax[1].set_xticks([])
            ax[1].set_xlabel(r"$\theta_1 = \Omega_M$")

            # Theta-fid
            ax[1].axvline(theta_fid[0], linestyle = "dashed", label = "$\\theta_{fid}$")

            if analytic_posterior is not None:
                ax[1].plot(param_array, analytic_posterior, linewidth = 1.5, color = 'C2'
                    , label = "Analytic posterior")
            

            ax[1].legend(frameon = False)
            fig.suptitle(f"Epsilon = {epsilon}")

            plt.savefig(f'{self.figuredir}ABC_{self.modelversion}_1D.png')
            if show: plt.show()
            plt.close()

        # plot approximate posterior of the accepted samples in 2D
        def plot_samples_twoD():
            sys.exit("Only 1 parameter = 1 dimension")
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

        return abc

    def do_PMC_ABC(self, n, real_data, prior, draws, num_keep, abc=None, criterion = 0.1
        , show=False, oneD='both',analytic_posterior = None, param_array = None):
        """ 
        Perform PMC ABC, which is a way of reducing the number of draws
        The inputs work in a very similar way to the ABC function above. If we 
        want 1000 samples from the approximate distribution at the end of the
        PMC we need to set num_keep = 1000. The initial random draw is initialised
        with num_draws, the larger this is the better proposal distr will be on
        the 1st iteration.


        INPUTS
        #______________________________________________________________
        n                       class   IMNN class as defined in IMNN.py
        real_data               array   array containing true data
        prior                   dict    "mean", "variance", "lower" and "upper" 
                                            bounds for a truncated Gaussian prior
        draws                   int     number of initial draws from the prior
        num_keep                int     number of samples in the approximate posterior
        abc                     class   output of the do_ABC function, to continue from 
                                        what the ABC function already has done
                                or None None implies starting from scratch 
        criterion               float   ratio of number of draws wanted over number
                                        of draws needed
        show                    bool    whether or not plt.show() is called
        oneD                    bool    whether to plot one dimensional posteriors
                                        or two dimensional with the corner module   
        analytic_posterior      array   normalized values of analytic posterior 
        param_array             array   parameter values of analytic posterior  

        RETURNS
        #_____________________________________________________________
        theta                   list    sampled parameter values in the approximate posterior           
        all_epsilon             list    progression of epsilon during PMC       
    
        """

        if abc is None:
            # Start from scratch
            print ("Starting PMC from scratch")
            restart = True
            Gaussprior = priors.TruncatedGaussian(prior["mean"],prior["variance"]
                                                  ,prior["lower"],prior["upper"])

            abc = ABC.ABC(real_data = real_data, prior = Gaussprior, sess=n.sess
            , get_compressor=n.get_compressor, simulator=generate_data_no_interp, seed=None
            , simulator_args=None, dictionary = self.validation_dictionary)

        else: # Use the abc object from the do_ABC() function
            print ("Starting PMC from previous ABC or PMC result")
            restart = False
    
        all_epsilon = abc.PMC(draws = draws, posterior = num_keep
            , criterion = criterion, at_once = True, save_sims = None, MLE = True) 

        # plot output samples and histogram of approximate posterior
        def plot_samples_oneD():
            
            fig, ax = plt.subplots(2, 1, sharex = 'col', figsize = (10, 10))
            plt.subplots_adjust(hspace = 0)
            ax[0].scatter(abc.PMC_dict["parameters"][:] , abc.PMC_dict["summaries"][:]
                , s = 1, label = "Accepted samples", alpha=0.5)
            ax[0].axhline(abc.summary[0], color = 'black', linestyle = 'dashed', label = "Summary of observed data")
            ax[0].set_ylabel('Network output', labelpad = 0)
            # ax[0].set_ylim([np.min(s_[:]), np.max(s_[:])])
            ax[0].set_xlim([prior["lower"][0], prior["upper"][0]])
            
            ax[1].hist(abc.PMC_dict["parameters"], np.linspace(prior["lower"][0]
                , prior["upper"][0], 100), histtype = u'step', density = True
                , linewidth = 1.5, label = "PMC posterior", color = 'k')

            ax[1].set_xlabel('$\\theta_1 = \Omega_M$ ')
            ax[1].set_ylabel('$\\mathcal{P}(\\theta1|{\\bf d})$')
            ax[1].set_yticks([])

            if analytic_posterior is not None:
                ax[1].plot(param_array, analytic_posterior, linewidth = 1.5, color = 'C2'
                    , label = "Analytic posterior")

            # Theta-fid
            ax[1].axvline(theta_fid[0], linestyle = "dashed", label = "$\\theta_{fid}$")

            ax[1].legend(frameon=False)

            # fig.suptitle("Only showing 1st network output summary out of %i \n Full network output on real data: %s"%(s_.shape[1],str(summary_)))
            fig.suptitle("Results of PMC")

            plt.savefig(f'{self.figuredir}PMC_ABC_{self.modelversion}_1D.png')
            if show: plt.show()
            plt.close()

        # plot output samples and histogram of the accepted samples
        def plot_samples_twoD():
            sys.exit('There is only 1 dimension, cannot make corner plot')
            hist_kwargs = {} # add kwargs to give to matplotlib hist funct
            fig, ax = plt.subplots(2, 2, figsize = (10, 10))
            fig = corner.corner(theta_, fig=fig, truths = theta_fid
                , labels=['$\\theta_1$ (variance)']
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

        return abc


#################################
# HELPER FUNCTIONS to generate data
#################################

def cosmosis_cells(Omega_M, save_dir):
    """
    Generate C_ell as function of ell for a given Omega_M

    Inputs
        Omega_M -- float: Matter density 
        save_dir -- string: Directory where cosmosis will save the data

    Assumed global variables
        nbin -- the amount of tomographic redshift bins
        zmax -- maximum redshift
        dz   -- redshift resolution element
        ell_min -- starting ell value
        ell_max -- final ell_value
        n_ell  -- number of ell values

    Outputs
        Cls -- np.array, shape (nbin*(nbin-1)/2 + nbin, len(ell)): 
                Cross/Auto correlation shear spectra for the tomographic bins

    """
    # Parameters from https://arxiv.org/pdf/1903.01473.pdf
    Omega_b_fraction = 0.15653724 # fraction of Omega_M
    
    A_s = 2.1e-9
    Omega_b = Omega_b_fraction * Omega_M
    h = 0.674
    n_s = 0.965
    w0 = -1.03

    # Calculate Cls with cosmosis
    generate_cells(save_dir, nbin, zmax, dz, ell_min, ell_max, n_ell
        , alpha=alpha, beta=beta, z0=z0, sigz=sigz, ngal=ngal, bias=bias
        , omega_m=Omega_M)

    def generate_cells(save_dir, nbin, zmax, dz, ell_min, ell_max, n_ell, omega_m
    , alpha=1.3, beta=1.5, z0=0.65, sigz=0.05, ngal=30, bias=0 # redshift params, default Euclid
    , h0=0.72, omega_b=0.04, tau=0.08, n_s=0.96
    , A_s=2.1e-9, omega_k=0.0, w=-1.0, wa=0.0):

    # Load calculated Cls from file
    ells, Cls = load_cells(save_dir, nbin)

    return Cls

def euclid_nzs(num_dens):
    """
    Calculate the (AVERAGE?!) number density of sub-sample galaxies per redshift bin

    num_dens = 354,543,086 galaxies per steradian
    
    Euclid num density = 30 arcmin^-2 = 108,000 deg^-2     (arcmin^2 to deg^2 = 60^2)
    In steradians: A steradian is (180/π)^2 square degrees, or 3282.8 deg^2
    So Euclid number density 
    = 108,000 * 3282.8 = 354,543,086 galaxies per steradian

    Returns
        nzs --- list --- (AVERAGE?!) number density of galaxies in redshift bin
    
    """

    # integrate over whole z range and
    # scale to the given number density of 30 per arcmin squared 
    dNdz_obs = ccl.dNdz_tomog(z=z, zmin=zmin, zmax=zmax, pz_func=pz, dNdz_func = dNdz_true)
    dNdz_obs = dNdz_obs/dNdz_obs.sum() * num_dens
    nzs = []
    for i in range(nbin):
        # calculate the number density of galaxies per steradian per bin
        zmin_i, zmax_i = i*(2./nbin), (i+1)*(2./nbin)
        mask = (z>zmin_i)&(z<zmax_i)
        nzs.append(dNdz_obs[mask].sum())

    return nzs

def calculate_Cls_obs(Cls):
    """
    Calculate the power spectrum contaminated by shape noise
    https://arxiv.org/pdf/0810.4170.pdf (Eq. 7)

    """
    Cls_obs = np.copy(Cls)
    counter = 0
    for i in range(nbin):
        for j in range(0,i+1):
            if i == j:
                shotnoise = sn**2/nzs[i]
            else: # cross spectra are not contaminated by shot noise
                shotnoise = 0

            Cls_obs[counter] += shotnoise
            counter +=1

    return Cls_obs
    
def calculate_covariance(Cls):
    """
    Calculate the Gaussian covariance according to https://arxiv.org/pdf/0810.4170.pdf

    First add shape noise with def calculate_Cls_obs
    Then calculate the Gaussian covariance with one equation

    Returns
    covariance_diag -- array of len(ells) elements with the variance of the 1 tomographic bin

    """

    # Cls with added shape noise
    Cls_obs = calculate_Cls_obs(Cls)

    # for just 1 tomographic bin Covariance is easily calculated with the following lines
    # (its diagonal)
    Neffmode = (2*ells+1) * delta_l * fsky
    covariance_diag = Cls_obs[0]*Cls_obs[0] + Cls_obs[0]*Cls_obs[0]
    covariance_diag /= Neffmode

    return covariance_diag

def add_variance(Cls_original, covariance_diag):
    """
    Perturb the data with a Gaussian, with variance given as an array
    of len(ells) points. For 1 tomographic bin the matrix is diagonal in ell
    so the covariance matrix is just one number for every l

    Only have to repeat this function to add noise to the determinitistic Cls_original
    """ 
    
    onesigma = np.sqrt(covariance_diag)
    Cls_perturbed = Cls_original + np.random.normal(0, onesigma)

    return Cls_perturbed

def generate_test_train_Cls(Omega_M, sigma8):
    """
    Since we make simulations at fiducial parameter values both for the training
    and for the test set, it is computationally smart to generate the Cls for 
    these values once, and then add the calculated covariance everytime we need 
    new simulations.

    The same goes for the upper and lower derivatives, this is also calculated twice.
    Once for the test set and once for the training set.

    """
    raise ValueError("TODO later")

    # a noise-free version of Cl 
    Cls, dNdzs = euclid_ccl(Omega_M, sigma8)
        
    # The covariance for every ell for this Cl
    covariance = calculate_covariance(Cls)

    return Cls_original, covariance


def generate_data(θ, train=None, flatten=False, preload=False, noiseless_deriv=True):
    """
    Holder function for the generation of the Cls
    
    θ = vector of lists of [Omega_M]'s to produce a Cl for
    train = either None or an array of [delta_theta1,delta_theta2] for generating
            the upper and lower derivatives
    preload = True / False, True if we can load the data from disk
    noiseless_deriv = True/False, whether to add noise to the upper/lower simulations
                        NOTE THAT THE RANDOM SEED SHOULD BE SET IF WE WANT TO ADD NOISE TO THESE


    Returns the weak lensing data vector flattened to use as input for the IMNN
            shape (num_simulations=len(θ), length of Cl vector)
    """
    θ = np.asarray(θ)
    # print ('Shape',θ.shape)

    if preload:
        raise ValueError("todo: divide between train/test data so they are not identical")

    if preload and (θ[:,0] == θ[0,0]).all():
        if train is not None:
            perturb_param1 = np.array([train[0]])
            θ_first_param = θ[0,0] + perturb_param1
            print (f"Checking disk for saved data with Omega_M = {θ_first_param}")
            try:
                all_Cls = np.load(f'./preloaded_data/Omega_M_{θ_first_param}')
                return all_Cls
            except FileNotFoundError:
                print ('File not found.')

        else:
            Omega_M = θ[0,0]
            print (f"Checking disk for saved data with Omega_M = {Omega_M}")
            try:
                all_Cls = np.load(f'./preloaded_data/Omega_M_{Omega_M}')
                return all_Cls
            except FileNotFoundError:
                print ('File not found')


    def helper_func(θ):
        """
        Generates noisy simulations at θ = vector of lists of [Omega_M]'s
        Called once if train = None, called twice if not
        """
        if (θ[:,0] == θ[0,0]).all():
            Omega_M = θ[0,0]
            print (f"List of parameters contains all the same parameters, Omega_M={Omega_M}")
            
            # generate cross power spectra Cl
            Cls = cosmosis_cells(Omega_M)

            # Calculate the covariance for every ell, have to do this before flattening
            covariance = calculate_covariance(Cls)

            # Cls are returned as array (ncombinations,100), if ncombinations=1, we must flatten it
            Cls = Cls.flatten() # to get rid of the redundant dimension

            # to keep a noise-free version of Cl as well
            Cls_original = np.copy(Cls)

            # For every item in the list of coordinates, perturb the original Cl with
            # a 1D Gaussian with std=sqrt(covariance), save it as a list of (flattened) simulations 
            if train is None:
                all_Cls = []
                for i in range(len(θ)): # can be done in parallel for all i 
                    Cls = add_variance(Cls_original, covariance)
                    all_Cls.append(Cls.flatten())
            
            else: 
                if noiseless_deriv:
                    # if it is for calculating derivatives, just use the noise free version 
                    all_Cls = [Cls_original for i in range(len(θ))]
                else:
                    all_Cls = []
                    for i in range(len(θ)):
                        Cls = add_variance(Cls_original, covariance)
                        all_Cls.append(Cls.flatten())
        
        # TODO // Think about how to generate multiple at once
        # Omega_M, sigma8 = θ,  not possible if they are different params
        else: # generate the simulations one by one...
            print ("List of parameters does not contain all the same parameters. Slow.")
            all_Cls = []        
            for Omega_M in θ[:,0]: # Just one parameter
                # print (Omega_M)
                # Can in theory be done in parallel for all different Omega_M's

                Cls, dNdzs = cosmosis_cells(Omega_M)

                # Calculate the covariance for every ell, have to do this before flattening
                covariance = calculate_covariance(Cls)

                # Cls are returned as array (ncombinations,100), if ncombinations=1, we must flatten it
                Cls = Cls.flatten() # to get rid of the redundant dimension

                # Perturb the original Cl with
                # a 1D Gaussian with std=sqrt(covariance)
                Cls = add_variance(Cls, covariance)

                all_Cls.append(Cls.flatten()) # flatten the Cl data

        # if not preload: np.save(f'./preloaded_data/Omega_M_{Omega_M}', np.asarray(all_Cls))

        return np.asarray(all_Cls) # shape (num_simulations, ncombinations*len(ells))

    if train is not None: # generate derivatives, with perturbed thetas, noise free
        
        perturb_param1 = np.array([train[0]])
        θ_first_param = θ + perturb_param1

        # the upper/lower of the first parameter
        all_Cls_first_param = helper_func(θ_first_param)

        # Return it as an array of shape (num_sim,num_params,length_vector)
        all_Cls_first_param = all_Cls_first_param.reshape(
                                        len(θ),1,all_Cls_first_param.shape[1])
        all_Cls = all_Cls_first_param

        # if not preload: np.save(f'./preloaded_data/Omega_M_{θ_first_param[0,0]}', all_Cls)

        return all_Cls # shape (num_simulations, num_params, ncombinations*len(ells)

    else: # generate simulations at value
        return helper_func(θ)


def generate_data_no_interp(θ, seed, simulator_args, 
    train=None, flatten=False, preload=False, noiseless_deriv=True):
    """
    Holder function for the generation of the Cls for the ABC function
    
    θ = vector of lists of [Omega_M]'s to produce a Cl for
    train = either None or an array of [delta_theta1,delta_theta2] for generating
            the upper and lower derivatives
    preload = True / False, True if we can load the data from disk
    noiseless_deriv = True/False, whether to add noise to the upper/lower simulations
                        NOTE THAT THE RANDOM SEED SHOULD BE SET IF WE WANT TO ADD NOISE TO THESE


    Returns the weak lensing data vector flattened to use as input for the IMNN
            shape (num_simulations=len(θ), length of Cl vector)
    """
    θ = np.asarray(θ)
    # print ('Shape',θ.shape)

    if preload:
        raise ValueError("todo: divide between train/test data so they are not identical")

    if preload and (θ[:,0] == θ[0,0]).all():
        if train is not None:
            perturb_param1 = np.array([train[0]])
            θ_first_param = θ[0,0] + perturb_param1
            print (f"Checking disk for saved data with Omega_M = {θ_first_param}")
            try:
                all_Cls = np.load(f'./preloaded_data/Omega_M_{θ_first_param}')
                return all_Cls
            except FileNotFoundError:
                print ('File not found.')

        else:
            Omega_M = θ[0,0]
            print (f"Checking disk for saved data with Omega_M = {Omega_M}")
            try:
                all_Cls = np.load(f'./preloaded_data/Omega_M_{Omega_M}')
                return all_Cls
            except FileNotFoundError:
                print ('File not found')


    def helper_func(θ):
        """
        Generates noisy simulations at θ = vector of lists of [Omega_M]'s
        Called once if train = None, called twice if not
        """
        if (θ[:,0] == θ[0,0]).all():
            Omega_M = θ[0,0]
            # print (f"List of parameters contains all the same parameters, Omega_M={Omega_M}")
            
            # generate cross power spectra Cl
            Cls, dNdzs = euclid_ccl(Omega_M)

            # Calculate the covariance for every ell, have to do this before flattening
            covariance = calculate_covariance(Cls)

            # Cls are returned as array (ncombinations,100), if ncombinations=1, we must flatten it
            Cls = Cls.flatten() # to get rid of the redundant dimension

            # to keep a noise-free version of Cl as well
            Cls_original = np.copy(Cls)

            # For every item in the list of coordinates, perturb the original Cl with
            # a 1D Gaussian with std=sqrt(covariance), save it as a list of (flattened) simulations 
            if train is None:
                all_Cls = []
                for i in range(len(θ)): # can be done in parallel for all i 
                    Cls = add_variance(Cls_original, covariance)
                    all_Cls.append(Cls.flatten())
            
            else: 
                if noiseless_deriv:
                    # if it is for calculating derivatives, just use the noise free version 
                    all_Cls = [Cls_original for i in range(len(θ))]
                else:
                    all_Cls = []
                    for i in range(len(θ)):
                        Cls = add_variance(Cls_original, covariance)
                        all_Cls.append(Cls.flatten())
        
        # TODO // Think about how to generate multiple at once
        # Omega_M, sigma8 = θ,  not possible if they are different params
        else: # generate the simulations one by one...
            print ("List of parameters does not contain all the same parameters. Slow.")
            all_Cls = []        
            for Omega_M in tqdm.tqdm(θ[:,0]): # Just one parameter
                # Can in theory be done in parallel for all different Omega_M's

                Cls, dNdzs = euclid_ccl(Omega_M)

                # Calculate the covariance for every ell, have to do this before flattening
                covariance = calculate_covariance(Cls)

                # Cls are returned as array (ncombinations,100), if ncombinations=1, we must flatten it
                Cls = Cls.flatten() # to get rid of the redundant dimension

                # Perturb the original Cl with
                # a 1D Gaussian with std=sqrt(covariance)
                Cls = add_variance(Cls, covariance)

                all_Cls.append(Cls.flatten()) # flatten the Cl data

        # if not preload: np.save(f'./preloaded_data/Omega_M_{Omega_M}', np.asarray(all_Cls))
        if nholder1.rescaled:
            # print ("Rescaling the data (hopefully) same as the network")
            all_Cls = -1 * np.log(all_Cls) 
            np.nan_to_num(all_Cls,copy=False)

        return np.asarray(all_Cls) # shape (num_simulations, ncombinations*len(ells))

    if train is not None: # generate derivatives, with perturbed thetas, noise free
        
        perturb_param1 = np.array([train[0]])
        θ_first_param = θ + perturb_param1

        # the upper/lower of the first parameter
        all_Cls_first_param = helper_func(θ_first_param)

        # Return it as an array of shape (num_sim,num_params,length_vector)
        all_Cls_first_param = all_Cls_first_param.reshape(
                                        len(θ),1,all_Cls_first_param.shape[1])
        all_Cls = all_Cls_first_param

        # if not preload: np.save(f'./preloaded_data/Omega_M_{θ_first_param[0,0]}', all_Cls)

        return all_Cls # shape (num_simulations, num_params, ncombinations*len(ells)

    else: # generate simulations at value
        return helper_func(θ)

def generate_data_ABC(theta, seed, simulator_args, train=False):
    '''
    Generate data for ABC, using linear extrapolation
    
    Returns: array of shape (n_s*n_train,input_shape) draws from norm dist
    '''

    # theta = theta.reshape(len(theta))
    # print ((theta).shape)

    def helper():
        # Can in theory be done in parallel for all different Omega_M's
        # print (Omega_M)
        
        # how far in x we have to go
        dOmega = Omega_M - theta_fid[0]

        # Extrapolate from the noiseless fiducial 
        Cls = nholder1.Cl_noiseless[np.newaxis, :] + (dOmega*deriv_Oc)

        # Calculate the covariance for every ell, have to do this before flattening
        covariance = calculate_covariance(Cls)

        # Cls are returned as array (ncombinations,100), if ncombinations=1, we must flatten it
        Cls = Cls.flatten() # to get rid of the redundant dimension

        # Perturb the Cl with
        # a 1D Gaussian with std=sqrt(covariance)
        Cls = add_variance(Cls, covariance)

        all_Cls.append(Cls.flatten()) # flatten the Cl data


    if train:
        sys.exit("No derivative data in ABC")

    else: 
        all_Cls = []   
        if len(theta[:,0]) < 1000:
            # Dont show progress bar for anything < 1000
            for Omega_M in theta[:,0]:  # Just one parameter so [:,0]
                # Can in theory be done in parallel for all different Omega_M's
                helper()
        else:   
            # Show progress bar
            for Omega_M in tqdm.tqdm(theta[:,0]):  # Just one parameter so [:,0]
                helper()
            
    if nholder1.rescaled:
        print ("Rescaling the data (hopefully) same as the network")
        print ("Check if the rescaling went correctly")
        all_Cls = -1 * np.log(all_Cls) 

    return np.asarray(all_Cls) # shape (num_simulations, ncombinations*len(ells))

def build_dense_network(data, hidden_layers, **kwargs):
    """ 
    Adjusted network Tom builder send me, biases are added as a hack 

    PARAMETERS
    hidden_layers -- list -- the number of nodes in the (dense) hidden layers
    
    KWARGS:
    for passing the dropout value (num_keep) if that is needed
    for passing the activation parameter if that is needed
    for passing the training_phase if that is needed

    """
    # Input layer
    with tf.variable_scope("layer_1"): 
        weights = tf.get_variable("weights", shape = [input_shape[-1] + 1
            , hidden_layers[0]], initializer = tf.variance_scaling_initializer())

        output = tf.nn.leaky_relu(tf.matmul(tf.concat([data, tf.ones(dtype = tf.float32
            , shape = (tf.shape(data)[0], 1))], axis = 1) # concat
            , weights, name = "multiply") # matmul
            , α, name = "output") # leaky relu

        # DROP-OUT after the activation func
        output = tf.nn.dropout(output, keep_prob=δ, name = "output")  

    # Hidden layers 1 to len(hidden_layers) - 1
    for i in range(2, len(hidden_layers)-1+2):

        with tf.variable_scope(f"layer_{i}"):
            n_nodes = hidden_layers[i-1]

            weights = tf.get_variable("weights", shape = [hidden_layers[i-2]+1, hidden_layers[i-1]], initializer = tf.variance_scaling_initializer())
            output = tf.nn.leaky_relu(tf.matmul(tf.concat([output, tf.ones(dtype = tf.float32, shape = (tf.shape(data)[0], 1))], axis = 1), weights, name = "multiply"), α, name = "output")

            # DROP-OUT after the activation func
            output = tf.nn.dropout(output, keep_prob=δ, name = "output")  

    # Output layer
    with tf.variable_scope(f"layer_{len(hidden_layers)+1}"):

        weights = tf.get_variable("weights", shape = (hidden_layers[1]+1, n_summaries), initializer = tf.variance_scaling_initializer())
        output = tf.identity(tf.matmul(tf.concat([output, tf.ones(dtype = tf.float32, shape = (tf.shape(data)[0], 1))], axis = 1), weights, name = "multiply"), name = "output")
        # NO DROP-OUT in the last layer


    return output

# #######################################
# COSMOSIS PARAMETERS 
#########################################
save_dir = '/net/reusel/data1/osinga/master_research_project/saved_data/cosmosis/oneparam'

nbin, zmax, dz = 3, 2.0, 0.002 # 1000 samples in redshift
ell_min, ell_max, n_ell = 50, 3000, 200 # 200 samples in ell

# Parameters for ccl to calculate the Euclid n(z) distribution
nz = 1000 #redshift resolution
zmin = 0.
zmax = 2.
z = np.linspace(zmin,zmax,nz)
# number of tomographic bins
nbin = 1 
# number of cross/auto angular power spectra
ncombinations = int(nbin*(nbin+1)/2)
# 100 log equal spaced ell samples should be fine according to https://arxiv.org/pdf/0705.0163.pdf
ells = np.logspace(np.log10(ell_min),np.log10(ell_max),n_ell) # cosmosis generates log spaced ells as well
# I think this is 1
delta_l = 1

"""
Assume a redshift distribution given by
    z^alpha * exp(z/z0)^beta
    with alpha=1.3, beta = 1.5 and z0 = 0.65
"""
alpha=1.3
beta=1.5
z0=0.65
sigz=0.05
ngal=30
bias=0 
dNdz_true = ccl.dNdzSmail(alpha = alpha, beta = beta, z0=z0)
# Assumes photo-z error is Gaussian with a bias is 0.05(1+z)
pz = ccl.PhotoZGaussian(sigma_z0=sigz)

fsky = 15000/41252.96 # fraction of the sky observed by Euclid
sn = 0.26 
num_dens = ngal * 3600 * (180/np.pi)**2# from arcmin^-2 to deg^-2 to sr
nzs = euclid_nzs(num_dens) 



# #######################################
# IMNN PARAMETERS 
#########################################
# The input shape is a 1D vector of length 1*len(ells) (100 most of the time in this case)
input_shape = [ncombinations*len(ells)] 

theta_fid = np.array([0.315]) # Omega_M 

# delta_theta = np.array([0.11]) # perturbation values
delta_theta = np.array([0.02])

n_s = 1000 # number of simulations used at a time to approximate the covariance 
n_train = 5 # splits, for if it doesnt fit into memory
n_train_val = 1 # validation splits, not sure why we would do > 1 ever

# use less simulations for numerical derivative
derivative_fraction = 1.0 # fraction of n_s
# use less simulations for numerical derivative of test data
derivative_fraction_val = 0.2 # fraction of n_s

eta = 1e-3 # learning rate
num_epochs = int(10) 
keep_rate = 1.0 # 1 minus the dropout
verbose = 0

fromdisk = False # whether to read data from disk DOESNT WORK FOR SOME REASON

# MLP
# hidden_layers = [1024, 512, 256, 128, 128]
hidden_layers = [256,256] 
# activation parameter (e.g., for leaky relu: Slope of the activation function at x < 0 )
actparam = 0.01
dtype = 32 # float32

noiseless_deriv = False # whether to not add noise to upper/lower simulations
flatten = False # data is already flat, don't have to flatten it again
rescale = True # wheter to take the negative log of the data

initial_version = int(sys.argv[1])

# For building the network, and defining number of summaries
n_summaries = 1
# Placeholders for build_network function, values are assigned in the
# training_dictionary and validation_dictionary
δ = tf.placeholder(dtype = tf.float32, shape = (), name = "dropout_value")
α = tf.placeholder(dtype = tf.float32, shape = (), name = "activation_parameter")
ϕ = tf.placeholder(dtype = tf.bool, shape = (), name = "training_phase")

# Network needs to be passed to the IMNN module only taking a single tensor. 
network = lambda x: build_dense_network(x, hidden_layers, activation_parameter = α, dropout_value = δ, training_phase = ϕ)


version = initial_version

# There is no dropout yet
training_dictionary = {"dropout_value:0": keep_rate, 
                       "activation_parameter:0": actparam,
                       "training_phase:0": True,
                       }

validation_dictionary = {"dropout_value:0": 1.0, # Always 1 for test data
                         "activation_parameter:0": actparam,
                         "training_phase:0": False,
                         }

version = initial_version

parameters = {
    'number of summaries': n_summaries,
    'filename': "Models/data/model"+str(version),
    'activation': tf.nn.leaky_relu, # CHANGE THIS MANUALLY IF YOU WANT SOMETHING ELSE
    'α': actparam, # negative gradient parameter in case of tf.nn.leaky_relu
    'hidden layers': hidden_layers,
    'flatten': flatten
}



#########################################
#  FUNCTIONS TO TRAIN/EVALUATE NETWORK
#########################################


# Network holder, creates the data as well
nholder1 = nholder(input_shape, generate_data, theta_fid, delta_theta, n_s,
        n_train, derivative_fraction, n_train_val, derivative_fraction_val, 
        eta, parameters, num_epochs, keep_rate, verbose, dtype, training_dictionary
        ,validation_dictionary, network, version, fromdisk, noiseless_deriv, flatten
        ,rescale,load_network=False)

# # creates IMNN network & passes the data to the network 
n = nholder1.create_network()

# # Plot covariance as error bars
nholder1.plot_covariance(show=False)

# to load it again later
# nholder1.save_data_to_disk()

# # Plot data
nholder1.plot_data(show=False)
# # plot derivatives
nholder1.plot_derivatives(show=False)

# # plot what MJ asked
# nholder1.plot_derivatives_divided(show=True)


# # Train network
diagnostics = False # whether to collect diagnostics (slow)
nholder1.train_network(n, restart=False, diagnostics=diagnostics)
# # Plot the output
nholder1.plot_variables(n,show=False,diagnostics=diagnostics)

# HAVE TO ADJUST IMNN.PY FOR THIS TO WORK
# nholder1.plot_train_output(n, show=True, amount=1000)

# sys.exit("Exit before ABC")

# Generate actual data 
# np.random.seed(113823) # for reproducibility of the "real data"
real_data = generate_data(np.array([theta_fid]), train = None, flatten=flatten)
if nholder1.rescaled:
    print ("Rescaling the real data (hopefully) same as the network. Check this")
    real_data = -1 * np.log(real_data)

# print ('Means of the "real data" generated at fiducial parameters')
# print (np.mean(real_data.reshape(input_shape),axis=0))

# Generate numerical derivative, needed for the generate_data_ABC() func
deriv_Oc = nholder1.calc_derivative_Omega_M()

# # Perform ABC
# A Gaussian prior with mean 0.30, variance 0.01, truncated at 0.1 and 0.6
prior = {'mean': np.array([0.31]), # np.array([0.30])
         'variance': np.array([[0.05]]), # np.array([[0.01]])
         'lower': np.array([0.29]), # np.array([0.25])
         'upper': np.array([0.33])  # np.array([0.35])
         }

draws = int(2.5e3) # on student57 3635/5000 is the maximum amount

print ('Running ABC for %i draws'%draws)
abc = nholder1.do_ABC(n, real_data, prior, draws, show=False, epsilon=None,oneD=True
    ,analytic_posterior=None, param_array=None, at_once=True,save_sims="True")


sys.exit("Exit before PMC")

# # Perform PMC
num_keep = int(1e3)
inital_draws = int(2e3)

nholder1.do_PMC_ABC(n, real_data, prior, inital_draws, num_keep, abc=None, criterion = 0.3
    , show=True,oneD=True, analytic_posterior=None, param_array=None)

def checkNaNs():
    for key in nholder1.data.keys(): 
        if np.isnan(nholder1.data[key]).any(): 
            print (key,'has nans')  


"""
# To continue training
nholder1.eta = 1e-3
nholder1.num_epochs = int(20e3)
nholder1.train_network(n, restart=False,diagnostics=diagnostics)
nholder1.plot_variables(n,show=True,diagnostics=diagnostics)
"""
