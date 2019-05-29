import sys
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
# change to the path where the IMNN git clone is located
# new version of IMNN by Tom
sys.path.insert(-1,'../../../../../IMNNv2/IMNN/')
import IMNN.IMNN as IMNN # make sure the path to the IMNN is given
import IMNN.ABC.ABC as ABC
import IMNN.ABC.priors as priors

sys.path.append('../../../../cosmosis_wrappers/') # change to correct path
from generate_cells_cosmosis import generate_cells, load_cells
import ABC_saved_sims_multiparam

import tqdm
sys.path.insert(-1,'../../../../') # change to path where utils_mrp is located
import utils_mrp_v2 as utils_mrp
import set_plot_sizes # set font sizes

# For making corner plots of the posterior
import corner # Reference: https://github.com/dfm/corner.py/


# for some reason pyccl doesnt work on eemmeer
import pyccl as ccl # for generating weak lensing cross power spectra

"""
Summarizing a weak lensing data vector. Generated with the pyccl module.
The goal is to predict Omega_M and Sigma_8 for a Euclid-like survey.

We assume a redshift distribution given by
    z^alpha * exp(z/z0)^beta
    with alpha=1.3, beta = 1.5 and z0 = 0.65

And assume the photo-z error is Gaussian with a bias is 0.05(1+z)

The weak lensing data vectors are of shape 
    (nbin*nbin/2 + nbin, len(ell))
    i.e., the number of cross/auto correlation spectra and the number of
    ell that is simulated

We use 3 tomographic bins, so we will have 3 Cl auto spectra and 3 cross
We sample it at 100 logarithmically equal spaced points, Thus our weak lensing
data vector will be of shape (6,100)

We flatten these data vectors and feed them to the IMNN,
producing 2 summary statistics: information about Omega_M and Sigma8

Therefore, flatten = True in the nholder object

"""
tf.reset_default_graph()


class nholder(object):
    """
    Class to hold and store all parameters for an IMNN 
    """
    def __init__(self, input_shape, generate_data, theta_fid, delta_theta, n_s, n_train, 
        derivative_fraction, n_train_val, derivative_fraction_val, eta, parameters,
        num_epochs, keep_rate, verbose, dtype, training_dictionary, validation_dictionary, 
        network, version, fromdisk, savedata, noiseless_deriv, flatten, 
        rescale, load_network=False):
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
        fromdisk                     bool    whether to read train/test data from disk
        savedata                     bool    whether to save train/test data to disk
        noiseless_deriv              bool    whether to add noise to derivatives or not
        flatten                      bool    whether to flatten the train/test data
        rescale                      bool    whether to take -1*log() of the data
        load_network                 bool    whether to load a previous network version

        """

        self.unflattened_shape = input_shape
        if flatten:
            print ("Flattening the data",input_shape,"->",[int(np.prod(input_shape))])
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
        self.save_dir = save_dir
        self.savedata = savedata
        self.dataname = 'data'+str(version)

        print (f"Network version {version}")

        if fromdisk:
            self.data = self.load_data_from_disk(self.dataname)
            if self.rescale:
                self.rescaled = True
        else:
            self.data = self.create_data()
            if savedata:
                print (f"Saving data with name: {self.dataname}")
                self.save_data_to_disk(self.dataname)

        # # Theoretical Cl with no noise
        self.Cl_noiseless = cosmosis_cells(*self.theta_fid, self.save_dir)
        # shape (6,200) for 3 tomographic bins

        # Covariance from theoretical Cl
        # Calculate the covariance for every ell, saved as a matrix (100,6,6)
        self.covariance = calculate_covariance(self.Cl_noiseless)

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

        simulator_args = {"train":-self.delta_theta, "flatten":self.flatten,
                        "noiseless_deriv":self.noiseless_deriv, "preload":False,
        "save_dir":save_dir+f"/OMs8{(self.theta_fid-self.delta_theta)[0]}"
        }
        # Perturb lower 
        np.random.seed(seed)
        t_m = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_p)]), None
                    ,simulator_args) 
        # Perturb higher 
        simulator_args = {"train":self.delta_theta, "flatten":self.flatten,
                        "noiseless_deriv":self.noiseless_deriv, "preload":False,
        "save_dir":save_dir+f"/OMs8{(self.theta_fid+self.delta_theta)[0]}"
        }
        np.random.seed(seed)
        t_p = self.generate_data(np.array([theta_fid for i in 
                    range(self.n_train * self.n_p)]), None
                    ,simulator_args)

        # Central
        simulator_args = {"train":None, "flatten":self.flatten,
                        "noiseless_deriv":self.noiseless_deriv, "preload":False,
        "save_dir":save_dir+f"/OMs8{self.theta_fid[0]}"
        }
        np.random.seed(seed)
        t = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_s)]), None
                    ,simulator_args)


        if self.rescale: # take -1*log
            print ("Rescaling data by taking -1*log() ")
            print ("Replacing NaN with 0 ")
            self.rescaled = True
            t_m = -1 * np.log(t_m) 
            # replace NaNs with 0, if they occur
            np.nan_to_num(t_m,copy=False)
            
            t_p = -1 * np.log(t_p)
            np.nan_to_num(t_p,copy=False)
            
            t = -1 * np.log(t)
            np.nan_to_num(t,copy=False)

        # derivative data
        t_d = (t_p - t_m) / (2. * self.delta_theta[np.newaxis,:,np.newaxis])

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
        simulator_args = {"train":-self.delta_theta, "flatten":self.flatten,
                        "noiseless_deriv":self.noiseless_deriv, "preload":False,
        "save_dir":save_dir+f"/OMs8{(self.theta_fid-self.delta_theta)[0]}"
        }
        np.random.seed(seed)
        tt_m = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_p)]), None
                    ,simulator_args)
        # Perturb higher 
        simulator_args = {"train":self.delta_theta, "flatten":self.flatten,
                        "noiseless_deriv":self.noiseless_deriv, "preload":False,
        "save_dir":save_dir+f"/OMs8{(self.theta_fid+self.delta_theta)[0]}"
        }
        np.random.seed(seed)
        tt_p = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_p)]), None
                    ,simulator_args)
        # Central sim
        simulator_args = {"train":None, "flatten":self.flatten,
                        "noiseless_deriv":self.noiseless_deriv, "preload":False,
        "save_dir":save_dir+f"/OMs8{self.theta_fid[0]}"
        }
        np.random.seed(seed)
        tt = self.generate_data(np.array([self.theta_fid for i in 
                    range(self.n_train * self.n_s)]), None
                    ,simulator_args)
        
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
        tt_d = (tt_p - tt_m) / (2. * self.delta_theta[np.newaxis,:,np.newaxis])

        data["validation_data"] = tt 
        data["validation_data_d"] = tt_d

        # for plotting purposes we save the upper/lower separately
        data["x_m_test"], data["x_p_test"] = tt_m, tt_p 

        return data

    def plot_covariance(self, show=False):
        """
        Plot the unperturbed Cl data vector and it's covariance as error bars
        to see whether we have calculated good values, since the other plots don't
        really tell us much about the covariance values.

        The errorbars are the sqrt of the diagonal of the covariance matrix
        """


        Cls = self.Cl_noiseless

        # save the diagonals for each ell index seperately for easy plotting
        diagonals = []
        for i in range(len(self.covariance)):
            diagonals.append(np.diag(self.covariance[i]))
        diagonals = np.array(diagonals) # shape (200,6)

        fig = plt.figure(figsize=(10,10))

        # Plot the auto/cross power spectra
        counter = 0
        for i in range(nbin):
            for j in range(0,i+1):
                ax = plt.subplot2grid((nbin,nbin), (i,j))

                # for the legend
                ax.plot(ells[0], ells[0]*(ells[0]+1)*Cls[counter][0]
                    ,color='white',label=f'{i},{j}')
                ax.legend(frameon=False,loc='upper left')

                # Plot the standard deviation as error bars
                onesigma = np.sqrt(diagonals[:,counter]) # only the diagonal of cov
                ax.errorbar(ells, ells*(ells+1)*Cls[counter]
                    , yerr=ells*(ells+1)*onesigma
                    , fmt='-',lw=1)

                ax.set_yscale('log')
                ax.set_xscale('log')

                counter += 1

                if i == 0 and j == 0:
                    ax.set_ylabel('$\ell  (\ell + 1) C_\ell$')
                if i == nbin-1 and j == 0:
                    ax.set_xlabel('$\ell$')

        plt.suptitle("Determinitistic Cl with diagonal 1sigma error bars")
        
        plt.tight_layout()
        plt.savefig(f'{self.figuredir}errorbars_{self.modelversion}.png')
        if show: plt.show()
        plt.close()

    def plot_sims_spectra(self, Cls, fig,label=None):
        """
        Plot the nbin*(nbin+1)/2 spectra, multiple simulations even

        Assumes Cls is a vector of shape (nsims, 6, 100)
        fig -- the plt.figure to plot it in

        """
        counter = 0
        for i in range(nbin):
            for j in range(0,i+1):
                ax = plt.subplot2grid((nbin,nbin), (i,j), fig=fig)

                # for the legend, plot a single white point
                if self.rescaled:
                    ax.plot(ells[0], Cls[0,counter,0]
                        ,color='white',label=f'{i},{j}')
                else:
                    ax.loglog(ells[0], ells[0]*(ells[0]+1)*Cls[0,counter,0]
                        ,color='white',label=f'{i},{j}')

                # plot all sims
                for sim in range(Cls.shape[0]):
                    if self.rescaled:
                        ax.plot(ells, Cls[sim,counter,:], label=label)
                    else:
                        ax.loglog(ells, ells*(ells+1)*Cls[sim,counter,:],label=label)
                
                ax.legend(frameon=False,loc='upper left')

                counter += 1

                if i == 0 and j == 0:
                    if self.rescaled:
                        ax.set_ylabel(r'$C_\ell$')
                    else:
                        ax.set_ylabel('$\ell  (\ell + 1) C_\ell$')

                if i == nbin-1 and j == 0:
                    ax.set_xlabel('$\ell$')

        return fig
                
    def plot_data(self, show=False):
        """ 
        Plot the data, nrows randomly picked examples

        VARIABLES
        #______________________________________________________________
        show                    bool    whether or not plt.show() is called

        """

        nrows = 10 # amount of random examples

        if self.flatten:
            print ('Plotting data... reshaping the flattened data to %s'%str(self.unflattened_shape))
        else:
            print ('Plotting data...')

        # plot nrows random examples from the simulated train data 
        fig = plt.figure(figsize=(12,12))
        
        Cls = self.data['data'][np.random.randint(0,self.n_train * self.n_s,nrows)].reshape([nrows,*self.unflattened_shape])
        fig = self.plot_sims_spectra(Cls, fig)
        fig.suptitle(f'{nrows} examples from training data, Cl (0,0)')
        plt.savefig(f'{self.figuredir}data_visualization_train_{self.modelversion}.png')
        if show: plt.show()
        plt.close()

        # plot nrows random examples from the simulated test data 
        fig = plt.figure(figsize=(12,12))

        Cls = self.data['validation_data'][np.random.randint(0,self.n_s,nrows)].reshape([nrows,*self.unflattened_shape])
        fig = self.plot_sims_spectra(Cls, fig)
        fig.suptitle(f'{nrows} examples from test data, Cl (0,0)')
        plt.savefig(f'{self.figuredir}data_visualization_test_{self.modelversion}.png')
        if show: plt.show()
        plt.close()

    def plot_data_flattened(self, show=False):
        nrows = 10 # amount of random examples

        if self.flatten:
            print ('Plotting data...')
        else:
            print ('Plotting data, flattening it...')

        # plot nrows random examples from the simulated train data 
        fig, axes = plt.subplots(2, 1, figsize=(12,12))
        
        Cls = self.data['data'][np.random.randint(0,self.n_train * self.n_s,nrows)].reshape([nrows,int(np.prod(self.input_shape))])
        ax = axes[0]

        for sim in range(nrows):
            ax.plot(Cls[sim])
        if self.rescaled:
            pass
        else:
            ax.set_yscale('log')
        ax.set_xlabel('Entry')
        ax.set_title(f'{nrows} data vectors from training data')

        Cls = self.data['validation_data'][np.random.randint(0,self.n_s,nrows)].reshape([nrows,int(np.prod(self.input_shape))])
        ax = axes[1]
        for sim in range(nrows):
            ax.plot(Cls[sim])
        if self.rescaled:
            pass
        else:
            ax.set_yscale('log')
        ax.set_xlabel('Entry')
        ax.set_title(f'{nrows} data vectors from training data')

        ax.set_title(f'{nrows} data vectors from test data')
        plt.savefig(f'{self.figuredir}data_visualization_flat_{self.modelversion}.png')
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

        num_examples = 1 # plot 1 example
        training_index = np.random.randint(0,self.n_train * self.n_p,num_examples)

        fig, ax = plt.subplots(5, 2, figsize = (15, 10), sharex='col')
        # plt.subplots_adjust(wspace = 0, hspace = 0.1)
        plt.subplots_adjust(hspace=0.5)


        if self.flatten:
            print ('Plotting derivatives... reshaping the flattened data to %s'%str(input_shape))
        else:
            print ('Plotting derivatives...')
        
        # Upper example 
        Cls_p = self.data['x_p'][training_index].reshape(num_examples,len(theta_fid),*self.unflattened_shape)
        # shape (num_examples, num_params, ncombinations, len(ells))

        # Plot only the first example 0,0 autocorrelation bin
        Cls_p = Cls_p[0,:,0,:]

        # Cls_p now has shape (num_params , 100) since it is 
        # for all params: the data vector for the upper training image 
        labels =[r'$θ_1$ ($\Omega_M$)', r'$θ_2$ ($\sigma8$)']

        # we loop over them in this plot to assign labels
        for i in range(Cls_p.shape[0]):
            if self.rescaled:
                ax[0, 0].plot(ells, Cls_p[i],label=labels[i])
            else:
                ax[0, 0].loglog(ells, ells*(ells+1)*Cls_p[i],label=labels[i])
        ax[0, 0].set_title('One upper training example, Cl 0,0')
        if self.rescaled:
            ax[0, 0].set_ylabel(r'$C_\ell$')
        else:
            ax[0, 0].set_ylabel(r'$\ell(\ell+1) C_\ell$')
        ax[0, 0].set_xscale('log')
        ax[0, 0].legend(frameon=False)

        Cls_m = self.data['x_m'][training_index].reshape(num_examples,len(theta_fid),*self.unflattened_shape)
        Cls_m = Cls_m[0,:,0,:]
        # loop for labels
        for i in range(Cls_m.shape[0]):
            if self.rescaled:
                ax[1, 0].plot(ells, Cls_m[i])
            else:
                ax[1, 0].loglog(ells, ells*(ells+1)*Cls_m[i])
        ax[1, 0].set_title('One lower training example, Cl 0,0')
        if self.rescaled:
            ax[1, 0].set_ylabel(r'$C_\ell$')
        else:
            ax[1, 0].set_ylabel(r'$\ell(\ell+1) C_\ell$')

        ax[1, 0].set_xscale('log')


        # Plot upper minus lower
        for i in range(Cls_m.shape[0]):
            ax[2, 0].plot(ells, (Cls_p[i]-Cls_m[i]))
        ax[2, 0].set_title('Upper - lower input data: train sample');
        ax[2, 0].set_ylabel(r'$C_\ell (u) - C_\ell (m) $')
        ax[2, 0].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[2, 0].set_xscale('log')


        for i in range(Cls_p.shape[0]):
            ax[3, 0].plot(ells, (Cls_p[i]-Cls_m[i])/(2*delta_theta[i]))
        ax[3, 0].set_title('Numerical derivative: train sample');
        ax[3, 0].set_ylabel(r'$\Delta C_\ell / 2\Delta \theta$')
        ax[3, 0].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[3, 0].set_xscale('log')

        for i in range(Cls_m.shape[0]):
            ax[4, 0].plot(ells, self.data['data_d'][training_index].reshape(
                len(theta_fid),ncombinations,len(ells))[i,0,:]) # plot only 0,0 bin

        ax[4, 0].set_title('Data_d: train sample Cl 0,0');
        ax[4, 0].set_ylabel(r'$\Delta C_\ell / 2\Delta \theta$')
        ax[4, 0].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[4, 0].set_xscale('log')

        ax[-1, 0].set_xlabel(r'$\ell$')


        # Repeat all for test data
        test_index = np.random.randint(self.n_p)

        # Upper example 
        Cls_p = self.data['x_p_test'][test_index].reshape(num_examples,len(theta_fid),*self.unflattened_shape)
        Cls_p = Cls_p[0,:,0,:]

        # we loop over params in this plot to assign labels
        for i in range(Cls_p.shape[0]):
            if self.rescaled:
                ax[0, 1].plot(ells, Cls_p[i],label=labels[i])
            else:
                ax[0, 1].loglog(ells, ells*(ells+1)*Cls_p[i],label=labels[i])
        ax[0, 1].set_title('One upper test example, Cl 0,0')
        if self.rescaled:
            ax[0, 1].set_ylabel(r'$C_\ell$')
        else:
            ax[0, 1].set_ylabel(r'$\ell(\ell+1) C_\ell$')
        ax[0, 1].set_xscale('log')
        ax[0, 1].legend(frameon=False)

        Cls_m = self.data['x_m'][test_index].reshape(num_examples,len(theta_fid),*self.unflattened_shape)
        Cls_m = Cls_m[0,:,0,:]
        # loop for labels
        for i in range(Cls_m.shape[0]):
            if self.rescaled:
                ax[1, 1].plot(ells, Cls_m[i])
            else:
                ax[1, 1].loglog(ells, ells*(ells+1)*Cls_m[i])
        ax[1, 1].set_title('One lower test example, Cl 0,0')
        if self.rescaled:
            ax[1, 1].set_ylabel(r'$C_\ell$')
        else:
            ax[1, 1].set_ylabel(r'$\ell(\ell+1) C_\ell$')

        ax[1, 1].set_xscale('log')


        # Plot upper minus lower
        for i in range(Cls_m.shape[0]):
            ax[2, 1].plot(ells, (Cls_p[i]-Cls_m[i]))
        ax[2, 1].set_title('Upper - lower input data: test sample');
        ax[2, 1].set_ylabel(r'$C_\ell (u) - C_\ell (m) $')
        ax[2, 1].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[2, 1].set_xscale('log')


        for i in range(Cls_m.shape[0]):
            ax[3, 1].plot(ells, (Cls_p[i]-Cls_m[i])/(2*delta_theta[i]))
        ax[3, 1].set_title('Numerical derivative: test sample');
        ax[3, 1].set_ylabel(r'$\Delta C_\ell / 2\Delta \theta$')
        ax[3, 1].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[3, 1].set_xscale('log')

        for i in range(Cls_m.shape[0]):
            ax[4, 1].plot(ells, self.data['validation_data_d'][test_index].reshape(
                len(theta_fid),ncombinations,len(ells))[i,0,:])
        ax[4, 1].set_title('Data_d: test sample Cl 0,0');
        ax[4, 1].set_ylabel(r'$\Delta C_\ell / 2\Delta \theta$')
        ax[4, 1].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')
        ax[4, 1].set_xscale('log')

        ax[-1, 1].set_xlabel(r'$\ell$')


        plt.savefig(f'{self.figuredir}derivatives_visualization_{self.modelversion}.png')
        if show: plt.show()
        plt.close()

    def plot_derivatives_flattened(self, show=False):
        """ 
        Plot the upper and lower perturbed data 
        Good to check if the sample variance is being 
        surpressed. This needs to be done or the network learns very slowly

        VARIABLES
        #______________________________________________________________
        show                    bool    whether or not plt.show() is called

        """

        if self.flatten:
            print ('Plotting data...')
        else:
            print ('Plotting data, flattening it...')

        num_examples = 1 # plot 1 example
        training_index = np.random.randint(0,self.n_train * self.n_p,num_examples)

        fig, ax = plt.subplots(4, 2, sharex='col', figsize = (15, 10))
        # plt.subplots_adjust(wspace = 0, hspace = 0.1)
        plt.subplots_adjust(hspace=0.5)

        # Upper example 
        Cls_p = self.data['x_p'][training_index].reshape([num_examples,int(np.prod(self.input_shape))])
        # shape (num_examples, 600) # For 2 params this will be a bit different

        ax[0, 0].plot(Cls_p[0])
        if not self.rescaled: ax[0, 0].set_yscale('log')

        ax[0, 0].set_title('One upper training example flattened')
        ax[0, 0].set_xlabel('Entry')
        ax[0, 0].set_ylabel(r'$C_\ell$')
        ax[0, 0].legend(frameon=False)

        Cls_m = self.data['x_m'][training_index].reshape([num_examples,int(np.prod(self.input_shape))])
        ax[1, 0].plot(Cls_m[0])
        if not self.rescaled: ax[1, 0].set_yscale('log')

        ax[1, 0].set_title('One lower training example flattened')
        ax[1, 0].set_xlabel('Entry')
        ax[1, 0].set_ylabel(r'$C_\ell$')


        # Plot upper minus lower
        ax[2, 0].plot((Cls_p[0]-Cls_m[0]))
        ax[2, 0].set_title('Upper - lower input data: train sample');
        ax[2, 0].set_xlabel('Entry')
        ax[2, 0].set_ylabel(r'$C_\ell (u) - C_\ell (m) $')
        ax[2, 0].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')


        # for every example
        for i in range(Cls_p.shape[0]):
            ax[3, 0].plot((Cls_p[i]-Cls_m[i])/(2*delta_theta[i]))
        ax[3, 0].set_title('Numerical derivative: train sample');
        ax[3, 0].set_xlabel('Entry')
        ax[3, 0].set_ylabel(r'$\Delta C_\ell / 2\Delta \theta$')
        ax[3, 0].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')


        # Repeat all for test data
        test_index = np.random.randint(self.n_p)

        # Upper example 
        Cls_p = self.data['x_p_test'][test_index].reshape([num_examples,int(np.prod(self.input_shape))])

        ax[0, 1].plot(Cls_p[i])
        if not self.rescaled: ax[0, 1].set_yscale('log')
        ax[0, 1].set_title('One upper training example, flattened')
        ax[0, 1].set_xlabel('Entry')
        ax[0, 1].legend(frameon=False)
        ax[0, 1].set_ylabel(r'$C_\ell$')

        Cls_m = self.data['x_m_test'][test_index].reshape([num_examples,int(np.prod(self.input_shape))])
        ax[1, 1].plot(Cls_m[i])
        if not self.rescaled: ax[1, 1].set_yscale('log')
        ax[1, 1].set_title('One lower training example, flattened')
        ax[1, 1].set_xlabel('Entry')
        ax[1, 1].set_ylabel(r'$C_\ell$')


        # Plot upper minus lower
        for i in range(Cls_m.shape[0]): # for ever example
            ax[2, 1].plot((Cls_p[i]-Cls_m[i]))
        ax[2, 1].set_title('Upper - lower input data: train sample');
        ax[2, 1].set_xlabel('Entry')
        ax[2, 1].set_ylabel(r'$C_\ell (u) - C_\ell (m) $')
        ax[2, 1].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')


        for i in range(Cls_m.shape[0]):
            ax[3, 1].plot((Cls_p[i]-Cls_m[i])/(2*delta_theta[i]))
        ax[3, 1].set_title('Numerical derivative: train sample');
        ax[3, 1].set_xlabel('Entry')
        ax[3, 1].set_ylabel(r'$\Delta C_\ell / 2\Delta \theta$')
        ax[3, 1].axhline(xmin = 0., xmax = 1., y = 0.
            , linestyle = 'dashed', color = 'black')

        plt.savefig(f'{self.figuredir}derivatives_visualization_flat_{self.modelversion}.png')
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

        raise ValueError("deprecated, see create_data")

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


    def save_data_to_disk(self, name):
        """
        Save current train/test data to disk. In the directory ./preloaded_data_sigma8/

        See also load_data_from_disk

        """
        for key in tqdm.tqdm(self.data.keys(),desc="Saving data"):
            np.save(f'./preloaded_data_sigma8/{name}_{key}.npy', self.data[key])

    def load_data_from_disk(self, name):
        """
        Load data from disk. Looks in the directory ./preloaded_data_sigma8/

        See also load_data_from_disk

        """
        data = dict()
        for key in tqdm.tqdm(['data', 'data_d', 'x_m', 'x_p', 'validation_data'
                            , 'validation_data_d', 'x_m_test', 'x_p_test'],desc="Loading data from disk"):
            data[key] = np.load(f'./preloaded_data_sigma8/{name}_{key}.npy')

        return data

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
        # Numerical derivative
        deriv_OM = (Cls_t - self.Cl_noiseless) / domega

        return deriv_OM # shape (6,100)

    def do_ABC(self, n, real_data, prior, draws, show=False, epsilon=None, oneD=True
        ,analytic_posterior = None, param_array = None):
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

        RETURNS
        #_____________________________________________________________
        abc                     class   abc object (see ABC.py)         
    
        """

        Gaussprior = priors.TruncatedGaussian(prior["mean"],prior["variance"],prior["lower"]
                                        ,prior["upper"])

        abc = ABC.ABC(real_data = real_data, prior = Gaussprior, sess=n.sess
            , get_compressor=n.get_compressor, simulator=generate_data_ABC, seed=None
            , simulator_args=None, dictionary = self.validation_dictionary)

        # actually perform ABC
        abc.ABC(draws=draws, at_once=True, save_sims=None, MLE=True)

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
        , show=False, oneD=True,analytic_posterior = None, param_array = None):
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
            , get_compressor=n.get_compressor, simulator=generate_data_ABC, seed=None
            , simulator_args=None, dictionary = self.validation_dictionary)

        else: # Use the abc object from the do_ABC() function
            print ("Starting PMC from previous ABC or PMC result")
            restart = False
    
        all_epsilon = abc.PMC(draws = draws, posterior = num_keep
            , criterion = criterion, at_once = True, save_sims = None, MLE = True) 

        # plot output samples and histogram of the accepted samples in 1D
        def plot_samples_oneD():
            fig, ax = plt.subplots(3, 2, sharex = 'col', figsize = (12, 12))
            plt.subplots_adjust(hspace = 0)
            
            # Plot the accepted/rejected samples
            ax[0,0].scatter(abc.PMC_dict["parameters"][:,0] # x
                , abc.PMC_dict["summaries"][:,0] # y
                , s = 1, alpha = 1.0, label = "Accepted samples", color = "C0") 
                        
            ax[0,0].axhline(abc.summary[0,0]
                , color = 'black', linestyle = 'dashed', label = "Summary of observed data")
            
            ax[0,0].legend(frameon=False)
            ax[0,0].set_ylabel('First network output', labelpad = 0)
            ax[0,0].set_xlim([prior["lower"][0], prior["upper"][0]])
            # ax[0].set_xticks([])
            # ax[1].set_xlabel(r"$\theta_1 = \Omega_M$")

            # Plot the accepted/rejected samples for the second network output
            ax[1,0].scatter(abc.PMC_dict["parameters"][:,0] # x
                , abc.PMC_dict["summaries"][:,1] # y
                , s = 1, alpha = 1.0, label = "Accepted samples", color = "C0") 
           
            ax[1,0].axhline(abc.summary[0,1]
                , color = 'black', linestyle = 'dashed', label = "Summary of observed data")
            
            ax[1,0].legend(frameon=False)
            ax[1,0].set_ylabel('Second network output', labelpad = 0)
            ax[1,0].set_xlim([prior["lower"][0], prior["upper"][0]])

            # plot the posterior
            ax[2,0].hist(abc.PMC_dict["parameters"][:,0], np.linspace(prior["lower"][0], prior["upper"][0], 100), histtype = u'step', density = True, linewidth = 1.5, color = "C0", label = "PMC posterior");
            ax[2,0].axvline(abc.MLE[0,0], linestyle = "dashed", color = "black", label = "(Gaussian) MLE")
            ax[2,0].set_xlim([prior["lower"][0], prior["upper"][0]])
            ax[2,0].set_ylabel('$\\mathcal{P}(\\theta_1|{\\bf d})$')
            # ax[1].set_yticks([])
            # ax[1].set_xticks([])
            ax[2,0].set_xlabel(r"$\theta_1 = \Omega_M$")

            # Theta-fid
            ax[2,0].axvline(theta_fid[0], linestyle = "dashed", label = "$\\theta_{fid}$")

            leg = ax[2,0].legend(frameon = False,loc='best',fontsize=14)
            # To set transparency in the legend
            for lh in leg.legendHandles: 
                lh.set_alpha(1)

            # Then repeat the whole ordeal for the second parameter
            ##################### SECOND PARAMETER, RIGHT COLUMN ############
            # Plot the accepted/rejected samples
            # Parameter 2: sigma8, first network output.
            ax[0,1].scatter(abc.PMC_dict["parameters"][:, 1]
                , abc.PMC_dict["summaries"][:,0]
                , s = 1, alpha = 1.0, label = "Accepted samples", color = "C0")
            ax[0,1].axhline(abc.PMC_dict["summary"][0,0], color = 'black', linestyle = 'dashed', label = "Summary of observed data")
            # ax[0,1].legend(frameon=False)
            ax[0,1].set_ylabel('First network output', labelpad = 0)
            ax[0,1].set_xlim([prior["lower"][1], prior["upper"][1]])
            # ax[0,1].set_xticks([])
            
            # second network output
            ax[1,1].scatter(abc.PMC_dict["parameters"][:, 1]
                , abc.PMC_dict["summaries"][:, 1], s = 1, alpha = 1.0, label = "Accepted samples", color = "C0")
            ax[1,1].axhline(abc.PMC_dict["summary"][0, 1], color = 'black', linestyle = 'dashed', label = "Summary of observed data")
            # ax[1,1].legend(frameon=False)
            ax[1,1].set_ylabel('Second network output', labelpad = 0)
            ax[1,1].set_xlim([prior["lower"][1], prior["upper"][1]])

            # plot the posterior
            ax[2,1].hist(abc.PMC_dict["parameters"][:, 1], bins=hbins, histtype = u'step', density = True, linewidth = 1.5, color = "C6", label = "ABC posterior");
            ax[2,1].axvline(abc.PMC_dict['MLE'][0, 1], linestyle = "dashed", color = "black", label = "(Gaussian) MLE")
            ax[2,1].set_xlim([prior["lower"][1], prior["upper"][1]])
            ax[2,1].set_ylabel('$\\mathcal{P}(\\mu|{\\bf d})$')
            # ax[2,1].set_yticks([])
            # ax[2,1].set_xticks([])
            ax[2,1].set_xlabel(r"$\theta_2 = \sigma_8$")
            
            fig.suptitle(f"Epsilon = {epsilon}")

            plt.savefig(f'{self.figuredir}ABC_{self.modelversion}_1D.png')
            if show: plt.show()
            plt.close()

        # plot approximate posterior of the accepted samples in 2D
        def plot_samples_twoD():
            hist_kwargs = {} # add kwargs to give to matplotlib hist funct
            fig, ax = plt.subplots(2, 2, figsize = (10, 10))
            fig = corner.corner(abc.PMC_dict["parameters"][:, :], fig=fig, truths = truths
                , labels=[r'$\\theta_1 = \Omega_M$ ', r'$\theta_2 = \sigma_8$']
                , plot_contours=True, range=[(prior["lower"][0],prior["upper"][0]), (prior["lower"][1],prior["upper"][1])], hist_kwargs=hist_kwargs)
            # fig.suptitle('Approximate posterior after ABC for %i draws'%draws)
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



    def investigate_parametric_model(self, n, real_data, prior, draws=5):
        """
        See if we can fit a parametric model to the summaries as 
        function of omega_m and sigma_8

        Draw 'draws' samples uniformly from a prior
        """

        all_omega_m = np.random.uniform(prior["lower"][0],prior["upper"][0],size=draws)
        all_sigma8 = np.random.uniform(prior["lower"][0],prior["upper"][0],size=draws)

        # for some reason this has to be run first to get the correct fisher info
        n.sess.run(n.get_compressor)
        summary = n.sess.run(
        "IMNN/summary:0",
        feed_dict={**self.validation_dictionary, **{"data:0": real_data}})

        fisher = n.sess.run("fisher:0")

        all_theta = []
        for om in all_omega_m:
            for s8 in all_sigma8:
                all_theta.append([om,s8])

        all_theta = np.array(all_theta)
        sims = generate_data_ABC(all_theta)

        summaries = n.sess.run(
            "IMNN/summary:0",
            feed_dict={**self.validation_dictionary, **{"data:0": sims}})
        if MLE:
            MLEs = n.sess.run(
                "MLE:0",
                feed_dict={**self.validation_dictionary, **{"data:0": sims}})

        differences = summaries - summary
        distances = np.sqrt(
            np.einsum(
                'ij,ij->i',
                differences,
                np.einsum(
                    'jk,ik->ij',
                    fisher,
                    differences)))

        ABC_dict = dict()
        ABC_dict["summary"] = summary
        ABC_dict["fisher"] = fisher
        ABC_dict["parameters"] = parameters
        ABC_dict["summaries"] = summaries
        ABC_dict["differences"] = differences
        ABC_dict["distances"] = distances
        if MLE:
            ABC_dict["MLE"] = MLEs

        return ABC_dict


#################################
# HELPER FUNCTIONS to generate data
#################################

def cosmosis_cells(Omega_M, sigma8, save_dir):
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
    generate_cells(save_dir, nbin, zmax, dz, ell_min, ell_max, n_ell, sigma8=sigma8
        , alpha=alpha, beta=beta, z0=z0, sigz=sigz, ngal=ngal, bias=bias
        , omega_m=Omega_M, h0=h,omega_b=Omega_b, n_s=n_s, A_s=A_s,w=w0)

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

def indexed_CL(Cls):
    """
    For indexing the C_l's, normally they are given as (ncombinations,len(ells))
    this transforms them to CL[i,j,ell_index]

    i and j are tomographic indices ell_index is index of ell number

    Input:
        Cls -- np.array -- shape(ncombinations,len(ells))

    Returns 
        CL -- np.array -- shape (nbin,nbin,len(ells)), to access any bin,ell combination

    """
    CL = np.zeros((nbin,nbin, Cls.shape[1]))
    counter = 0
    for i in range(nbin):
            for j in range(0,i+1):
                CL[i,j,:] = Cls[counter]
                counter += 1
    # symmetric
    for i in range(nbin):
        for j in range(nbin):
            CL[i,j,:] = CL[j,i,:]

    return CL

def covariance_takada_jain(CLS_obs):
    """
    Calculates the covariance matrices 
    corresponding to all ells. 

    Input:
    CLS_obs -- array -- shape (ncombinations,ncombinations) easy indexed version
                        of the Cls with added shape noise.

    Returns
    covariance 
    """
    delta_l = 1
    Modes_per_bin = fsky*(2*ells+1)*delta_l

    # for a particular l, covariance is (ncombinations,ncombinations)
    covariance = np.zeros((len(ells),ncombinations,ncombinations))
    index1 = 0 # first index of covariance matrix (row index)
    index2 = 0 # second index of covariance matrix (column index)

    for i in range(nbin):
        for j in range(0, i+1):
                index2 = 0
                for m in range(nbin):
                    for n in range(0, m+1):
                        # taka and jain formula for Gaussian covariance
                        covariance[:,index1,index2] = (CLS_obs[i,m] * CLS_obs[j,n]
                                                     + CLS_obs[i,n] * CLS_obs[j,m] ) / Modes_per_bin                            
                        index2 += 1

                index1 += 1

    return covariance
    
def calculate_covariance(Cls):
    """
    Calculate the Gaussian covariance according to https://arxiv.org/pdf/0810.4170.pdf

    First add shape noise with def calculate_Cls_obs
    Then calculate the Gaussian covariance with one equation

    Returns
    cov_matrices -- np.array -- shape (100,6,6) covariance matrix for every ell

    """

    # Cls with added shape noise
    Cls_obs = calculate_Cls_obs(Cls)

    # make an easy indexed version of this
    CLS_obs = indexed_CL(Cls_obs)

    # Calculate the (6,6) covariance matrix for every ell
    cov_matrices = covariance_takada_jain(CLS_obs)

    return cov_matrices
    
def add_variance(Cls_original, covariance):
    """
    Add Gaussian variance to the original Cls

    Cls_original -- (6,100) array of noiseless angular power spectra
    covariance   -- (100,6,6) array containing the cov matrix for every ell 
    
    Only have to repeat this function to add noise to the determinitistic Cls_original
    """

    # The (little bit slower?) way; loop over ell
    Cls_perturbed = np.zeros(Cls_original.shape)
    for ell_index, ell in enumerate(ells):
        Cls_perturbed[:,ell_index] =  np.random.multivariate_normal(Cls_original[:,ell_index]
                                                                    ,covariance[ell_index])
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
    Cls = cosmosis_cells(Omega_M, sigma8)
        
    # The covariance for every ell for this Cl
    covariance = calculate_covariance(Cls)

    return Cls_original, covariance


def generate_data(θ, seed=None, simulator_args=None):
    """
    Holder function for the generation of the Cls
    
    θ = vector of lists of [Omega_M,sigma8]'s to produce a Cl for
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

    if simulator_args is None:
        # Default values
        simulator_args = dict()
        simulator_args["train"] = None
        simulator_args["flatten"] = True 
        simulator_args["preload"] = False
        simulator_args["noiseless_deriv"] = False
        # Default in this directory, overwritten every time
        simulator_args["save_dir"] = f"/net/reusel/data1/osinga/master_research_project/saved_data/cosmosis/generate_cells_test{version}"

    train = simulator_args["train"]
    preload = simulator_args["preload"]
    noiseless_deriv = simulator_args["noiseless_deriv"]
    save_dir = simulator_args["save_dir"]
    flatten = simulator_args["flatten"]    

    if not flatten:
        raise ValueError("TODO: Unflattened data still to be implemented")

    if preload:
        raise ValueError("todo: divide between train/test data so they are not identical")

    if preload and (θ[:,0] == θ[0,0]).all():
        if train is not None:
            perturb_param1 = np.array([train[0]])
            θ_first_param = θ[0,0] + perturb_param1
            print (f"Checking disk for saved data with Omega_M = {θ_first_param}")
            try:
                all_Cls = np.load(f'./preloaded_data_sigma8/Omega_M_{θ_first_param}')
                return all_Cls
            except FileNotFoundError:
                print ('File not found.')

        else:
            Omega_M = θ[0,0]
            print (f"Checking disk for saved data with Omega_M = {Omega_M}")
            try:
                all_Cls = np.load(f'./preloaded_data_sigma8/Omega_M_{Omega_M}')
                return all_Cls
            except FileNotFoundError:
                print ('File not found')


    def helper_func(θ):
        """
        Generates noisy simulations at θ = vector of lists of [Omega_M]'s
        Called once if train = None, called twice if not
        """
        if (θ[:,0] == θ[0,0]).all() and (θ[:,1] == θ[0,1]).all():
            Omega_M, sigma8 = θ[0,0], θ[0,1]
            print (f"List of parameters contains all the same parameters, O_M={Omega_M}, sigma8 = {sigma8}")
            
            # Something weird makes it so Omega_M is sometimes 0 in PMC, if that happens,
            # we return garbage, to make sure ABC/PMC knows that it is incorrect
            if Omega_M == 0:
                print (f"For some reason Omega_M = 0, we return garbage")
                return np.ones( (len(θ),ncombinations*len(ells)) )*1e20 # shape (num_simulations, ncombinations*len(ells))

            # generate cross power spectra Cl
            Cls = cosmosis_cells(Omega_M,sigma8, save_dir)

            # Calculate the covariance for every ell, have to do this before flattening
            covariance = calculate_covariance(Cls)

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
                    all_Cls = [Cls_original.flatten() for i in range(len(θ))]
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
            for Omega_M, sigma8 in tqdm.tqdm(θ,desc="Generating one at a time"): 
                # Can in theory be done in parallel for all different params
                
                # Something weird makes it so Omega_M is sometimes 0 in PMC, if that happens,
                # we return garbage, to make sure ABC/PMC knows that it is incorrect
                if Omega_M == 0 or sigma8 == 0:
                    print (f"For some reason Omega_M or sigma8 = 0, we return garbage")
                    return np.ones( (len(θ),ncombinations*len(ells)) )*1e20 

                Cls = cosmosis_cells(Omega_M, sigma8, save_dir)

                # Calculate the covariance for every ell, have to do this before flattening
                covariance = calculate_covariance(Cls)

                # Perturb the original Cl with
                # a 1D Gaussian with std=sqrt(covariance)
                Cls = add_variance(Cls, covariance)

                all_Cls.append(Cls.flatten()) # flatten the Cl data

        # if not preload: np.save(f'./preloaded_data_sigma8/Omega_M_{Omega_M}', np.asarray(all_Cls))

        return np.asarray(all_Cls) # shape (num_simulations, ncombinations*len(ells))

    if train is not None: # generate derivatives, with perturbed thetas, noise free
        
        perturb_param1 = np.array([train[0],0])
        θ_first_param = θ + perturb_param1

        perturb_param2 = np.array([0,train[1]])
        θ_second_param = θ + perturb_param2

        # the upper/lower of the first parameter
        all_Cls_first_param = helper_func(θ_first_param)

        # then the upper/lower of the second parameter
        all_Cls_second_param = helper_func(θ_second_param)

        # Return it as an array of shape (num_sim,num_params,length_vector)
        all_Cls_first_param = all_Cls_first_param.reshape(
                                        len(θ),1,all_Cls_first_param.shape[1])
        all_Cls_second_param = all_Cls_second_param.reshape(
                                        len(θ),1,all_Cls_second_param.shape[1])
        all_Cls = np.concatenate([all_Cls_first_param,all_Cls_second_param],axis=1) 

        # if not preload: np.save(f'./preloaded_data_sigma8/Omega_M_{θ_first_param[0,0]}_Sigma8_{θ_second_param}'
        #                        , all_Cls)

        return all_Cls # shape (num_simulations, num_params, ncombinations*len(ells)

    else: # generate simulations at value
        return helper_func(θ)

def generate_data_ABC(θ, seed=None, simulator_args=None):
    """
    Holder function for the generation of the Cls for ABC
    This function simply calls the generate_data function and returns the 
    rescaled version of the Cells that are produced by that function
    
    θ = vector of lists of [Omega_M]'s to produce a Cl for
    
    simulator_args:
    Dictionary containing keyword arguments:
        train = either None or an array of [delta_theta1,delta_theta2] for generating
                the upper and lower derivatives
        flatten -- not used with only 1 param, default is flatten
        preload = True / False, True if we can load the data from disk
        noiseless_deriv = True/False, whether to add noise to the upper/lower simulations
            NOTE THAT THE RANDOM SEED SHOULD BE SET IF WE WANT TO ADD NOISE TO THESE
        save_dir -- where to save the data generated by cosmosis


    Returns the weak lensing data vector flattened to use as input for the IMNN
            shape (num_simulations=len(θ), length of Cl vector)
    """
    all_Cls = generate_data(θ, seed, simulator_args)
        
    if nholder1.rescaled:
        print ("Rescaling the data (hopefully) same as the network")
        print ("Check if the rescaling went correctly")
        all_Cls = -1 * np.log(all_Cls) 

    return all_Cls # shape (num_simulations, ncombinations*len(ells))

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
    if flatten:
        shape = int(np.prod(input_shape))
    else:
        raise ValueError("Todo: think of another network")
        shape = input_shape[-1]

    # Input layer
    with tf.variable_scope("layer_1"): 
        weights = tf.get_variable("weights", shape = [shape + 1
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

        weights = tf.get_variable("weights", shape = (hidden_layers[-1]+1, n_summaries), initializer = tf.variance_scaling_initializer())
        output = tf.identity(tf.matmul(tf.concat([output, tf.ones(dtype = tf.float32, shape = (tf.shape(data)[0], 1))], axis = 1), weights, name = "multiply"), name = "output")
        # NO DROP-OUT in the last layer


    return output

# #######################################
# COSMOSIS PARAMETERS 
#########################################
initial_version = 4 # Network parameter, the version of the network
save_dir = f'/net/reusel/data1/osinga/master_research_project/saved_data/cosmosis/twoparam/3bins{initial_version}'

# nbin = number of tomographic bins
nbin, zmax, dz = 3, 2.0, 0.002 # 1000 samples in redshift
ell_min, ell_max, n_ell = 50, 1000, 100 # 100 samples in ell

# Parameters for ccl to calculate the Euclid n(z) distribution
nz = 1000 #redshift resolution
zmin = 0.
zmax = 2.
z = np.linspace(zmin,zmax,nz)
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
# The input shape is a 2D vector of shape (nbin,len(ells))
input_shape = [ncombinations,len(ells)] # But we flatten it later

theta_fid = np.array([0.315,0.811]) # Omega_M, sigma8

# delta_theta = np.array([0.11]) # perturbation values
delta_theta = np.array([0.02, 0.02])

n_s = 1000 # number of simulations used at a time to approximate the covariance 
n_train = 5 # splits, for if it doesnt fit into memory
n_train_val = 1 # validation splits, not sure why we would do > 1 ever

# use less simulations for numerical derivative
derivative_fraction = 1.0 # fraction of n_s
# use less simulations for numerical derivative of test data
derivative_fraction_val = 0.2 # fraction of n_s

eta = 1e-3 # learning rate
num_epochs = int(1e3) 
keep_rate = 1.0 # 1 minus the dropout
verbose = 0

fromdisk = False #
savedata = True # save data under ./preloaded_data_sigma8/data{modelversion}

# MLP
# hidden_layers = [1024, 512, 256, 128, 128]
hidden_layers = [256,256] 
# activation parameter (e.g., for leaky relu: Slope of the activation function at x < 0 )
actparam = 0.01
dtype = 32 # float32

noiseless_deriv = False # whether to not add noise to upper/lower simulations
flatten = True # Flatten (6,100) to (600,)
rescale = True # take -1*log() of the data


# For building the network, and defining number of summaries
n_summaries = 2
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
        ,validation_dictionary, network, version, fromdisk, savedata, noiseless_deriv, flatten
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
# sys.exit("TODO: Have to cut off the last part of the power spectrum")
# sys.exit("The error bars are too large for ell>10^3")

# reload(ABC_saved_sims)

# Generate actual data 
# np.random.seed(113823) # for reproducibility of the "real data"
real_data = generate_data(np.array([theta_fid]), None, {"train":None, "flatten":flatten,
    "noiseless_deriv":noiseless_deriv, "preload":False, "save_dir":save_dir+f"/real_data"})
if nholder1.rescaled:
    print ("Rescaling the real data (hopefully) same as the network. Check this")
    real_data = -1 * np.log(real_data)

# OLD: Generate numerical derivative, needed for the generate_data_ABC() func
# OLD: deriv_Om = nholder1.calc_derivative_Omega_M()

# # Perform ABC
# A Gaussian prior with mean 0.30, variance 0.01, truncated at 0.1 and 0.6
prior = {'mean': np.array([0.30,0.805]),
         'variance': np.array([[0.01,0],[0,0.01]]), # cov matrix
         'lower': np.array([0.27,0.69]),
         'upper': np.array([0.34,0.91]) 
         }

# draws = int(1e3)

save_sims = '/net/reusel/data1/osinga/master_research_project/saved_data/cosmosis/twoparam/apriori/3bins/ells1000'
simnames = ['ap1']#,'ap2']
draws = int(10e3) # must correspond to number of sims in simnames

# loadparams = ['omega_m', 'sigma_8']
# # Perform ABC with saved simulations
# abc = ABC_saved_sims_multiparam.ABC_saved(nholder1, n, save_sims, simnames, real_data, nbin
#                 , sn, nzs, ells, loadparams
#                 , at_once=True, draws=draws, MLE=True, notebook=False, rescale=rescale)
# # plot the results
# ABC_saved_sims_multiparam.plot_ABC_2params(abc, nholder1, theta_fid, prior, oneD='both', hbins=30, epsilon=None,show=False)

# sys.exit("Exit before PMC")
# # Perform PMC
num_keep = int(50)
inital_draws = int(100)

pmc = nholder1.do_PMC_ABC(n, real_data, prior, inital_draws, num_keep, abc=None, criterion = 0.3
    , show=False,oneD='both', analytic_posterior=None, param_array=None)

def checkNaNs():
    for key in nholder1.data.keys(): 
        if np.isnan(nholder1.data[key]).any(): 
            print (key,'has nans') 

def save_ABC_results(nholder, abc): 
    for key in abc.keys():
        np.save(f'./preloaded_data_sigma8/ABC_results/abc{nholder.modelversion}{key}',abc[key])




"""
# To continue training
nholder1.eta = 1e-3

nholder1.num_epochs = int(0.5e3)
nholder1.train_network(n, restart=False,diagnostics=diagnostics)
nholder1.plot_variables(n,show=True,diagnostics=diagnostics)
"""

# """
# Look at the output:
allparams, some_sims = ABC_saved_sims_multiparam.load_sims(save_sims, simnames, nbin, sn, nzs, ells, at_once=True, it=None, numsims=100
    , rescale=True, addnoise=True, cutoff=None, loadparams=['omega_m','sigma_8'])




# """