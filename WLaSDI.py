import numpy as np
import numpy.linalg as LA
import wsindy as ws
#import LaSDI.pywsindy as ws
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, Rbf
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from itertools import combinations_with_replacement
import time

class WLaSDI:
    """
    WLaSDI class for data-driven ROM. Functions: train_dynamics approximates dynamical systems of the latent-space. 
                                                generate_FOM uses an initial condition and parameter values to generate a new model
    NOTE: To avoid errors, make sure to set NN = True for use with autoencoder.
    
    Inputs:
       encoder: either neural network (with pytorch) or matrix (LS-ROM)
       decoder: either neural network (with pytorch) or matrix (LS-ROM)
       NN: Boolean on whether a NN is used
       device: device NN is on. Default 'cpu', use 'cuda' if necessary
       Local: Boolean. Determines Local or Global DI (still in progress)
       Coef_interp: Boolean. Determines method of Local DI
       nearest_neigh: Number of nearest neigh in Local DI
       Coef_interp_method: Either interp2d or Rbf method for coefficient interpolation.
    """
    def __init__(self, encoder, decoder, NN = False, device = 'cpu', Local = False, Coef_interp = False, nearest_neigh = 4, Coef_interp_method = None, plot_fname = 'latent_space_dynamics.png'):
        
        self.Local = Local
        self.Coef_interp = Coef_interp
        self.nearest_neigh = nearest_neigh
        self.NN = NN
        self.plot_fname = plot_fname
        if Coef_interp == True:
            if Coef_interp_method == None:
                print('WARNING: Please specify an interpolation method either interp2d or Rbf')
            else:
                self.Coef_interp_method = Coef_interp_method
            if nearest_neigh <4:
                print('WARNING: More minimum 4 nearest neighbors required for interpolation')
                return
        if NN == False:
            self.IC_gen = lambda params: np.matmul(encoder, params)
            self.decoder = lambda traj: np.matmul(decoder, traj.T)
            
        else:
            import torch
            self.IC_gen = lambda IC: encoder(torch.tensor(IC).to(device)).cpu().detach().numpy()
            self.decoder = lambda traj: decoder(torch.tensor(traj.astype('float32')).to(device)).cpu().detach().numpy()
            
        return
    
    def train_dynamics(self, ls_trajs, training_values, t, normal = 1, degree = 1, LS_vis = True, gamma = 0, threshold = 0, L = 30, overlap = 0.5, useGLS = 1e-12):
        """
        Approximates the dynamical system for the latent-space. Local == True, use generate_FOM. 
        
        Inputs:
           ls_trajs: latent-space trajectories in a list of arrays formatted as [time, space]
           training_values: list/array of corresponding parameter values to above
           dt: time-step used in FOM
           normal: normalization constant. Default as 1
           LS_vis: Boolean to visulaize a trajectory and discovered dynamics in the latent-space. Default True

           WSINDy parameters
           L: test function support
           overlap: how much 2 consecutive test functions overlap. 
           gamma: regularization param
           threshold: sparsification param (always set to 0)

        """
        self.training_values = training_values
        self.ls_trajs = ls_trajs
        self.t = t
        self.normal = normal
        self.degree = degree
        self.LS_vis = LS_vis
        self.gamma = gamma
        self.threshold = threshold
        self.L = L
        self.overlap = overlap
        self.useGLS = useGLS
        
        data_LS = []
        for traj in self.ls_trajs:
            data_LS.append(traj/self.normal)
        self.data_LS = data_LS

    
    
        if self.Local == False:
            model = ws.wsindy(polys=np.arange(0, degree+1), multiple_tracjectories=True, ld = threshold, gamma = gamma, useGLS = self.useGLS)
            time = [self.t]
            for i in range(1, len(data_LS)):
                time.append(self.t)

            model.getWSindyUniform(data_LS, time, L = L, overlap=overlap)
            self.model = model
            if LS_vis == True:
                if self.NN == True:
                    DcTech = 'LaSDI-NM Latent-Space Visualization'
                    DcTech = 'Latent-Space Dynamics by Nonlinear Compression'
                else:
                    DcTech = 'LaSDI-LS Latent-Space Visualization'
                    DcTech = 'Latent-Space Dynamics by Linear Compression'
                fig = plt.figure()
                fig.set_size_inches(9,6)
                #ax = plt.axes()
                #ax.set_title(DcTech)
                labels = {'orig': 'Latent-Space Trajectory', 'new': 'Approximated Dynamics'}
                for dim in range(data_LS[-1].shape[1]):
                    plt.plot(t[:-1], data_LS[-1][:-1,dim], alpha = .5, label = labels['orig'])
                    labels['orig'] = '_nolegend_'
                plt.gca().set_prop_cycle(None)

                new = model.simulate(x0 = data_LS[-1][0], t_span = np.array([t[0], t[-1]]), t_eval = self.t)
                for dim in range(data_LS[-1].shape[1]):
                    plt.plot(t, new[:,dim], '--', label = labels['new'])
                    labels['new'] = '_nolegend_'
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Magnitude')
                plt.tight_layout()
                #plt.savefig('figures/resimu', transparent = True)
            return model.coef 
        elif self.Coef_interp == True:
            #print("Local approach WITH SINDy coefficient interpolation")
            if self.Coef_interp_method == None:
                print('WARNING: Please specify an interpolation method either interp2d or Rbf')
            model_list = []
           
            for i, _ in enumerate(training_values):
                model = ws.wsindy(polys=np.arange(0, self.degree+1), multiple_tracjectories=True, ld=threshold, gamma=gamma, useGLS = self.useGLS)
                model.getWSindyUniform([data_LS[i]], [t], L = L, overlap=overlap)
                model_list.append(model.coef)
                self.tags = model.tags
                if LS_vis == True:
                    if self.NN == True:
                        DcTech = 'LaSDI-NM Latent-Space Visualization'
                        DcTech = 'Latent-Space Dynamics by Nonlinear Compression'
                    else:
                        DcTech = 'LaSDI-LS Latent-Space Visualization'
                        DcTech = 'Latent-Space Dynamics by Linear Compression'
                    fig = plt.figure()
                    fig.set_size_inches(9,6)
                    ax = plt.axes()
                    ax.set_title(DcTech)
                    labels = {'orig': 'Latent-Space Trajectory', 'new': 'Approximated Dynamics'}
                    for dim in range(data_LS[-1].shape[1]):
                        plt.plot(t[:-1], data_LS[i][:-1,dim], alpha = .5, label = labels['orig'])
                        labels['orig'] = '_nolegend_'
                    plt.gca().set_prop_cycle(None)

                    new = model.simulate(data_LS[i][0], t_span = np.array([t[0], t[-1]]), t_eval = self.t)
                    for dim in range(data_LS[-1].shape[1]):
                        plt.plot(t, new[:,dim], '--', label = labels['new'])
                        labels['new'] = '_nolegend_'
                    ax.legend()
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Magnitude')
                    #plt.savefig(self.plot_fname)
            self.model_list = model_list
            return self.model_list
        else:
            print("Local approach WITHOUT SINDy coefficient interpolation")
            return 
        
    
        
    def generate_ROM(self,pred_IC,pred_value,t, epsilon = 1, function='gaussian'):
        """
        Takes initial condition in full-space and associated parameter values and generates forward in time using the trained dynamics from above.
        Inputs:
            pred_IC: Initial condition of the desired simulation
            pred_value: Associated parameter values
            t: time stamps corresponding to training FOMs
        """
        IC = self.IC_gen(pred_IC)
        if self.Local == False: # Global approach
            self.latent_space_recon = self.normal*self.model.simulate(IC/self.normal, np.array([t[0], t[-1]]), t)
            FOM_recon = self.decoder(self.latent_space_recon)
            if self.NN == False:
                return FOM_recon.T
            else:
                return FOM_recon
        else: # Local approach
            training_time_start = time.time()
            dist = np.empty(len(self.training_values))
            for iii,P in enumerate(self.training_values):
                dist[iii]=(LA.norm(P-pred_value))

            k = self.nearest_neigh
            dist_index = np.argsort(dist)[0:k]
            self.dist_index = dist_index

            if self.Coef_interp == False: # WITHOUT SINDy coefficient interpolation
                local = []
                for iii in dist_index:
                    local.append(self.data_LS[iii])
                model = ws.wsindy(polys=np.arange(0, self.degree+1), multiple_tracjectories=True, ld=self.threshold, gamma=self.gamma, useGLS = self.useGLS)
                t_val = [self.t]
                for i in range(1, len(local)):
                    t_val.append(self.t)

                #using uniform grid
                model.getWSindyUniform(local, t_val, L = self.L, overlap = self.overlap )

                #using adaptive grid
                #model.getWsindyAdaptive(local, t_val, K = 200)

                self.training_time = time.time()-training_time_start
                self.latent_space_recon = self.normal*model.simulate(IC/self.normal, np.array([t[0], t[-1]]), t)
                FOM_recon = self.decoder(self.latent_space_recon)
                
                if self.LS_vis == True:
                    if self.NN == True:
                        DcTech = 'LaSDI-NM Latent-Space Visualization'
                        DcTech = 'Latent-Space Dynamics by Nonlinear Compression'
                    else:
                        DcTech = 'LaSDI-LS Latent-Space Visualization'
                        DcTech = 'Latent-Space Dynamics by Linear Compression'
                    #ti = np.linspace(0, self.dt*(len(local[-1])-1), len(local[-1]))
                    fig = plt.figure()
                    fig.set_size_inches(9,6)
                    ax = plt.axes()
                    ax.set_title(DcTech)
                    labels = {'orig': 'Latent-Space Trajectory', 'new': 'Approximated Dynamics'}
                    for dim in range(local[-1].shape[1]):
                        plt.plot(t[:-1], local[-1][:-1,dim], alpha = .5, label = labels['orig'])
                        labels['orig'] = '_nolegend_'
                    plt.gca().set_prop_cycle(None)

                    new = model.simulate(x0 = local[-1][0], t_span = np.array([t[0], t[-1]]), t_eval = t)
                    for dim in range(local[-1].shape[1]):
                        plt.plot(t, new[:,dim], '--', label = labels['new'])
                        labels['new'] = '_nolegend_'
                    ax.legend()
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Magnitude')
                
                if self.NN == False:
                    return FOM_recon.T
                else:
                    return FOM_recon
            else: # WITH SINDy coefficient interpolation
                self.coeff_interp_model = np.empty(self.model_list[0].shape)
                self.training_time = 0
                # Compute SINDy coefficients of the testing parameter by interpolation
                for ls_dim in range(self.model_list[0].shape[0]):
                    for func_index in range(self.model_list[0].shape[1]):
                        f = self.Coef_interp_method(self.training_values[dist_index,0], 
                                                    self.training_values[dist_index,1], 
                                                    np.array(self.model_list)[dist_index,ls_dim,func_index], function=function, epsilon = epsilon)

                        self.coeff_interp_model[ls_dim, func_index] = f(pred_value[0], pred_value[1])
        
                def simulate(x0, t_span, t_eval, coef):
                    rows, cols = self.tags.shape
                    tol_ode = 10**(-13)
                    def rhs(t, x):
                        term = np.ones(rows)
                        for row in range(rows):
                            for col in range(cols): 
                                term[row] = term[row]*x[col]**self.tags[row, col]
                        return term.dot(coef)
                    sol = solve_ivp(fun = rhs, t_eval=t_eval, t_span=t_span, y0=x0, rtol=tol_ode)
                    return sol.y.T
                self.latent_space_recon = self.normal*simulate(IC/self.normal, t_eval=t, t_span = np.array([t[0], t[-1]]), coef = self.coeff_interp_model)
                FOM_recon = self.decoder(self.latent_space_recon)
                if self.NN == False:
                    return FOM_recon.T
                else:
                    return FOM_recon
                return
            
        
        

            
