
import numpy as np
import numpy.linalg as LA
import wsindy as ws
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, Rbf
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from itertools import combinations_with_replacement
import time
import WENDy as wd

class WLaSDI_wendy:
    """
    WLaSDI with WENDy class for data-driven ROM. Functions: train_dynamics approximates dynamical systems of the latent-space. 
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
    
    def train_dynamics(self, ls_trajs, training_values, t, features = None,  normal = 1, LS_vis = True, toggle_VVp_svd = np.nan, mt_params=[2**i for i in range(4)], subsample = 4, gamma = 0.1, ls_meth = "LS", mt_min = None, mt_max = None):
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
           opt_tfsupp: toggle the use of optimal test function support
        """

        self.training_values = training_values
        self.ls_trajs = ls_trajs
        self.t = t
        self.normal = normal
        self.LS_vis = LS_vis
       
        #WENDy params
        self.toggle_VVp_svd = toggle_VVp_svd
        self.mt_params = mt_params
        self.subsample = subsample
        self.features = features
        self.gamma = gamma
        self.ls_meth = ls_meth
        self.mt_min = mt_min
        self.mt_max = mt_max

        data_LS = []
        for traj in ls_trajs:
            data_LS.append(traj/normal)
        self.data_LS = data_LS


    
        if self.Local == False:
            print('mt_min', self.mt_min)
            
            model = wd.WENDy(toggle_VVp_svd = toggle_VVp_svd, mt_params= mt_params, subsample = subsample, ls_meth=ls_meth, gamma=gamma, mt_min = self.mt_min, mt_max = self.mt_max)
            #generating t
            t_val = [t]
            for i in range(1, len(data_LS)):
                t_val.append(t)

            coef = model.fit(x = data_LS, t = t_val, features=features)
            self.coef  = coef

            self.model = model
            if LS_vis == True:
                if self.NN == True:
                    DcTech = 'LaSDI-NM Latent-Space Visualization'
                    DcTech = 'Latent-Space Dynamics by Nonlinear Compression'
                else:
                    DcTech = 'LaSDI-LS Latent-Space Visualization'
                    DcTech = 'Latent-Space Dynamics by Linear Compression'
                #time = np.linspace(0, dt*(len(data_LS[-1])-1), len(data_LS[-1]))
                fig = plt.figure()
                fig.set_size_inches(9,6)
                ax = plt.axes()
                ax.set_title(DcTech)
                labels = {'orig': 'Latent-Space Trajectory', 'new': 'Approximated Dynamics'}
                for dim in range(data_LS[-1].shape[1]):
                    plt.plot(t[:-1], data_LS[-1][:-1,dim], alpha = .5, label = labels['orig'])
                    labels['orig'] = '_nolegend_'
                plt.gca().set_prop_cycle(None)
                new = model.simulate(x0=data_LS[-1][0], t = t_val[-1] )
                for dim in range(data_LS[-1].shape[1]):
                    plt.plot(t, new[:,dim], '--', label = labels['new'])
                    labels['new'] = '_nolegend_'
                ax.legend()
                ax.set_xlabel('Time')
                ax.set_ylabel('Magnitude')
                #plt.savefig(self.plot_fname)
            return model.w_hat 
        elif self.Coef_interp == True:
            print("Local approach WITH SINDy coefficient interpolation")
            if self.Coef_interp_method == None:
                print('WARNING: Please specify an interpolation method either interp2d or Rbf')
            self.model_list = []
            for i, _ in enumerate(training_values):
                model = wd.WENDy(toggle_VVp_svd = toggle_VVp_svd, mt_params= mt_params, subsample = subsample, ls_meth=ls_meth, gamma=gamma, mt_min = self.mt_min, mt_max = self.mt_max)
                coef = model.fit(x = [data_LS[i]], t = [t], features=features)
                self.model_list.append(coef)
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

                    new = model.simulate(x0 = data_LS[i][0], t = t)
                    for dim in range(data_LS[-1].shape[1]):
                        plt.plot(t, new[:,dim], '--', label = labels['new'])
                        labels['new'] = '_nolegend_'
                    ax.legend()
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Magnitude')
                    #plt.savefig(self.plot_fname)
            return self.model_list
        else:
            print("Local approach WITHOUT SINDy coefficient interpolation")
            return 
        
    
        
    def generate_ROM(self,pred_IC,pred_value,t):
        """
        Takes initial condition in full-space and associated parameter values and generates forward in time using the trained dynamics from above.
        Inputs:
            pred_IC: Initial condition of the desired simulation
            pred_value: Associated parameter values
            t: time stamps corresponding to training FOMs
        """
        IC = self.IC_gen(pred_IC)
        #self.time = t
        if self.Local == False: # Global approach
            latent_space_recon = self.normal*self.model.simulate(x0 = IC/self.normal,t = t)
            FOM_recon = self.decoder(latent_space_recon)
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
                    
                self.model = wd.WENDy(toggle_VVp_svd = self.toggle_VVp_svd, mt_params= self.mt_params, subsample = self.subsample, ls_meth=self.ls_meth, gamma=self.gamma, mt_min = self.mt_min, mt_max = self.mt_max)
                t_val = [self.t]
                for i in range(1, len(local)):
                    t_val.append(self.t)

                self.model.fit(x = local, t = t_val, features=self.features)

                self.training_time = time.time()-training_time_start
                latent_space_recon = self.normal*self.model.simulate(x0 = IC/self.normal,t = t)
                FOM_recon = self.decoder(latent_space_recon)
                
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

                    new = self.model.simulate(x0 = local[-1][0], t = t)
                    
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
                # Compute SINDy coefficients of the testing parameter by interpolation
                for ls_dim in range(self.model_list[0].shape[0]):
                    for func_index in range(self.model_list[0].shape[1]):
                        f = self.Coef_interp_method(self.training_values[dist_index,0], 
                                                   self.training_values[dist_index,1], 
                                                    np.array(self.model_list)[dist_index,ls_dim,func_index])

                        self.coeff_interp_model[ls_dim, func_index] = f(pred_value[0], pred_value[1])
                def simulate(w_hat, x0, t, tspan):
                    tol_ode = 1e-8
                    
                    w_hat_tolist = []
                    count = 0
                    for i in range(len(self.features)): 
                        a = self.features[i]
                        coef = []
                        for j in range(len(a)):
                            coef.append(w_hat[count+j][0])
                        count = count + len(a)
                        w_hat_tolist.append(coef)
                      
                    def rhs_fun(features, params, x):
                        nstates = len(x)
                        x = tuple(x)
                        dx = np.zeros(nstates)
                        for i in range(nstates):
                            dx[i] = np.sum([f(*x)*p for f, p in zip(features[i], params[i])])
                        return dx
                    rhs_p = lambda t, x: rhs_fun(self.features, w_hat_tolist, x)
                    sol = solve_ivp(rhs_p, t_span = tspan, y0=x0, t_eval=t, rtol=tol_ode, atol=tol_ode)
                    return sol.y.T
                self.latent_space_recon = self.normal*simulate(x0 = IC/self.normal, t=t, tspan = np.array([t[0], t[-1]]), w_hat = self.coeff_interp_model)
                # high-dimensional dynamics
                FOM_recon = self.decoder(self.latent_space_recon)
                if self.NN == False:
                    return FOM_recon.T
                else:
                    return FOM_recon
                return
            
        
        

            