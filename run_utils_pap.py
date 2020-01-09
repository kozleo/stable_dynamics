import numpy as np
import sdeint
from scipy import integrate
from scipy import linalg


class RunParams:
    def __init__(self, n, W_nn, tspan, sigma,uts,dt):
        self.n = n
        self.s = n**2
        self.u = n + n**2
        self.W_nn = W_nn
        self.tspan = tspan
        self.sigma = sigma
        self.uts = uts
        self.dt = dt
        
        
def create_input(n,tspan):
    freqs = np.random.uniform(1,100,n)
    phases = np.random.uniform(0,2*np.pi,n)
    amps = np.random.uniform(0,20,n)
    counts = tspan.size
    u_ts = (np.sin(np.outer(freqs,tspan) + np.outer(phases,np.ones(counts))))*np.outer(amps,np.ones(counts))

    return u_ts

def disturb_input(n,tspan,u_ts):
    counts = tspan.size
    disturbance_dur = 100
    zero_tmp = np.zeros(u_ts.shape)
    zero_tmp[:,int(counts/2):int(counts/2)+disturbance_dur] += 10*np.ones((n,disturbance_dur))
    
    
    u_ts_new = u_ts + zero_tmp
    
    return u_ts_new
        
             
def create_run_funcs(RunParams):
    
    def f(x, t):
        #first n are neurons 
        #remaining u-n are synapses
        
        x_vec_old = x[0:RunParams.n]
        w_vec_old = x[RunParams.n::]
        w_square_old = np.reshape(w_vec_old,(RunParams.n,RunParams.n))
        
        w_update = - w_square_old - np.outer(x_vec_old,x_vec_old)
        w_update = np.reshape(w_update,(RunParams.s))
        
        
        x_update = -np.eye(RunParams.n).dot(x_vec_old) + w_square_old @ x_vec_old + RunParams.uts[:,int(t/RunParams.dt)]

        
        whole_update = np.concatenate((x_update,w_update))
        
       
        return whole_update

    def G(x, t):
        return RunParams.sigma*np.eye(RunParams.u)
    
    return f,G


def do_single_run(f,G,RunParams,x0, dense_output = False):
    print('Doing one run.')
    tcount = RunParams.tspan/RunParams.dt
    result = sdeint.itoEuler(f, G, x0, RunParams.tspan)       
    return result 
