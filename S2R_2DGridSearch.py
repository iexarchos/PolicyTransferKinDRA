import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from S2RPolicyObjFun import S2RPolicyObjectiveFunction
from pdb import set_trace as bp

class Sim2Real2DGridSearch():
    def __init__(self,gap, env_name,true_scales, policy,trials,processes,N,**kwargs):
        self.env_name = env_name
        self.gap = gap
        self.true_scales = true_scales
        self.N = N
        self.trials=trials
        self.policy = policy
        self.kwargs = kwargs
        self.processes = processes
        self.ObjFun = S2RPolicyObjectiveFunction(env_name = env_name, policy=self.policy, true_scales=self.true_scales,processes=self.processes, renders=False,seeds=False,**self.kwargs)

    def Obfun(self,x):
            
        reward, std = self.ObjFun.evaluate(scale=x,trials =self.trials) #PPO

        return reward






    def Search(self, bounds, x_samples=None):
        self.N = N
        x_list = []
        y_list = []

        x_s = np.linspace(bounds[0],bounds[1],self.N)
        x1,x2 = np.meshgrid(x_s,x_s)
        x1 = np.reshape(x1,(self.N*self.N,1))
        x2 = np.reshape(x2,(self.N*self.N,1))
        x_samples = np.concatenate((x1,x1,x2,x2),axis=1)


        for i in range(0,x_samples.shape[0]):
            Rew = self.Obfun(x_samples[i,:])
            y_list.append(Rew)
            print('-----------------------Sample ',i+1,' of ', x_samples.shape[0],' completed.------------------------')


        
        
        filename = 'GridSearch2D_'+self.policy[-11:-4]+'_'+self.gap+'.npz'
        np.savez(filename,x_samples,y_list)
        
        return x_samples, np.array(y_list)

if __name__ == '__main__':
    import sys
    
    env_name = 'QuadrupedBulletScaledEnv-v4'
    #POSITION CONTROL
    control_mode='position'


    policy = '/home/Research_Projects/co-design-control/quadruped/Policies/POS_0515_s1/ppo'



    trials = 15
    N = 30
    gap = 'joint_gap_env'
    true_scales = [1.0, 1.0, 1.0, 1.0]
    random_IC = True
    init_noise = True
    shrink_IC_dist = True
    seeded_IC = False
    adjust_reward_by_std = True



    obs_noise = False
    act_noise = False
    processes = trials #do not change 
    deform_floor_env = True if gap == 'deform_floor_env' else False
    soft_floor_env = True if gap == 'soft_floor_env' else False
    low_power_env = True if gap == 'low_power_env' else False
    emf_power_env = True if gap == 'emf_power_env' else False
    joint_gap_env = True if gap == 'joint_gap_env'else False
    max_tar_vel = 20

    sess = Sim2Real2DGridSearch(gap=gap, env_name=env_name,true_scales=true_scales, policy=policy,trials=trials,N=N,processes=processes,  control_mode=control_mode,max_tar_vel=20.0,
        soft_floor_env=soft_floor_env,deform_floor_env=deform_floor_env,low_power_env=low_power_env,emf_power_env=emf_power_env,
        random_IC =random_IC, init_noise = init_noise, obs_noise=obs_noise, act_noise=act_noise, shrink_IC_dist=shrink_IC_dist, seeded_IC=seeded_IC, joint_gap_env=joint_gap_env)
    xlist, ylist = sess.Search(bounds=np.array([0.5, 1.5]),x_samples=None)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(xlist[:, 0], xlist[:, 2], c=ylist)#, vmin=0.0, vmax=10000.0)
    plt.xlim([0.5, 1.5])
    plt.ylim([0.5, 1.5])
    plt.colorbar()
    plt.xlabel('Front leg scale')
    plt.ylabel('Back leg scale')
    plt.scatter(true_scales[0],true_scales[1],c='r')
    plt.show()
    #import matplotlib
    #matplotlib.use('TkAgg')
    #import matplotlib.pyplot as plt
    bp()
    #ind = np.argmax(ylist)
    #plt.figure()
    #plt.plot(xlist,ylist)
    #plt.plot(ind,ylist[ind],'ro')
    #plt.show()