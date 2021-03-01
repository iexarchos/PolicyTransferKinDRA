import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from S2RPolicyObjFun import S2RPolicyObjectiveFunction
from pdb import set_trace as bp

class S2RBayesOpt():
    def __init__(self, seed, gap, env_name,true_scales, policies,trials,n_iters,processes,n_pre_samples,adjust_reward_by_std=False,**kwargs):
        self.trials = trials
        self.n_iters = n_iters
        self.seed = seed
        self.env_name = env_name
        self.processes = processes
        self.true_scales = true_scales
        self.kwargs = kwargs
        self.gap = gap
        self.policies = policies
        self.adjust_reward_by_std = adjust_reward_by_std
        self.n_pre_samples = n_pre_samples
        self.result_list = []
        
    def Obfun(self,x,ID):
        print('Sampling policy ', ID)
        ObjFun = S2RPolicyObjectiveFunction(env_name = env_name, policy=self.policies[ID], true_scales=self.true_scales[ID],processes=self.processes, renders=False,seeds=False,**self.kwargs)
        reward, std = ObjFun.evaluate(scale=x,trials =self.trials)
        self.result_list.append([ID, x, reward, std, reward-std])
        if self.adjust_reward_by_std:
            reward = reward - std
            print('Policy ', ID,' std-Adjusted reward: ', reward, flush=True)
        ObjFun.env.close()
        return reward

    def expected_improvement(self,x, gaussian_process, evaluated_loss, greater_is_better=True, n_params=1):
        """ expected_improvement
        Expected improvement acquisition function.
        Arguments:
        ----------
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the expected improvement needs to be computed.
            gaussian_process: GaussianProcessRegressor object.
                Gaussian process trained on previously evaluated hyperparameters.
            evaluated_loss: Numpy array.
                Numpy array that contains the values off the loss function for the previously
                evaluated hyperparameters.
            greater_is_better: Boolean.
                Boolean flag that indicates whether the loss function is to be maximised or minimised.
            n_params: int.
                Dimension of the hyperparameter space.
        """
        x_to_predict = x.reshape(-1, n_params)
        mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)
        if greater_is_better:
            loss_optimum = np.max(evaluated_loss)
        else:
            loss_optimum = np.min(evaluated_loss)
        scaling_factor = (-1) ** (not greater_is_better)
        # In case sigma equals zero
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (mu - loss_optimum) / sigma
            expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0
        return -1 * expected_improvement




    def bayesian_optimisation(self,n_iters, ObjFun, bounds, x0=None, n_pre_samples=5,
                              gp_params=None, random_search=2000, alpha=1e-4, epsilon=1e-7, callback=None, termination_callback=None):
        """ bayesian_optimisation
        Uses Gaussian Processes to optimise the loss function `sample_loss`.
        Arguments:
        ----------
            n_iters: integer.
                Number of iterations to run the search algorithm.
            sample_loss: function.
                Function to be optimised.
            bounds: array-like, shape = [n_params, 2].
                Lower and upper bounds on the parameters of the function `sample_loss`.
            x0: array-like, shape = [n_pre_samples, n_params].
                Array of initial points to sample the loss function for. If None, randomly
                samples from the loss function.
            n_pre_samples: integer.
                If x0 is None, samples `n_pre_samples` initial points from the loss function.
            gp_params: dictionary.
                Dictionary of parameters to pass on to the underlying Gaussian Process.
            random_search: integer.
                Flag that indicates whether to perform random search or L-BFGS-B optimisation
                over the acquisition function.
            alpha: double.
                Variance of the error term of the GP.
            epsilon: double.
                Precision tolerance for floats.
            callback: function handler.
                Callback function at each iteration.
            termination_callback: function handler.
                Callback function that determines if the optimization should be terminated.
        """
        n_pol = len(self.true_scales)
        np.random.seed(self.seed)
        x_samples = []
        y = []
        n_params = bounds.shape[0]
        models = []
        for i in range(n_pol):
            Params = []
            rew = []
            Params.append(np.array(self.true_scales[i]))
            rew.append(ObjFun(np.array(self.true_scales[i]),i))
            for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples-1, bounds.shape[0])):
                Params.append(params)
                rew.append(ObjFun(params,i))
            #bp()
            x_samples.append(Params)
            y.append(rew)

        #xp = np.array(x_list)
        #yp = np.array(y_list)
        # Create the GP
        #if gp_params is not None:
        #    model = gp.GaussianProcessRegressor(**gp_params)
        #else:
            #kernel = gp.kernels.Matern(length_scale_bounds=(1e-2, 1e5)) + gp.kernels.WhiteKernel()
            models.append(gp.GaussianProcessRegressor(#kernel=kernel,
                                                alpha=alpha,
                                                n_restarts_optimizer=10,
                                                normalize_y=True))
            xp = np.array(x_samples[i])
            yp = np.array(y[i])
            models[i].fit(xp, yp)

        for n in range(n_iters):
            #ypmean = np.mean(yp)
            #ypstd = np.std(yp)
            #yp = (yp-ypmean) / (ypstd + 0.0001)
           
            #train_xp = np.copy(xp)
            #train_yp = np.copy(yp)
            
            #model.fit(train_xp, train_yp)
            # Sample next hyperparameter

            Proposed_sample_and_imp = np.zeros((n_pol,1+n_params))
            for i in range(n_pol):

                x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
                
                ei = -1 * self.expected_improvement(x_random, models[i], np.array(y[i]), greater_is_better=True, n_params=n_params)
                
                next_sample = x_random[np.argmax(ei), :]

            # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
                if np.any(np.abs(next_sample - np.array(x_samples[i])) <= epsilon):
                    next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

                Proposed_sample_and_imp[i,0] = np.max(ei)
                Proposed_sample_and_imp[i,1:] = next_sample
            print(Proposed_sample_and_imp[:,0])
            UCBAS = np.array([np.mean(np.array(y[p]))+100.0*np.sqrt(np.log(i)/len(y[p])) for p in range(n_pol)])
            print('UCBAS: ',UCBAS)
            next_sampled_policy = np.argmax(UCBAS*Proposed_sample_and_imp[:,0]) 
            #adjustor = 10*np.array([max(y[p])/len(y[p]) for p in range(n_pol)]) # multiply ei by max_policy_reward/ n_samples_in_policy    
            #next_sampled_policy = np.argmax(Proposed_sample_and_imp[:,0]*adjustor)
            next_sample = Proposed_sample_and_imp[next_sampled_policy,1:]
            # Sample loss for new set of parameters
            cv_score = ObjFun(next_sample,next_sampled_policy)
            # Update lists
            x_samples[next_sampled_policy].append(next_sample)
            y[next_sampled_policy].append(cv_score)
            # Update xp and yp
            xp = np.array(x_samples[next_sampled_policy])
            yp = np.array(y[next_sampled_policy])
            models[next_sampled_policy].fit(xp, yp)
            print('-----------------------Sample ',n+1,' of ', n_iters,' completed.------------------------',flush=True)
            if callback is not None:
                callback(None)
            if termination_callback is not None:
                terminate = termination_callback(None)
                if terminate:
                    break
        return x_samples, y, models

    def Optimize(self):
        xlist, ylist, models = self.bayesian_optimisation(n_iters=self.n_iters, ObjFun=self.Obfun, bounds= np.array(4*[[0.5, 1.5]]), x0=None, n_pre_samples=self.n_pre_samples,
                          gp_params=None, random_search=2000, alpha=1e-2, epsilon=1e-7, callback=None)
        return xlist, ylist, models, self.result_list





if __name__ == '__main__':
    seed = 1
    env_name = 'QuadrupedBulletScaledEnv-v4'


    #POSITION CONTROL
    control_mode='position'


    #kin-UP
    policies = ['/home/Research_Projects/co-design-control/quadruped/Policies/POS_0515_s1/ppo',
    '/home/Research_Projects/co-design-control/quadruped/Policies/POS_0515_s2/ppo',
    '/home/Research_Projects/co-design-control/quadruped/Policies/OS_0515_s3/ppo']

    true_scales = np.ones((3,4))



    gap = 'deform_floor_env'
    #gap = 'soft_floor_env'
    #gap = 'low_power_env'
    #gap = 'emf_power_env'
    #gap = 'joint_gap_env'



    n_pol = len(policies)
    n_random_start_per_policy = 3 
    iterations = 15-n_random_start_per_policy*n_pol
    trials = 2 
    
    random_IC = True
    init_noise = True
    shrink_IC_dist = True
    seeded_IC = False
    adjust_reward_by_std = True



    obs_noise = False
    act_noise = False
    processes = trials #do not change this! 
    deform_floor_env = True if gap == 'deform_floor_env' else False
    soft_floor_env = True if gap == 'soft_floor_env' else False
    low_power_env = True if gap == 'low_power_env' else False
    emf_power_env = True if gap == 'emf_power_env' else False
    joint_gap_env = True if gap == 'joint_gap_env'else False
    max_tar_vel = 20

    sess = S2RBayesOpt(seed=seed, gap=gap, env_name=env_name,true_scales=true_scales, policies=policies,trials=trials,n_iters=iterations,processes=processes, adjust_reward_by_std=adjust_reward_by_std, control_mode=control_mode,max_tar_vel=20.0,
        soft_floor_env=soft_floor_env,deform_floor_env=deform_floor_env,low_power_env=low_power_env,emf_power_env=emf_power_env,
        random_IC =random_IC, init_noise = init_noise, obs_noise=obs_noise, act_noise=act_noise, shrink_IC_dist=shrink_IC_dist, seeded_IC=seeded_IC,n_pre_samples=n_random_start_per_policy, joint_gap_env=joint_gap_env)
    xlist, ylist, models, res = sess.Optimize()
    print('---------------------------------------------------------',flush=True)
    print('-----------------------Completed.------------------------',flush=True)
    print('---------------------------------------------------------',flush=True)

    
    ymax = np.array([np.max(np.array(ylist[p])) for p in range(n_pol)])
    n_samples = np.array([len(ylist[p]) for p in range(n_pol)])
    sample_id = np.array([np.argmax(np.array(ylist[p])) for p in range(n_pol)])
    best_policy = np.argmax(ymax)
    best_sample_id = sample_id[best_policy]
    print("Optimal adjusted rewards: ", ymax)
    print('Sampling distribution: ', n_samples, ', total: ', n_samples.sum())
    R2 = np.array([models[p].score(np.array(xlist[p]),np.array(ylist[p])) for p in range(n_pol)])
    print('R^2 of models: ', R2)
    res = np.array(res)
    ind = np.argmax(res[:,-1])
    print('Optimal reward: ', res[ind, -3], ', std: ', res[ind,-2], ', policy ID: ', res[ind,0]+1)
    print('Optimal parameters: ', res[ind,1])
    bp()
    
