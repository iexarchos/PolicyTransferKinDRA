import os
import sys
import time
import numpy as np
import torch
import gym
import my_pybullet_envs
import pickle
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from pdb import set_trace as bp
import random


class S2RPolicyObjectiveFunction():
	def __init__(self,env_name,policy,true_scales,renders,processes=1,seeds=True,**kwargs):
		self.env_name = env_name
		self.policy = policy
		self.true_scales = true_scales 
		extra_dict = {}
		self.processes=processes
		extra_dict['render'] = renders
		#save_final_states = 0
		self.extra_dict ={**extra_dict,**kwargs}

		self.seed = 1
		torch.manual_seed(self.seed)
		if seeds:
			np.random.seed(self.seed)
			random.seed(self.seed)
		is_cuda = False
		self.device = "cuda" if is_cuda else "cpu"



		path = os.path.join(self.policy, self.env_name + ".pt")
		if is_cuda:
			self.actor_critic, self.ob_rms = torch.load(path)
		else:
			self.actor_critic, self.ob_rms = torch.load(path, map_location="cpu")
		

		self.recurrent_hidden_states = torch.zeros(
    		1, self.actor_critic.recurrent_hidden_state_size)
		self.masks = torch.zeros(1,self.processes)

		self.extra_dict['scales'] = self.true_scales
		self.env = make_vec_envs(
    		self.env_name,
    		self.seed,
    		self.processes,
    		None,
    		None,
    		device=self.device,
    		allow_early_resets=False,
    		**self.extra_dict)
		vec_norm = get_vec_normalize(self.env)
		if vec_norm is not None:
			vec_norm.eval()
			vec_norm.ob_rms = self.ob_rms



	def evaluate(self,scale,trials):
		assert trials == self.processes
		TotalReward = np.zeros((self.processes,))	
		obs=self.env.reset()
		#bp()
		obs[:,-4:] = torch.FloatTensor(scale) # replace scale in the input of the policy
		#bp()
		n = 0
		while n<trials:
			obs[:,-4:] = torch.FloatTensor(scale)
			with torch.no_grad():
					value, action, _, recurrent_hidden_states = self.actor_critic.act(
            		obs, self.recurrent_hidden_states, self.masks, deterministic=True)
			obs, reward, done, _ = self.env.step(action)
			obs[:,-4:] = torch.FloatTensor(scale) # replace scale in the input of the policy

			TotalReward += reward.numpy().flatten()
			for D in done:
				if D:
					#print(done)
					n+=1
		#print('TotalReward: ', TotalReward, flush=True)
		masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
		AverageTotalReward = np.mean(TotalReward)
		Std = np.std(TotalReward)
    	#print(TotalReward)
		print('Av. Total reward: ',AverageTotalReward, ', std: ',Std,', virtual scale: ', obs[0,-4:], flush=True)

		#self.env.close()
		#del self.env
		return AverageTotalReward, Std
		