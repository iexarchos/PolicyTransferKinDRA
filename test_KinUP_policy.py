#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:09:05 2020

@author: yannis
"""

import torch
import random
from pdb import set_trace as bp
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import  get_vec_normalize
import my_pybullet_envs
import time
import numpy as np



def testPolicy(path,gap,scales,pol_scales):

    max_tar_vel = 20
    processes = 1

    control_mode = 'position' #torque or position
    deform_floor_env = True if gap == 'deform_floor_env' else False
    soft_floor_env = True if gap == 'soft_floor_env' else False
    low_power_env = True if gap == 'low_power_env' else False
    emf_power_env = True if gap == 'emf_power_env' else False
    joint_gap_env = True if gap == 'joint_gap_env'else False

    render = True
    random_IC = True
    init_noise = True
    shrink_IC_dist = True
    seeded_IC = False



    seed = 1


    obs_noise = False
    act_noise = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = make_vec_envs(
       'QuadrupedBulletScaledEnv-v4',
        seed,
        processes,
        None,
        None,
        device='cpu',
        allow_early_resets=True, render=render, scales=scales,control_mode=control_mode,max_tar_vel=max_tar_vel, 
        soft_floor_env=soft_floor_env,deform_floor_env=deform_floor_env,low_power_env=low_power_env,emf_power_env=emf_power_env, 
        random_IC =random_IC, init_noise = init_noise, obs_noise=obs_noise, act_noise=act_noise,shrink_IC_dist=shrink_IC_dist, seeded_IC=seeded_IC, joint_gap_env=joint_gap_env)

    env_core = env.venv.venv.envs[0].env.env
    actor_critic,  ob_rms = torch.load(path,map_location=torch.device('cpu'))
    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms
    recurrent_hidden_states = torch.zeros(1,actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, processes)
    #env_core = env.venv.venv.envs[0]
    if processes==1:
        N_sim = 100
        Reward = np.zeros((N_sim,))
        input('press enter')
        n=0
        R=0
        obs=env.reset()
        while n<N_sim: 
            obs[:,-4:] = torch.FloatTensor(pol_scales)
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(obs,recurrent_hidden_states,masks, deterministic = True )   
            obs, reward, done, _ = env.step(action[0])
            obs[:,-4:] = torch.FloatTensor(pol_scales)
            env_core.cam_track_torso_link()
            R+=reward
            #control_steps +=1
            time.sleep(5*1.0/240.0)
            if done:
                n+=1
                Reward[n]=R
                print('Reward: ',R)
                R=0
                #obs=env.reset()
                #obs[:,-4:] = torch.FloatTensor(pol_scales)
                #input('press enter')
            
            masks.fill_(0.0 if done else 1.0)
        #print('Scale: ', Scale[j,:], ', total reward:' , Reward)
        input('press enter')
    else:
        N_sim = processes
        TotalReward = np.zeros((processes,))   
        obs=env.reset()
        #bp()
        #bp()
        n = 0
        while n<N_sim:
            obs[:,-4:] = torch.FloatTensor(pol_scales) # replace scale in the input of the policy
            with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=True)

            obs, reward, done, _ = env.step(action)
            obs[:,-4:] = torch.FloatTensor(pol_scales) # replace scale in the input of the policy

            TotalReward += reward.numpy().flatten()
            for D in done:
                if D:
                    #print(done)
                    n+=1
            masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
        print('TotalReward: ', TotalReward, flush=True)
        AverageTotalReward = np.mean(TotalReward)
        Std = np.std(TotalReward)
        #print(TotalReward)
        print('Av. Total reward: ',AverageTotalReward, ', std: ',Std,', virtual scale: ', obs[0,-4:], flush=True)

    #bp()

        N_sim = processes
        TotalReward = np.zeros((processes,))   
        obs=env.reset()
        #bp()
        #bp()
        n = 0
        while n<N_sim:
            obs[:,-4:] = torch.FloatTensor(pol_scales) # replace scale in the input of the policy
            with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=True)

            obs, reward, done, _ = env.step(action)
            obs[:,-4:] = torch.FloatTensor(pol_scales) # replace scale in the input of the policy

            TotalReward += reward.numpy().flatten()
            for D in done:
                if D:
                    #print(done)
                    n+=1
            masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
        print('TotalReward: ', TotalReward, flush=True)
        AverageTotalReward = np.mean(TotalReward)
        Std = np.std(TotalReward)
        #print(TotalReward)
        print('Av. Total reward: ',AverageTotalReward, ', std: ',Std,', virtual scale: ', obs[0,-4:], flush=True)
    env.close()
#bp()


if __name__ == '__main__':
    # kinUP
    scales = [1.0, 1.0, 1.0, 1.0]
    #path = '/home/Research_Projects/co-design-control/quadruped/Policies/POS_0515_s1/ppo/QuadrupedBulletScaledEnv-v4.pt'
    #path = '/home/Research_Projects/co-design-control/quadruped/Policies/POS_0515_s2/ppo/QuadrupedBulletScaledEnv-v4.pt'
    path = '/home/Research_Projects/co-design-control/quadruped/Policies/POS_0515_s3/ppo/QuadrupedBulletScaledEnv-v4.pt'



    #gap = 'None'
    #gap = 'low_power_env'
    #gap = 'emf_power_env'
    gap = 'joint_gap_env'
    #gap = 'deform_floor_env'
    #gap = 'soft_floor_env'
    #pol_scales = scales

    #pol_scales = [1.00472172, 1.29299243, 1.43391748, 1.44283027] #deform pol 1
    #pol_scales = [1.45281672, 0.50917272, 1.48601903, 0.764402  ] #soft pol 3
    #pol_scales =[0.52509703, 0.60478981, 1.05566793, 0.72000786] #low pol 2
    #pol_scales =  [0.70445225, 1.37811744, 0.52738759, 1.17046751] #emf pol 2
    pol_scales =  [0.6267561,  0.51621219, 1.45087057, 1.44768888] #joint pol 2


    testPolicy(path,gap,scales,pol_scales)