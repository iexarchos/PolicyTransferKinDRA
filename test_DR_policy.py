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
from pathlib import Path

def testPolicy(path,gap,scales ):
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
       'QuadrupedBulletDesRandEnv-v4',
        seed,
        processes,
        None,
        None,
        device='cpu',
        allow_early_resets=True, render=render, scales=scales,control_mode=control_mode,max_tar_vel=max_tar_vel, 
        soft_floor_env=soft_floor_env,deform_floor_env=deform_floor_env,low_power_env=low_power_env,emf_power_env=emf_power_env, 
        random_IC =random_IC, init_noise = init_noise, obs_noise=obs_noise, act_noise=act_noise,shrink_IC_dist=shrink_IC_dist, seeded_IC=seeded_IC, joint_gap_env=joint_gap_env)
        #randomization_train=True)
        #train_universal_policy=True)

    env_core = env.venv.venv.envs[0].env.env
    actor_critic,  ob_rms = torch.load(path,map_location=torch.device('cpu'))
    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms
    recurrent_hidden_states = torch.zeros(1,actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, processes)
    #env_core = env.venv.venv.envs[0]
    #p = Path('data.npy')
    if processes==1:
        N_sim = 100
        Reward = np.zeros((N_sim,))
        input('press enter')
        n=0
        R=0
        obs=env.reset()
        while n<N_sim: 
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(obs,recurrent_hidden_states,masks, deterministic = True )   
            obs, reward, done, _ = env.step(action[0])
            env_core.cam_track_torso_link()
            R+=reward
            #control_steps +=1
            time.sleep(5*1.0/240.0)

            #with p.open('ab') as f:
            #    np.save(f,obs.cpu().numpy())
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
            with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=True)

            obs, reward, done, _ = env.step(action)

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
        print('Av. Total reward: ',AverageTotalReward, ', std: ',Std, flush=True)

    #bp()

        N_sim = processes
        TotalReward = np.zeros((processes,))   
        obs=env.reset()
        #bp()
        #bp()
        n = 0
        while n<N_sim:
            with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=True)

            obs, reward, done, _ = env.step(action)

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
        print('Av. Total reward: ',AverageTotalReward, ', std: ',Std, flush=True)
    env.close()

#bp()

if __name__ == '__main__':
    #enter path to a kin-DR or dyn-DR policy:
    path = '/home/yannis/Research_Projects/co-design-control/quadruped/Policies/POS_Fixed_Nominal_s2/ppo/QuadrupedBulletDesRandEnv-v4.pt'


    scales = [1.0, 1.0, 1.0 , 1.0]
    #gap = 'None'
    #gap = 'low_power_env'
    #gap = 'emf_power_env'
    #gap = 'joint_gap_env'
    #gap = 'deform_floor_env'
    gap = 'soft_floor_env'
    testPolicy(path,gap,scales)
