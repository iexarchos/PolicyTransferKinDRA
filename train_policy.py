#  MIT License
#
#  Copyright (c) 2017 Ilya Kostrikov
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

import my_pybullet_envs
import logging
import csv
from pathlib import Path
import sys
from pdb import set_trace as bp
def main():
    args,extra_dict = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if 'scales' in extra_dict.keys():
        scales = list(map(float,extra_dict['scales'].strip('[]').split(',')))
        extra_dict['scales'] = scales

    if 'scale_range' in extra_dict.keys():
        scale_range = list(map(float,extra_dict['scale_range'].strip('[]').split(',')))
        extra_dict['scale_range'] = scale_range

    if 'sigma' in extra_dict.keys():
        mu = np.array(scales)
        sigma = extra_dict['sigma']
        sigma_increase_factor = extra_dict['sigma_increase_factor']
        sigma_clip = extra_dict['sigma_clip']
        extra_dict.pop('sigma_clip')
        extra_dict.pop('sigma_increase_factor')
        mu_sigma_R_hist = []
        mu_sigma_R_hist.append([mu,sigma])  

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False,render=False,**extra_dict)

    if args.warm_start == '':
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy, 'hidden_size': args.hidden_size})
        actor_critic.to(device)
    else:
        # args.warm_start = "./trained_models_0219_cyl_2_place_0226_5/ppo/InmoovHandPlaceBulletEnv-v6.pt"
        # TODO: assume no state normalize ob_rms
        if args.cuda:
            actor_critic, _ = torch.load(args.warm_start)
        else:
            actor_critic, _ = torch.load(args.warm_start, map_location='cpu')
        if args.warm_start_logstd is not None:
            actor_critic.reset_variance(envs.action_space, args.warm_start_logstd)
        actor_critic.to(device)


    #dummy = gym.make(args.env_name, renders=False, **extra_dict)
    save_path = os.path.join(args.save_dir, args.algo)
    print("SAVE PATH:")
    print(save_path)
    try:
        os.makedirs(save_path)
    except FileExistsError:
        print("warning: path existed")
        # input("warning: path existed")
    except OSError:
        exit()
    pathname = os.path.join(save_path, "source_test.py")
    #text_file = open(pathname, "w+")
    #text_file.write(dummy.getSourceCode())
    #text_file.close()
    #print("source file stored")
    ##input("source file stored press enter")
    #dummy.close()
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler("{0}/{1}.log".format(save_path, "console_output"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    # if args.gail:
    #     assert len(envs.observation_space.shape) == 1
    #     discr = gail.Discriminator(
    #         envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
    #         device)
    #     file_name = os.path.join(
    #         args.gail_experts_dir, "trajs_{}.pt".format(
    #             args.env_name.split('-')[0].lower()))
        
    #     expert_dataset = gail.ExpertDataset(
    #         file_name, num_trajectories=4, subsample_frequency=20)
    #     drop_last = len(expert_dataset) > args.gail_batch_size
    #     gail_train_loader = torch.utils.data.DataLoader(
    #         dataset=expert_dataset,
    #         batch_size=args.gail_batch_size,
    #         shuffle=True,
    #         drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10000)
    episode_scale = deque(maxlen=10000)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    best_mean = -np.inf

    total_rollouts = 0
    #p = Path('data.npy')
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            #if j%4==0:
            #    with p.open('ab') as f:
            #        np.save(f,obs.cpu().numpy())
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_scale.append(info['scales'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        # if args.gail:
        #     if j >= 10:
        #         envs.venv.eval()

        #     gail_epoch = args.gail_epoch
        #     if j < 10:
        #         gail_epoch = 100  # Warm up
        #     for _ in range(gail_epoch):
        #         discr.update(gail_train_loader, rollouts,
        #                      utils.get_vec_normalize(envs)._obfilt)

        #     for step in range(args.num_steps):
        #         rollouts.rewards[step] = discr.predict_reward(
        #             rollouts.obs[step], rollouts.actions[step], args.gamma,
        #             rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()


        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + "_" + str(j) + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            total_rollouts+= len(episode_rewards)
            rootLogger.info(
                ("Updates {}, num timesteps {}, FPS {}, Total number of rollouts {} \n Last {} training episodes:" +
                 " mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, dist en {}, l_pi {}, l_vf {}\n")
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),total_rollouts,
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            
            #rootLogger.info(("Total number of rollouts: {}").format(total_rollouts))
            if 'sigma' in extra_dict.keys():
                
                ep_rew = np.array(list(episode_rewards))
                ep_sc = np.array(list(episode_scale))
                perc = 0.10 #top 10%
                indeces = np.flip(np.argsort(ep_rew))
                #print(ep_rew[indeces])
                #bp()
                elite = indeces[0:np.int(perc*indeces.shape[0])]
                mu = 0.5*mu+0.5*np.mean(ep_sc[elite,:],axis=0)
                sigma *= sigma_increase_factor
                sigma = np.clip(sigma, 0, sigma_clip)
                envs.upd_mu_sigma(sigma,mu=mu)
            #bp()
                rootLogger.info(("Setting mu = {}, sigma = {}").format(mu,sigma))
                mu_sigma_R_hist.append([mu,sigma,np.mean(episode_rewards)])

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

        episode_rewards.clear()
        episode_scale.clear()
               
    if 'sigma' in extra_dict.keys():
        npy_data = np.zeros((len(mu_sigma_R_hist),6))
        for i in range(len(mu_sigma_R_hist)):
            if i==0:
                npy_data[i,0:4]=mu_sigma_R_hist[i][0]
                npy_data[i,4]=mu_sigma_R_hist[i][1]
                npy_data[i,5]=mu_sigma_R_hist[i+1][2]
            else:    
                npy_data[i,0:4]=mu_sigma_R_hist[i][0]
                npy_data[i,4]=mu_sigma_R_hist[i][1]
                npy_data[i,5]=mu_sigma_R_hist[i][2]
        filename = os.path.join(save_path, "mu_sigma_R_his.npy")
        np.save(filename, npy_data)

if __name__ == "__main__":
    main()
