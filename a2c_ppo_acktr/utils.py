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

import glob
import os

import torch
import torch.nn as nn

import numpy as np
import pickle
import pybullet
from collections import deque

from a2c_ppo_acktr.envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def load(policy_dir: str, env_name: str, is_cuda: bool, iter_num=None):
    """Loads parameters for a specified policy.

    Args:
        policy_dir: The directory to load the policy from.
        env_name: The environment name of the policy.
        is_cuda: Whether to use gpu.
        iter_num: The iteration of the policy model to load.

    Returns:
        actor_critic: The actor critic model.
        ob_rms: ?
        recurrent_hidden_states: The recurrent hidden states of the model.
        masks: ?
    """
    if iter_num is not None and iter_num >= 0:
        path = os.path.join(policy_dir, env_name + "_" + str(int(iter_num)) + ".pt")
    else:
        path = os.path.join(policy_dir, env_name + ".pt")
    print(f"| loading policy from {path}")
    if is_cuda:
        actor_critic, ob_rms = torch.load(path)
    else:
        actor_critic, ob_rms = torch.load(path, map_location="cpu")
    d = "cuda" if is_cuda else "cpu"
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size, device=torch.device(d))
    masks = torch.zeros(1, 1, device=torch.device(d))
    return (
        actor_critic,
        ob_rms,
        recurrent_hidden_states,
        masks,
    )


def wrap(obs, is_cuda: bool) -> torch.Tensor:
    obs = torch.Tensor([obs])
    if is_cuda:
        obs = obs.cuda()
    return obs


def unwrap(action: torch.Tensor, is_cuda: bool, clip=False) -> np.ndarray:
    action = action.squeeze()
    action = action.cpu() if is_cuda else action
    if clip:
        action = np.clip(action.numpy(), -1.0, 1.0)
    else:
        action = action.detach().numpy()
    return action


def perturb(arr, r=0.02, np_rand_gen=np.random):
    r = np.abs(r)
    return np.copy(
        np.array(arr) + np_rand_gen.uniform(low=-r, high=r, size=len(arr))
    )


def perturb_scalar(num, r=0.02, np_rand_gen=np.random):
    r = np.abs(r)
    return num + np_rand_gen.uniform(low=-r, high=r)


def push_recent_value(deque_d, value):
    max_len = deque_d.maxlen
    if len(deque_d) == 0:
        # pad init value max_len times
        for i in range(max_len):
            deque_d.appendleft(list(value))
    else:
        # update and pop right the oldest one
        deque_d.appendleft(list(value))


def select_and_merge_from_s_a(s_mt, a_mt, s_idx=np.array([0, ]), a_idx=np.array([])):
    # s_mt and a_mt are two lists of lists
    # each element being either a single length-l s/a list
    # return a combined_length, vector

    merged_sa = np.array([])
    for i in s_idx:
        merged_sa = np.concatenate((merged_sa, s_mt[i]))
    for j in a_idx:
        merged_sa = np.concatenate((merged_sa, a_mt[j]))

    return merged_sa


def get_link_com_xyz_orn(body_id, link_id, bullet_session):
    # get the world transform (xyz and quaternion) of the Center of Mass of the link
    # We *assume* link CoM transform == link shape transform (the one you use to calculate fluid force on each shape)
    assert link_id >= -1
    if link_id == -1:
        link_com, link_quat = bullet_session.getBasePositionAndOrientation(body_id)
    else:
        link_com, link_quat, *_ = bullet_session.getLinkState(body_id, link_id, computeForwardKinematics=1)
    return list(link_com), list(link_quat)


def apply_external_world_force_on_local_point(body_id, link_id, world_force, local_com_offset, bullet_session):
    link_com, link_quat = get_link_com_xyz_orn(body_id, link_id, bullet_session)
    _, inv_link_quat = bullet_session.invertTransform([0., 0, 0], link_quat)  # obj->world
    local_force, _ = bullet_session.multiplyTransforms([0., 0, 0], inv_link_quat, world_force, [0, 0, 0, 1])
    bullet_session.applyExternalForce(body_id, link_id, local_force,
                                      local_com_offset, flags=bullet_session.LINK_FRAME)

