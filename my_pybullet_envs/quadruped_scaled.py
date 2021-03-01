#  Copyright 2020 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# DoF index, DoF (joint) Name, joint type (0 means hinge joint), joint lower and upper limits, child link of this joint
# (0, b'FR_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'FR_hip_motor'
# (1, b'FR_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'FR_upper_leg'
# (2, b'FR_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'FR_lower_leg'
# (3, b'jtoeFR', 4) 0.0 -1.0 b'toeFR'
# (4, b'FL_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'FL_hip_motor'
# (5, b'FL_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'FL_upper_leg'
# (6, b'FL_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'FL_lower_leg'
# (7, b'jtoeFL', 4) 0.0 -1.0 b'toeFL'
# (8, b'RR_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'RR_hip_motor'
# (9, b'RR_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'RR_upper_leg'
# (10, b'RR_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'RR_lower_leg'
# (11, b'jtoeRR', 4) 0.0 -1.0 b'toeRR'
# (12, b'RL_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'RL_hip_motor'
# (13, b'RL_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'RL_upper_leg'
# (14, b'RL_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'RL_lower_leg'
# (15, b'jtoeRL', 4) 0.0 -1.0 b'toeRL'
# ctrl dofs: [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

import pybullet_utils.bullet_client as bc
import time
import gym, gym.utils.seeding
import numpy as np
import math
from gan import utils
import pybullet
import pybullet as p
import os
import inspect
from pdb import set_trace as bp
import contextlib

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class QuadrupedSCALE:
    def __init__(self,
                 scales = [1.0,1.0,1.0,1.0],
                 init_noise=True,
                 time_step=1. / 500,
                 np_random=None,
                 control_mode = 'torque',   #choose 'torque' or 'position'
                 random_IC = True, 
                 shrink_IC_dist = False,
                 seeded_IC = False,
                 design_joint_angle = False,
                 joint_gap_env = False
                 ):
        self.scales = scales
        self.init_noise = init_noise
        self.control_mode = control_mode
        self.random_IC = random_IC
        self.shrink_IC_dist = shrink_IC_dist
        self.IC_factor = 100.0 if shrink_IC_dist else 1.0
        self.seeded_IC = seeded_IC
        self.design_joint_angle = design_joint_angle
        self.joint_gap_env = joint_gap_env
        if self.joint_gap_env:
            assert self.design_joint_angle == False


        self._ts = time_step
        self.np_random = np_random

        self.base_init_pos = np.array([0, 0, .56])  # starting position
        self.base_init_euler = np.array([1.5708, 0, 1.5708])  # starting orientation

        self.feet = [3, 7, 11, 15]

        self.max_forces = [30.0] * 12  # joint torque limits
        self.nominal_max_forces = self.max_forces.copy()

        # ang vel scaled to 0.2, dq scaled to 0.04
        self.robo_obs_scale = np.array([1.0] * (1 + 9 + 3 + 12 + 12) + [0.2] * 3 + [0.04] * 12)

        self.init_q = [0.0, 0.0, -0.5] * 4
        #self.init_q = [1.5, -0.9, 0.5] * 4
        self.ctrl_dofs = []

        self._p = None  # bullet session to connect to
        self.go_id = -2  # bullet id for the loaded humanoid, to be overwritten
        self.torque = None  # if using torque control, the current torque vector to apply

        self.ll = None  # stores joint lower limits
        self.ul = None  # stores joint upper limits
        
        # for domain randomization  
        self.last_mass_scaling = np.array([1.0] * 13)   
        self.last_inertia_scaling = np.array([1.0] * 13)

        #front pair of legs 
        front_upper = self.scales[0]*0.24
        front_lower = self.scales[1]*0.24
        back_upper = self.scales[2]*0.24
        back_lower = self.scales[3]*0.24
        #link offsets
        front_upper_off = -front_upper*0.5
        front_lower_off = -front_lower*0.5
        back_upper_off = -back_upper*0.5
        back_lower_off = -back_lower*0.5
        #joint angles
        if self.design_joint_angle:
            Fang1 = -0.95*self.scales[4]
            Fang2 = -0.60*self.scales[5]
            Bang1 = -0.95*self.scales[6]
            Bang2 = -0.60*self.scales[7]
        elif self.joint_gap_env:
            Fang1 = -0.95+0.6 #-0.95*1.2
            Fang2 = -0.60-0.6 #-0.60*0.8
            Bang1 = -0.95 - 0.6#-0.95*0.7
            Bang2 = -0.60+ 0.6#-0.60*1.3
        else:
            Fang1 = -0.95
            Fang2 = -0.60
            Bang1 = -0.95
            Bang2 = -0.60

        FU = "{0:.6f}".format(front_upper)
        FL = "{0:.6f}".format(front_lower)
        BU = "{0:.6f}".format(back_upper)
        BL = "{0:.6f}".format(back_lower)
        OFU = "{0:.6f}".format(front_upper_off)
        OFL = "{0:.6f}".format(front_lower_off)
        OBU = "{0:.6f}".format(back_upper_off)
        OBL = "{0:.6f}".format(back_lower_off)
        Fang1 = "{0:.6f}".format(Fang1)
        Fang2 = "{0:.6f}".format(Fang2)
        Bang1 = "{0:.6f}".format(Bang1)
        Bang2 = "{0:.6f}".format(Bang2)
        

        self.proc_ID = str(os.getpid())
        self.exec_dir = os.getcwd() #<-- absolute dir the script is in
        #abs_file_path = self.exec_dir+'/my_pybullet_envs/assets/laikago/laikago_SCALE.urdf'
        abs_file_path = self.exec_dir+'/my_pybullet_envs/assets/quadruped/quadruped.urdf'
        with open(abs_file_path, 'r') as file :
            filedata = file.read()
            filedata = filedata.replace('FU', FU)
            filedata = filedata.replace('FL', FL)
            filedata = filedata.replace('BU', BU)
            filedata = filedata.replace('BL', BL)
            filedata = filedata.replace('OfsFrU', OFU)
            filedata = filedata.replace('OfsFrL', OFL)
            filedata = filedata.replace('OfsBaU', OBU)
            filedata = filedata.replace('OfsBaL', OBL)
            filedata = filedata.replace('FANG1', Fang1)
            filedata = filedata.replace('FANG2', Fang2)
            filedata = filedata.replace('BANG1', Bang1)
            filedata = filedata.replace('BANG2', Bang2)


        with open(self.exec_dir+'/my_pybullet_envs/assets/quadruped/temp_URDFs/'+self.proc_ID+'.urdf', 'w') as file:
            file.write(filedata)
        p.setPhysicsEngineParameter(enableFileCaching=0)

    def reset(
            self,
            bullet_client
    ):
        self._p = bullet_client
        if self.random_IC:
            if self.seeded_IC:
                with self.temp_seed(os.getpid()):
                    base_init_pos = utils.perturb(self.base_init_pos, 0.01/self.IC_factor,self.np_random)
                    base_init_euler = utils.perturb(self.base_init_euler, 0.1/self.IC_factor,self.np_random)
                    base_init_vel = utils.perturb([0.0] * 6, 0.2/self.IC_factor,self.np_random)
                    #base_init_vel = [0.0] * 6
            else:        
                base_init_pos = utils.perturb(self.base_init_pos, 0.01/self.IC_factor,self.np_random)
                base_init_euler = utils.perturb(self.base_init_euler, 0.1/self.IC_factor,self.np_random)
                base_init_vel = utils.perturb([0.0] * 6, 0.2/self.IC_factor,self.np_random)
                #base_init_vel = [0.0] * 6
        else:
            base_init_pos = self.base_init_pos.copy()
            base_init_euler = self.base_init_euler.copy()
            base_init_vel = [0.0] * 6
        
                            #e.g. max(1.3+1.4, 0.9+0.8) = 2.7 - 2 = 0.7 above nominal, scaled by 0.62 for angle and 0.24 for actual height
                            #(max(front_total, back_total)-(1+1))*cos(angle)*scale_to_length
                            #for nominal [1 1 1 1] heigh adjustment is zero.
        height_adjustment = (np.max([self.scales[0]+self.scales[1],self.scales[2]+self.scales[3]])-2.0)*0.62*0.24

        self.go_id = self._p.loadURDF(self.exec_dir+'/my_pybullet_envs/assets/quadruped/temp_URDFs/'+self.proc_ID+'.urdf',
                                      list(base_init_pos-[0.043794, 0.0, 0.03]+[0.,0.,height_adjustment]),
                                      list(self._p.getQuaternionFromEuler(list(base_init_euler))),
                                      #flags=self._p.URDF_USE_SELF_COLLISION,
                                      useFixedBase=0)
        os.remove(self.exec_dir+'/my_pybullet_envs/assets/quadruped/temp_URDFs/'+self.proc_ID+'.urdf')
        # self.print_all_joints_info()
        
        self._p.resetBaseVelocity(self.go_id, base_init_vel[:3], base_init_vel[3:])

        for j in range(self._p.getNumJoints(self.go_id)):
            self._p.changeDynamics(self.go_id, j, jointDamping=0.5)  # TODO

        if len(self.ctrl_dofs) == 0:
            for j in range(self._p.getNumJoints(self.go_id)):
                info = self._p.getJointInfo(self.go_id, j)
                joint_type = info[2]
                if joint_type == self._p.JOINT_PRISMATIC or joint_type == self._p.JOINT_REVOLUTE:
                    self.ctrl_dofs.append(j)

        # print("ctrl dofs:", self.ctrl_dofs)

        self.reset_joints(self.init_q, np.array([0.0] * len(self.ctrl_dofs)))

        # turn off root default control:
        # use torque control
        self._p.setJointMotorControlArray(
            bodyIndex=self.go_id,
            jointIndices=self.ctrl_dofs,
            controlMode=self._p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.ctrl_dofs))
        self.torque = [0.0] * len(self.ctrl_dofs)

        self.ll = np.array([self._p.getJointInfo(self.go_id, i)[8] for i in self.ctrl_dofs])
        self.ul = np.array([self._p.getJointInfo(self.go_id, i)[9] for i in self.ctrl_dofs])
        #bp()
        #if self.shrink_IC_dist:
        #    for _ in range(500):
        #        self._p.stepSimulation()
        #        input('enter')
        #    bp()
        #    q, dq = self.get_q_dq(self.ctrl_dofs)


        assert len(self.ctrl_dofs) == len(self.init_q)
        assert len(self.max_forces) == len(self.ctrl_dofs)
        assert len(self.max_forces) == len(self.ll)

    #def soft_reset(
    #        self,
    #        bullet_client
    #):
    #    self._p = bullet_client
    #    if self.random_IC:
    #        base_init_pos = utils.perturb(self.base_init_pos, 0.03/self.IC_factor)
    #        base_init_euler = utils.perturb(self.base_init_euler, 0.1/self.IC_factor)
    #        base_init_vel = utils.perturb([0.0] * 6, 0.2/self.IC_factor)
    #    else:
    #        base_init_pos = self.base_init_pos.copy()
    #        base_init_euler = self.base_init_euler.copy()
    #        base_init_vel = [0.0]*6

    #    self._p.resetBaseVelocity(self.go_id, base_init_vel[:3], base_init_vel[3:])
    #    self._p.resetBasePositionAndOrientation(self.go_id,
    #                                            list(base_init_pos),
    #                                            list(self._p.getQuaternionFromEuler(list(base_init_euler)))
    #                                            )

    #    self.reset_joints(self.init_q, np.array([0.0] * len(self.ctrl_dofs)))

    #def soft_reset_to_state(self, bullet_client, state_vec):
    #    # state vec following this order:
    #    # root dq [6]
    #    # root q [3+4(quat)]
    #    # all q/dq
    #    self._p = bullet_client
    #    self._p.resetBasePositionAndOrientation(self.go_id,
    #                                            state_vec[6:9],
    #                                            state_vec[9:13])
    #    self._p.resetBaseVelocity(self.go_id, state_vec[:3], state_vec[3:6])

    #    qdq = state_vec[13:]
    #    assert len(qdq) == len(self.ctrl_dofs) * 2
    #    init_noise_old = self.init_noise
    #    self.init_noise = False
    #    self.reset_joints(qdq[:len(self.ctrl_dofs)], qdq[len(self.ctrl_dofs):])
    #    self.init_noise = init_noise_old

    def reset_joints(self, q, dq):
        if self.init_noise:
            if self.seeded_IC:
                with self.temp_seed(os.getpid()):
                    init_q = utils.perturb(q, 0.01/self.IC_factor, self.np_random)
                    init_dq = utils.perturb(dq, 0.1/self.IC_factor, self.np_random)
            else:
                init_q = utils.perturb(q, 0.01/self.IC_factor, self.np_random)
                init_dq = utils.perturb(dq, 0.1/self.IC_factor, self.np_random)
        else:
            init_q = q#utils.perturb(q, 0.0, self.np_random)
            init_dq = dq #utils.perturb(dq, 0.0, self.np_random)

        for pt, ind in enumerate(self.ctrl_dofs):
            self._p.resetJointState(self.go_id, ind, init_q[pt], init_dq[pt])
        self.tar_q = init_q.copy()   

    def zero_out_dq(self):
        q, dq = self.get_q_dq(self.ctrl_dofs)
        dq = np.zeros(dq.shape)
        self.reset_joints(q,dq)
        self._p.resetBaseVelocity(self.go_id,[0.0]*3,[0.0]*3)


    def print_all_joints_info(self):
        for i in range(self._p.getNumJoints(self.go_id)):
            print(self._p.getJointInfo(self.go_id, i)[0:3],
                  self._p.getJointInfo(self.go_id, i)[8], self._p.getJointInfo(self.go_id, i)[9],
                  self._p.getJointInfo(self.go_id, i)[12])

    def apply_action(self, a):
        if self.control_mode == 'torque':
            self.torque = np.array(a) * self.max_forces
            self._p.setJointMotorControlArray(
                bodyIndex=self.go_id,
                jointIndices=self.ctrl_dofs,
                controlMode=self._p.TORQUE_CONTROL,
                forces=self.torque)
        elif self.control_mode == 'position':
            a = np.array(a)
            self.tar_q += a
            self.tar_q = np.clip(self.tar_q,self.ll,self.ul)
            self._p.setJointMotorControlArray(
                bodyIndex=self.go_id,
                jointIndices = self.ctrl_dofs,
                controlMode = self._p.POSITION_CONTROL,
                targetPositions = self.tar_q,
                forces = self.max_forces)    

    def get_q_dq(self, dofs):
        joints_state = self._p.getJointStates(self.go_id, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    def get_link_com_xyz_orn(self, link_id, fk=1):
        # get the world transform (xyz and quaternion) of the Center of Mass of the link
        assert link_id >= -1
        if link_id == -1:
            link_com, link_quat = self._p.getBasePositionAndOrientation(self.go_id)
        else:
            link_com, link_quat, *_ = self._p.getLinkState(self.go_id, link_id, computeForwardKinematics=fk)
        return list(link_com), list(link_quat)

    def get_robot_raw_state_vec(self):
        # state vec following this order:
        # root dq [6]
        # root q [3+4(quat)]
        # all q/dq
        state = []
        base_vel, base_ang_vel = self._p.getBaseVelocity(self.go_id)
        state.extend(base_vel)
        state.extend(base_ang_vel)
        root_pos, root_orn = self.get_link_com_xyz_orn(-1)
        state.extend(root_pos)
        state.extend(root_orn)
        q, dq = self.get_q_dq(self.ctrl_dofs)
        state.extend(q)
        state.extend(dq)
        return state

    def get_robot_observation(self, with_vel=False):
        obs = []

        # root z and root rot mat (1+9)
        root_pos, root_orn = self.get_link_com_xyz_orn(-1)
        root_x, root_y, root_z = root_pos
        obs.extend([root_z])
        obs.extend(pybullet.getMatrixFromQuaternion(root_orn))

        # root lin vel (3)
        base_vel, base_ang_vel = self._p.getBaseVelocity(self.go_id)
        obs.extend(base_vel)

        # non-root joint q (12)
        q, dq = self.get_q_dq(self.ctrl_dofs)
        obs.extend(q)

        # feet (offset from root) (12)
        for link in self.feet:
            pos, _ = self.get_link_com_xyz_orn(link, fk=1)
            pos[0] -= root_x
            pos[1] -= root_y
            pos[2] -= root_z
            obs.extend(pos)

        length_wo_vel = len(obs)

        obs.extend(base_ang_vel)    # (3)
        obs.extend(dq)      # (12)

        obs = np.array(obs) * self.robo_obs_scale

        if not with_vel:
            obs = obs[:length_wo_vel]

        return list(obs)

    def is_root_com_in_support(self):
        root_com, _ = self._p.getBasePositionAndOrientation(self.go_id)
        feet_max_x = -1000
        feet_min_x = 1000
        feet_max_y = -1000
        feet_min_y = 1000
        for foot in self.feet:
            x, y, _ = list(self.get_link_com_xyz_orn(foot)[0])
            if x > feet_max_x:
                feet_max_x = x
            if x < feet_min_x:
                feet_min_x = x
            if y > feet_max_y:
                feet_max_y = y
            if y < feet_min_y:
                feet_min_y = y
        return (feet_min_x - 0.05 < root_com[0]) and (root_com[0] < feet_max_x + 0.05) \
            and (feet_min_y - 0.05 < root_com[1]) and (root_com[1] < feet_max_y + 0.05)

    def randomize_robot(self, mass_scale, inertia_scale, power_scale, damping_scale):    
        # the scales being numpy vectors    
        for link_ind, dof in enumerate([-1] + self.ctrl_dofs):  
            dyn = self._p.getDynamicsInfo(self.go_id, dof)  
            mass = dyn[0] / self.last_mass_scaling[link_ind] * mass_scale[link_ind] 
            lid = np.array(dyn[2]) / self.last_inertia_scaling[link_ind] * inertia_scale[link_ind]  
            self._p.changeDynamics(self.go_id, dof, mass=mass)  
            self._p.changeDynamics(self.go_id, dof, localInertiaDiagonal=tuple(lid))    
    
        for joint_ind, j in enumerate(self.ctrl_dofs):  
            self._p.changeDynamics(self.go_id, j, jointDamping=damping_scale[joint_ind])    
        self.max_forces = self.nominal_max_forces * power_scale 
    
        self.last_mass_scaling = np.copy(mass_scale)    
        self.last_inertia_scaling = np.copy(inertia_scale)


    @contextlib.contextmanager
    def temp_seed(self,seed):
        state = self.np_random.get_state()
        self.np_random.seed(seed)
        try:
            yield
        finally:
            self.np_random.set_state(state)
