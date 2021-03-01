from .quadruped_scaled import QuadrupedSCALE
from pybullet_utils import bullet_client
import pybullet
import pybullet_data
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
from gan import utils
from collections import deque

import os
import inspect
from pdb import set_trace as bp
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class QuadrupedBulletEnvSCALE(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 scales = None,#[1.5, 1.5, 1.5, 1.5],
                 scale_range = [0.5,1.5],#[0.5, 1.5],
                 init_noise=True,
                 act_noise=True,
                 obs_noise=True,
                 control_skip=10,

                 max_tar_vel=2.5,
                 energy_weight=0.1,
                 jl_weight=0.5,
                 ab=4.5,
                 q_pen_weight=0.35,
                 dq_pen_weight=0.001,
                 vel_r_weight=4.0,

                 enlarge_act_range=0.0,     # during data collection, make pi softer
                 train_universal_policy = False,
                 soft_floor_env=False,
                 deform_floor_env=False,
                 low_power_env=False,
                 emf_power_env=False,
                 joint_gap_env = False,
                 randomization_train=False,
                 randomforce_train=False,
                 random_IC = True, # variation in the spawn position and orientation of the robot
                 control_mode = 'torque', #choose 'torque' or 'position'
                 action_scale = 0.01, #action scale only for position control
                 sigma = 0.01, #only used for universal policy with changing scale distribution
                 shrink_IC_dist = False, #reduce variation in IC's during sim-to-real transfer
                 seeded_IC = False, # fix seed of IC aross different rollouts of the same process

                 design_joint_angle = False #whether or not to include joint angles as design parameters in the quadruped URDF
                 ):

        self.render = render
        self.scales = scales
        self.scale_range = scale_range
        self.train_universal_policy = train_universal_policy
        self.init_noise = init_noise
        self.obs_noise = obs_noise
        self.act_noise = act_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 500.

        self.max_tar_vel = max_tar_vel
        self.energy_weight = energy_weight
        self.jl_weight = jl_weight
        self.ab = ab
        self.q_pen_weight = q_pen_weight
        self.dq_pen_weight = dq_pen_weight
        self.vel_r_weight = vel_r_weight

        self.enlarge_act_range = enlarge_act_range

        self.soft_floor_env = soft_floor_env
        self.deform_floor_env = deform_floor_env
        self.low_power_env = low_power_env
        self.emf_power_env = emf_power_env
        self.joint_gap_env = joint_gap_env
        self.randomization_train = randomization_train
        self.randomforce_train = randomforce_train
        self.control_mode = control_mode
        self.action_scale = action_scale
        self.sigma = sigma
        self.random_IC = random_IC
        self.shrink_IC_dist = shrink_IC_dist
        self.seeded_IC = seeded_IC
        self.design_joint_angle = design_joint_angle

        self.randomize_params = {}
        self.n_design_params = 8 if self.design_joint_angle else 4

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.np_random = None
        self.robot = QuadrupedSCALE(init_noise=self.init_noise,
                                     time_step=self._ts,
                                     np_random=self.np_random,
                                     control_mode=self.control_mode,
                                     random_IC =self.random_IC,
                                     shrink_IC_dist = self.shrink_IC_dist,
                                     seeded_IC = self.seeded_IC)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        self.floor_id = None
        self.velx = 0

        self.behavior_past_obs_t_idx = np.array([0, 3, 6, 9])  # t-3. t-6. t-9
        #self.behavior_past_obs_t_idx = np.array([0, 4, 8])


        self.past_obs_array = deque(maxlen=11)
        self.past_act_array = deque(maxlen=10)

        self.init_state = None
        obs = self.reset()  # and update init obs

        self.action_dim = len(self.robot.ctrl_dofs)
        self.act = [0.0] * len(self.robot.ctrl_dofs)
        self.action_space = gym.spaces.Box(low=np.array([-1.] * self.action_dim),
                                           high=np.array([+1.] * self.action_dim))
        self.obs_dim = len(obs)
        # print(self.obs_dim)
        obs_dummy = np.array([1.12234567] * self.obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf * obs_dummy, high=np.inf * obs_dummy)



    def reset(self):
        self.SCALES = self.set_design()
        if self.deform_floor_env or self.soft_floor_env:
            self._p.resetSimulation(self._p.RESET_USE_DEFORMABLE_WORLD)
            # always use hard reset if soft-floor-env
        else:
            self._p.resetSimulation()
            
        self._p.setTimeStep(self._ts)
        self._p.setGravity(0, 0, -10)
        self._p.setPhysicsEngineParameter(numSolverIterations=100)

        self.robot = QuadrupedSCALE(scales=self.SCALES, init_noise=self.init_noise,
                                     time_step=self._ts,
                                     np_random=self.np_random,
                                     control_mode=self.control_mode,
                                     random_IC =self.random_IC,
                                     shrink_IC_dist = self.shrink_IC_dist,
                                     seeded_IC = self.seeded_IC,
                                     design_joint_angle = self.design_joint_angle,
                                     joint_gap_env = self.joint_gap_env)
        self.robot.reset(self._p)

        if self.soft_floor_env:
            self.floor_id = self._p.loadURDF(   
                      os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1    
                    )   
                    # reset 
            for ind in self.robot.feet: 
                self._p.changeDynamics(self.robot.go_id, ind,   
                                               contactDamping=100, contactStiffness=100)    
                self._p.changeDynamics(self.floor_id, -1, contactDamping=50, contactStiffness=100)

        elif self.deform_floor_env: 
            self.floor_id = self._p.loadURDF(   
                os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, -10.02], useFixedBase=1   
            )   

            # something is wrong with this API  
            # seems that you must add cube.obj to pybullet_data folder for this to work 
            # it cannot search relative path in the repo    
            _ = self._p.loadSoftBody("cube_fat.obj", basePosition=[7, 0, -15], scale=60, mass=4000., 
                                     useNeoHookean=0,   
                                     useBendingSprings=1, useMassSpring=1, springElasticStiffness=60000,    
                                     springDampingStiffness=150, springDampingAllDirections=1,  
                                     useSelfCollision=0,    
                                     frictionCoeff=1.0, useFaceContact=1)

        else:
            if self.randomization_train:    
                self.set_randomize_params() 
                self.robot.randomize_robot( 
                            self.randomize_params["mass_scale"],    
                            self.randomize_params["inertia_scale"], 
                            self.randomize_params["power_scale"],   
                            self.randomize_params["joint_damping"]  
                        )   

            fric = self.randomize_params["friction"] if self.randomization_train else 0.5   
            resti = self.randomize_params["restitution"] if self.randomization_train else 0.0
            self.floor_id = self._p.loadURDF(
              os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1
            )
            
            self._p.changeDynamics(self.floor_id, -1, lateralFriction=fric)    
            self._p.changeDynamics(self.floor_id, -1, restitution=resti)    
        
            for ind in self.robot.feet: 
                self._p.changeDynamics(self.robot.go_id, ind,   
                                               lateralFriction=1.0, restitution=1.0)
        
        self.init_state = self._p.saveState()
        
        if self.low_power_env:
        # # for pi51, 53 (52)
            self.robot.max_forces = [30.0] * 3 + [15.0] * 3 + [30.0] * 6

        self._p.stepSimulation()

        for foot in self.robot.feet:
            cps = self._p.getContactPoints(self.robot.go_id, self.floor_id, foot, -1)
            if len(cps) > 0:
                print("warning")

        if self.shrink_IC_dist:
            if self.deform_floor_env:
                feet_locs = np.array([self._p.getLinkState(self.robot.go_id, foot)[0][2] for foot in self.robot.feet])
                while any(feet_locs>0.0):
                    self._p.stepSimulation()
                    feet_locs = np.array([self._p.getLinkState(self.robot.go_id, foot)[0][2] for foot in self.robot.feet])
                #print([self._p.getLinkState(self.robot.go_id, foot)[0][2] for foot in self.robot.feet])

            else:
                feet_not_in_contact = [(len(self._p.getContactPoints(self.robot.go_id, self.floor_id, foot, -1))==0) for foot in self.robot.feet]
                while any(feet_not_in_contact):
                    self._p.stepSimulation()
                    feet_not_in_contact = [(len(self._p.getContactPoints(self.robot.go_id, self.floor_id, foot, -1))==0) for foot in self.robot.feet]
            self.robot.zero_out_dq()

        self.timer = 0
        self.past_obs_array.clear()
        self.past_act_array.clear()

        obs = self.get_extended_observation()

        return np.array(obs)

    def set_randomize_params(self): 
        self.randomize_params = {   
            # robot 
            "mass_scale": self.np_random.uniform(0.8, 1.2, 13), 
            "inertia_scale": self.np_random.uniform(0.5, 1.5, 13),  
            "power_scale": self.np_random.uniform(0.8, 1.2, 12),    
            "joint_damping": self.np_random.uniform(0.2, 2.0, 12),  
            # act/obs latency   
            "act_latency": self.np_random.uniform(0, 0.02), 
            "obs_latency": self.np_random.uniform(0, 0.02), 
            # contact   
            "friction": self.np_random.uniform(0.4, 1.25),  
            "restitution": self.np_random.uniform(0., 0.4), 
            }

    def step(self, a):

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_0 = root_pos[0]

        # TODO: parameter space noise.
        # make pi softer during data collection, different from hidden act_noise below
        # print(self.enlarge_act_range)
        a = utils.perturb(a, self.enlarge_act_range, self.np_random) #this changes a into double precision even if enlarge_act_range = 0.0
        if self.control_mode=='torque':
            a = np.tanh(a)
        elif self.control_mode=='position':
            a = a*self.action_scale 

        # ***push in deque the a after tanh
        utils.push_recent_value(self.past_act_array, a)

        act_latency = self.randomize_params["act_latency"] if self.randomization_train else 0   
        
        a0 = np.array(self.past_act_array[0])   
        a1 = np.array(self.past_act_array[1])   
        interp = act_latency / 0.02 
        a = a0 * (1 - interp) + a1 * interp

        if self.act_noise and self.control_mode=='torque':
            a = utils.perturb(a, 0.05, self.np_random)


        if self.emf_power_env:  
            _, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)   
            max_force_ratio = np.clip(1 - dq/8., 0, 1)  
            a *= max_force_ratio

        past_info = self.construct_past_traj_window()

        for _ in range(self.control_skip):
            if a is not None:
                self.act = a
                self.robot.apply_action(a)

            if self.randomforce_train:
                for foot_ind, link in enumerate(self.robot.feet):
                    # first dim represents fz
                    fz = np.random.uniform(-80, 80)
                    # second dim represents fx
                    fx = np.random.uniform(-80, 80)
                    # third dim represents fy
                    fy = np.random.uniform(-80, 80)

                    utils.apply_external_world_force_on_local_point(self.robot.go_id, link,
                                                                    [fx, fy, fz],
                                                                    [0, 0, 0],
                                                                    self._p)
            self._p.stepSimulation()
            if self.render:
                time.sleep(self._ts * 1.0)
            self.timer += 1

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_1 = root_pos[0]
        self.velx = (x_1 - x_0) / (self.control_skip * self._ts)

        q, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)

        reward = self.ab  # alive bonus
        tar = np.minimum(5.0*self.timer / 500, self.max_tar_vel)
        reward += np.minimum(self.velx, tar) * self.vel_r_weight
        # print("v", self.velx, "tar", tar)
        reward += -self.energy_weight * np.square(a).sum()
        # print("act norm", -self.energy_weight * np.square(a).sum())

        pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
        q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
        joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
        reward += -self.jl_weight * joints_at_limit
        # print("jl", -self.jl_weight * joints_at_limit)

        reward += -np.minimum(np.sum(np.square(dq)) * self.dq_pen_weight, 5.0)
        weight = np.array([2.0, 1.0, 1.0] * 4)
        reward += -np.minimum(np.sum(np.square(q - self.robot.init_q) * weight) * self.q_pen_weight, 5.0)

        # reward = self.ab
        # tar = np.minimum(self.timer / 500, self.max_tar_vel)
        # reward += np.minimum(self.velx, tar) * self.vel_r_weight
        # # print("v", self.velx, "tar", tar)
        #
        # # reward += np.maximum((self.max_tar_vel - tar) * self.vel_r_weight - 3.0, 0.0)     # alive bonus
        #
        # reward += -self.energy_weight * np.linalg.norm(a)
        # # print("act norm", -self.energy_weight * np.square(a).sum())
        #
        # q, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)
        # # print(np.max(np.abs(dq)))
        # pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
        # q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
        # joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
        # reward += -self.jl_weight * joints_at_limit
        # # print("jl", -self.jl_weight * joints_at_limit)
        #
        # reward += -np.minimum(np.linalg.norm(dq) * self.dq_pen_weight, 5.0)
        # weight = np.array([2.0, 0.2, 1.0] * 4)
        # reward += -np.minimum(np.linalg.norm((q - self.robot.init_q) * weight) * self.q_pen_weight, 5.0)
        # # print("vel pen", -np.minimum(np.sum(np.abs(dq)) * self.dq_pen_weight, 5.0))
        # # print("pos pen", -np.minimum(np.sum(np.square(q - self.robot.init_q)) * self.q_pen_weight, 5.0))

        y_1 = root_pos[1]
        reward += -y_1 * 0.5
        # print("dev pen", -y_1*0.5)
        height = root_pos[2]

        # in_support = self.robot.is_root_com_in_support()
        # print("______")
        # print(in_support)

        # print("h", height)
        # print("dq.", np.abs(dq))
        # print((np.abs(dq) < 50).all())

        cps = self._p.getContactPoints(bodyA=self.robot.go_id, linkIndexA=-1,bodyB=self.floor_id)
        body_in_contact = (len(cps) > 0)
        not_done = not body_in_contact
        if (self.deform_floor_env or self.soft_floor_env) and height <0.12:  # additional check for soft/deformable floor
            not_done = False

        # print("------")
        obs = self.get_extended_observation()
        # rpy = self._p.getEulerFromQuaternion(obs[8:12])

        #   # for data collection
        #    not_done = (np.abs(dq) < 90).all() and (height > 0.3) and (height < 1.0)
        #   # TODO
        #   # not_done = (np.abs(dq) < 90).all() and (height > 0.3) and (height < 1.0) and in_support
        #   # not_done = True

        past_info += [self.past_obs_array[0]]       # s_t+1

        return obs, reward, not not_done, {"sas_window": past_info,"scales":self.SCALES}
    
    def set_design(self):
        #print(self.sigma)
        if self.train_universal_policy:
            np.random.seed([os.getpid()+int(time.time())]) #different for each process and each iteration!
            if self.scales is None:
                self.SCALES = np.random.uniform(low=self.scale_range[0],high=self.scale_range[1],size=self.n_design_params)
            else:
                self.SCALES = np.clip(self.scales+self.sigma*np.random.normal(size=self.n_design_params),self.scale_range[0],self.scale_range[1])
        else:
            self.SCALES = self.scales 

        return self.SCALES

    def construct_past_traj_window(self):
        # st, ... st-9, at, ..., at-9
        # call this before s_t+1 enters deque
        # order does not matter as long as it is the same in policy & expert batch
        # print(list(self.past_obs_array) + list(self.past_act_array))
        return list(self.past_obs_array) + list(self.past_act_array)

    def get_dist(self):
        return self.robot.get_link_com_xyz_orn(-1)[0][0]

    def get_ave_dx(self):
        return self.velx

    def get_extended_observation(self):
        obs = self.robot.get_robot_observation(with_vel=False)

        if self.obs_noise:
            obs = utils.perturb(obs, 0.1, self.np_random)

        utils.push_recent_value(self.past_obs_array, obs)

        # print(obs[-4:])

        # obs_all = []
        # list_past_obs = list(self.past_obs_array)
        # for t_idx in self.behavior_past_obs_t_idx:
        #     obs_all.extend(list_past_obs[t_idx])

        obs_latency = self.randomize_params["obs_latency"] if self.randomization_train else 0   
    
        obs_all_0 = utils.select_and_merge_from_s_a(    
            s_mt=list(self.past_obs_array), 
            a_mt=list(self.past_act_array), 
            s_idx=self.behavior_past_obs_t_idx, 
            a_idx=np.array([])  
        )
        obs_all_1 = utils.select_and_merge_from_s_a(    
            s_mt=list(self.past_obs_array), 
            a_mt=list(self.past_act_array), 
            s_idx=self.behavior_past_obs_t_idx + 1, 
            a_idx=np.array([])  
        )   
    
        interp = obs_latency / 0.02 
        obs_all = obs_all_0 * (1 - interp) + obs_all_1 * interp
        observation = list(obs_all)
        observation.extend(self.SCALES)
        return observation

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s

    def cam_track_torso_link(self):
        distance = 2.0
        yaw = -30
        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        distance -= root_pos[1]
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, [root_pos[0], 0.0, 0.1])

    def update_mu_sigma(self,data):
        if data[0] is not None:
            #print('scales changed',flush=True)
            self.scales = data[0]
        self.sigma = data[1]
        #print(self.sigma)
        #return self.sigma

