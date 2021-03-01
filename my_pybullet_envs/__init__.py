import gym
from gym.envs.registration import registry, make, spec

from .quadruped_env_scaled import QuadrupedBulletEnvSCALE
from .quadruped_env_des_randomization import QuadrupedBulletEnvDesRand


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# ------------bullet-------------


register(
    id="QuadrupedBulletScaledEnv-v4",
    entry_point="my_pybullet_envs:QuadrupedBulletEnvSCALE",
    max_episode_steps=500,
)

register(
    id="QuadrupedBulletDesRandEnv-v4",
    entry_point="my_pybullet_envs:QuadrupedBulletEnvDesRand",
    max_episode_steps=500,
)



def getList():
    btenvs = [
        "- " + spec.id
        for spec in gym.envs.registry.all()
        if spec.id.find("Bullet") >= 0
    ]
    return btenvs
