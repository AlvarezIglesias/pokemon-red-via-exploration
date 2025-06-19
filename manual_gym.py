from gym import Env
import numpy as np
from exploration_gym_env import ExplorationGymEnv
from os.path import exists
from pathlib import Path
import uuid
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import time
actions = {"s": 0,
           "w": 1,
           "a": 2,
           "d": 3,
           "k": 4,
           "l": 5,
           "" : 5,
           "q" : 6,
           "o" : -1,
           "z": -1,
           "x": -1,
           "p": -1,
           "m": -1}

acumulative_input = ""

def fixed_input():
    while True:
        action = input("Action: ")
        for a in list(action):
            if a in actions:
                yield a

input_handler = fixed_input()
ep_length = 2048 * 10 // 8

env_config = {
            'rank': 0,
            'init_state': './states/SK.state',
            'checkpoint_path': 'C:/Users/chisp/Documents/GitHub/TFM//files/checkpoints',
            'img_path': 'C:/Users/chisp/Documents/GitHub/TFM//files/imgs',
            'gb_path': './roms/SurvivalKids.gbc',
            'input_resolution': (144,160),
            'headless': False, 
            'gbc_mode': True,
            'print_rewards': True,
            'debug': False,
            'action_freq': 20, 
            'random_movement': False,
            'max_steps': ep_length,
            'smooth_frames': 10,
            'smooth_frames_thresh': 0.004,
            # 'frame_comparision_margin': 16,
            'preamble_size': ep_length,
            'new_map_multiplier': 1/25,
            'use_distance': False,
            'distance_k' : 1/100,
            'save_checkpoints': True,
            'use_checkpoints': True,
            'prioritize_recent_checkpoints': True,
            'prioritize_checkpoint_value': 'total',
            'baby_ranks': [0], #list of ranks that, even when using checkpoints, start from the begining
            'initial_map_resolution': (500,500),
            'chunk_extension_size': 160,
            'inmediate_offsets' : [8],
            'inmediate_threshold': 0.05, #pokemon 0.04
            'stay_threshold_margin': 0.005,
            'inmediate_entry_point_threshold': 0.04,
            'close_entry_point_threshold': 0.04,
            'close_entry_point_offsets' : [8],
            'past_entry_point_threshold': 0.04,
            'past_entry_point_list_max_size' : 100,
            'repeat_ward_max_size' : 100,

        }


env = ExplorationGymEnv(env_config)
env.reset()

for i in range(6000):
    action = input_handler.__next__()
    acumulative_input += action
    if action == "z": 
        env.save_checkpoint_efficient()     
    elif action == "x":
        env.load_random_checkpoint()
        obs, rewards, term, trunc, info = env.step(5)
        frame_actual = np.array(obs)[:,:,0]
    elif action == "p":
        env.reward_engine.save("./imgs/")
    elif action == "m":
        env.save_checkpoint_efficient_dict()
    elif action == "o":
        with open("zero_crystal.state", "wb") as f:
            env.pyboy.save_state(f)
    else:
        action = actions[action]
        obs, rewards, term, trunc, info = env.step(action)
        frame_actual = np.array(obs)[:,:,0]


print(acumulative_input)

