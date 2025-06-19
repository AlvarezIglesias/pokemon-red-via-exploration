import gym
from pathlib import Path
import uuid

#from stable_baselines3.common.policies import  https://stable-baselines.readthedocs.io/en/master/modules/policies.html#stable_baselines.common.policies.CnnLstmPolicy
# https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from exploration_gym_env import ExplorationGymEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EventCallback, BaseCallback


class TensorboardCallback(BaseCallback):

    def __init__(self, n_steps: int, verbose=0, workers=8):
        super().__init__(verbose)
        self.n_steps = n_steps
        self.last_time_trigger = 0
        self.workers = workers

    def _on_training_start(self):
        # total_rewards = np.array(self.training_env.get_attr("total_reward"))
        # initial_rewards = np.array(self.training_env.get_attr("initial_reward"))
        self.logger.record("time/total_reward", 0)
        self.logger.record("time/delta_reward", 0)

    def _on_step(self) -> bool:
        # print(((self.n_calls + 1) % self.n_steps))
        if ((self.n_calls + 1) % self.n_steps) == 0: 
            total_rewards = np.array(self.training_env.get_attr("total_reward"))
            delta_reward = np.array(self.training_env.get_attr("delta_reward"))
            self.logger.record("time/total_reward", np.mean(total_rewards))
            self.logger.record("time/delta_reward", np.mean(delta_reward))
            print("logging to tensorboard!", np.mean(total_rewards))
        

        return True

def make_env(config, seed=0, rank=0):
    def _init():
        config['rank'] = rank
        env = ExplorationGymEnv(config)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

num_cpu = 9
run_name = "cnn_lstm_ep_pre" 
sess_id = str(uuid.uuid4())[:8]
# sess_id = "3b1cca56"
# rom = env_config['gb_path'].split("/")[-1]
rom = "PR"
ep_length = 2048 * 10 // 8
env_config = {
            'rank': 0,
            'init_state': './states/initial.state',
            'checkpoint_path': f'./files/{rom}_{run_name}_{sess_id}/checkpoints',
            'img_path': f'./files/{rom}_{run_name}_{sess_id}/imgs',
            'gb_path': './roms/PokemonRed_water.gb',
            'input_resolution': (144,160),
            'headless': True, 
            'gbc_mode': False,
            'print_rewards': True,
            'debug': False,
            'action_freq': 10, 
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
            'inmediate_offsets' : [16,32],
            'inmediate_threshold': 0.04, #pokemon 0.04
            'stay_threshold_margin': 0.005,
            'inmediate_entry_point_threshold': 0.04,
            'close_entry_point_threshold': 0.04,
            'close_entry_point_offsets' : [16],
            'past_entry_point_threshold': 0.04,
            'past_entry_point_list_max_size' : 100,
            'repeat_ward_max_size' : 100,

        }

# Custom MLP policy of 2 layers of size 128 each
class CustomPolicy(RecurrentActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                            net_arch=dict(pi=[128, 128], # defecto 64 64 
                                                           vf=[128, 128]),  # defecto 64 64 
                                            n_lstm_layers=1, lstm_hidden_size=256)  # defecto 1, 254

if __name__ == '__main__':
    
    sess_path = Path(f'./files/{rom}_{run_name}_{sess_id}')
    env = SubprocVecEnv([make_env(env_config, rank=i) for i in range(num_cpu)]) #make_env(0)() #
    # model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=128, n_epochs=3, gamma=0.998, tensorboard_log=sess_path, learning_rate=0.00010)
    model = RecurrentPPO(CustomPolicy, env, verbose=1, n_steps=ep_length, batch_size=128, n_epochs=3, gamma=0.998, tensorboard_log=sess_path, learning_rate=0.00010, use_preamble=True, cores=num_cpu)
    # model = model.load(f"E:/ALVARO/{rom}_{run_name}_{sess_id}//PR_11110400_steps.zip", env, print_system_info=True)
    model.learn(total_timesteps=(ep_length)*num_cpu*5000*1, reset_num_timesteps=False, tb_log_name=f'{rom}_{run_name}', progress_bar=True, callback=CallbackList([TensorboardCallback(ep_length, num_cpu), CheckpointCallback(save_freq=ep_length, save_path=sess_path, name_prefix=rom)]))

