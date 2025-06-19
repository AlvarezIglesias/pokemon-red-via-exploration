import uuid 
import os
import glob
import json
import zipfile
import io
import numpy as np
from pyboy import PyBoy
import pandas as pd
from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from reward_engine import Engine
import pickle
import random
import json

class ExplorationGymEnv(Env):


    def __init__(self, config):
        
        self.i = 0

        self.config = config
        self.reward_engine = Engine(config)
        self.act_freq = config['action_freq']
        self.extra_buttons = False
        self.init_state = config['init_state']
        self.rom = config['gb_path']
        self.step_count = 0
        self.rank = config['rank']
        self.smooth_frames = config['smooth_frames']
        self.print_rewards = config['print_rewards']
        self.max_steps = config['max_steps']
        self.checkpoint_path = config['checkpoint_path']
        self.img_path = config['img_path']
        self.total_reward = 0.0
        self.initial_reward = 0.0
        self.checkpoint_list = []
        self.preamble = []
        self.use_checkpoints = config['use_checkpoints']
        self.delta_reward = 0

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            # WindowEvent.PRESS_BUTTON_START,
            # WindowEvent.PRESS_BUTTON_SELECT,
        ]

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START,
            # WindowEvent.RELEASE_BUTTON_SELECT,
        ]

        self.output_shape = (144, 160, 3)

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)

        self.headless = config['headless']
        head = 'null' if self.headless else 'SDL2'

        #log_level("ERROR")
        self.pyboy = PyBoy(
                self.rom,
                window=head,
                cgb=config['gbc_mode']
            )

        self.screen = self.pyboy.screen

        if not config['headless']:
            self.pyboy.set_emulation_speed(0)
        

            
        # self.reset()
    
    def reset(self, seed=None, options=None):

        self.delta_reward = self.total_reward - self.initial_reward
        self.save()

        self.seed = seed
        # if self.total_reward > 1:
        #     self.save_checkpoint_efficient_dict()
        self.load_random_checkpoint()
        self.preamble = self.get_preamble()
        obs = self.render()
        self.reward_engine.append(obs[:,:,0], 5)
        return obs, {}  # empty info dict
    
    
    def send_action(self, action=None):
        
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        # if self.headless:
        #     self.pyboy._rendering(False)

        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action==None: continue
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < len(self.valid_actions):
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])

            if i == self.act_freq-1:
                self.pyboy.tick(1,True)
            else:
                self.pyboy.tick(1, False)


    def step(self, action):
    
        self.send_action(action)
        unstable = True
        last_obs = self.render().astype(float)
        while unstable:
            self.pyboy.tick(self.smooth_frames, True)
            obs = self.render().astype(float)
            total_pixels = self.config['input_resolution'][0] * self.config['input_resolution'][1] * 3
            dif = np.sum((abs(obs - last_obs)/255.0))/total_pixels
            if dif < self.config['smooth_frames_thresh']:
                unstable = False
            else:
                last_obs = obs
        score = self.reward_engine.append(obs[:,:,0], action)
        #score -= 0.05 # Factor aburrimiento

        if not self.headless:
            self.reward_engine.show()

        self.total_reward += score
        self.print_info()
        self.step_count += 1
        if self.step_count >= self.max_steps:
            return obs, score, False, True, {}
        else:
            return obs, score, False, False, {}
    
    def print_info(self):
        prog_string = ""
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
        prog_string += f' sum: {self.total_reward:5.2f}'
        prog_string += f' delta: {self.delta_reward:5.2f}'
        print(f'\r{prog_string}', end='', flush=True)



    def render(self):
        game_pixels_render = self.screen.ndarray[:,:,1:] #144,160,4  #.screen_ndarray() # (144, 160, 3)
        return np.array(game_pixels_render)

    def save(self):
        path = self.img_path + "/" + str(self.rank) + "_agent/"
        os.makedirs(path, exist_ok=True) 
        files = glob.glob(path + "*")
        for f in files:
            try:
                os.remove(f)
            except:
                print("No se ha podido borrar un archivo")
        self.reward_engine.save(path + str(self.step_count))
        print("Rewards by agent ", self.rank, ": Total", self.total_reward, " , Delta: ", self.delta_reward)

    def close(self):
        pass

    def load_initial_state(self):

        if self.init_state != "" and self.init_state != None:
            try:
                with open(self.init_state, "rb") as f:
                    self.pyboy.load_state(f)
            except:
                print("INITIAL STATE NOT FOUND")
            self.step_count = 0
            self.total_reward = 0.0
            self.reward_engine = Engine(self.config)

    def save_checkpoint_efficient_dict(self):

        if self.rank in self.config['baby_ranks']: return

        check_id = str(uuid.uuid4())[:8]
        check_path = (f'checkpoint_{check_id}.checkpoint')
        path = self.checkpoint_path + '/' + check_path
        
        zip_buffer = io.BytesIO()

        # PyBoy state
        pyboy_state = io.BytesIO()
        pyboy_state_data = None
        with pyboy_state as f:
            f.seek(0)
            self.pyboy.save_state(f)
            pyboy_state_data = pyboy_state.getvalue()

        # Json data
        stats_dict = {'reward' : self.total_reward, 'rom_path': self.rom, }
        state_stats_json = json.dumps(stats_dict, ensure_ascii=False)

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('pyboy.state', pyboy_state_data)
            zip_file.writestr('sim_stats.json', state_stats_json)
            self.reward_engine.efficient_dict_save(zip_file)


        # Save the zip archive to a file on the disk
        with open(path + 'arrays_compressed.zip', 'wb') as f:
            zip_buffer.seek(0)
            f.write(zip_buffer.getvalue())

        if self.config['prioritize_recent_checkpoints'] and self.config['prioritize_checkpoint_value'] == 'delta':
            self.checkpoint_list.append((path + 'arrays_compressed.zip', self.delta_reward))
        elif self.config['prioritize_recent_checkpoints'] and self.config['prioritize_checkpoint_value'] == 'total':
            self.checkpoint_list.append((path + 'arrays_compressed.zip', self.total_reward))

        # print("ZIP file with NumPy arrays created and saved as 'arrays_compressed.zip'")


    
    def load_random_checkpoint(self):
        os.makedirs(self.checkpoint_path, exist_ok=True) 
        files = glob.glob(self.checkpoint_path + "/*")
        if len(files) == 0 or not self.use_checkpoints or self.rank in self.config['baby_ranks']: # or random.uniform(0,1) > 0.3
            self.load_initial_state()
        else:
            if not self.config['prioritize_recent_checkpoints']:
                r = random.choice(files)
                self.load_checkpoint_efficient(r)
            else:
                # weights = [math.exp(i - len(self.checkpoint_list))*0.3 for i in range(len(self.checkpoint_list))]  # Exponentially decay weights

                # if len(self.checkpoint_list) < 50:
                #     r = random.choice(files)
                #     self.load_checkpoint_efficient(r)
                # else:
                weights = [reward for (file, reward) in self.checkpoint_list]
                picked = random.choices(self.checkpoint_list, weights=weights, k=1)
                self.load_checkpoint_efficient(picked[0][0]) #long time since i last used this, might crash


    def load_checkpoint(self, checkpoint_path):

        with open(checkpoint_path, "rb") as save_file:
            save_data = pickle.load(save_file)

        self.rom = save_data.rom
        #self.reward_engine.close()
        self.reward_engine = save_data.engine
        self.total_reward = save_data.reward
        self.initial_reward = save_data.reward
        state = io.BytesIO(save_data.pypi_state)
        self.pyboy.load_state(state)
        self.step_count = 0
        # print(self.total_reward)
        #self.reward_engine.show()
        # print("cargado!")

    def load_checkpoint_efficient(self, checkpoint_file):
        path = checkpoint_file

        # Load the zip archive from the disk
        with open(path, 'rb') as f:
            zip_buffer = io.BytesIO(f.read())

        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            # Load the pyboy state
            with zip_file.open('pyboy.state') as state_file:
                state_data = state_file.read()
                state = io.BytesIO(state_data)
                state.seek(0)
                self.pyboy.load_state(state)

            # Load the reward engine
            self.reward_engine = Engine(self.config)
            self.reward_engine.efficient_dict_load(zip_file)
        
            with zip_file.open('sim_stats.json') as f:
                sim_stats_dict = json.load(f)
            
        self.rom = sim_stats_dict["rom_path"]
        self.total_reward = sim_stats_dict["reward"]
        self.initial_reward = sim_stats_dict["reward"]
        self.step_count = 0
        
        # print("efficient load!")
    
    def get_preamble(self):
        if self.rank == 0:
            return [np.zeros(self.config['input_resolution'])] * (self.config['preamble_size'] + 1)
        return self.reward_engine.previous_data
