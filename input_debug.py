import numpy as np
import os
import glob
import io
import zipfile
from reward_engine import Engine
import time
import cv2
import tempfile

# This script is for testing the reward engine and check if the image is stitched correctly into the map

def load_all_at_once(npz_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(npz_file, 'r') as zip_ref:
            zip_ref.extract("dabug_inputs.npz", tmpdir)
            extracted_npz_path = os.path.join(tmpdir, "dabug_inputs.npz")

            with np.load(extracted_npz_path) as npz:
                data = [npz[name] for name in npz.files]
                return data

def open_checkpoint(dir):
    os.makedirs(dir, exist_ok=True) 
    files = glob.glob(os.path.join(dir, "*"))
    file_index = int(input(f"Select file from {len(files)} files: "))
    
    # Load the .npz file all at once by unzipping it first
    print(files[file_index])
    all_data = load_all_at_once(files[file_index])
    return all_data


# most of this is not used in this script
ep_length = 2048 * 10 // 8
env_config = {
            'rank': 0,
            'init_state': '',
            'checkpoint_path': f'',
            'img_path': f'',
            'gb_path': '',
            'input_resolution': (144,160),
            'headless': False, 
            'gbc_mode': True,
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

if __name__ == "__main__":
    path = '.\\examples\\PokemonRed\\PR_cnn_lstm_ep_pre\\checkpoints'
    engine = Engine(env_config)
    
    frames = open_checkpoint(path)
    i = 0
    print(len(frames))
    for frame in frames:
        start_gen = time.time()
        engine.append(frame, 0)
        end_append = time.time()
        engine.show()
        print(i)
        #input()
        time.sleep(0.1)
        i += 1

    print(engine.debug_data)
    time.sleep(1000)
    


    

